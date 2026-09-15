# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniCPM-o 4.5 rollout pipeline adapter.

Deploys the full three-stage pipeline (Thinker -> Talker -> Code2Wav) for
Thinker-LLM GSPO training:

- Stage 0 (Thinker, ``final_output_type="text"``) is the policy: text / turn
  tokens with per-token logprobs on a single text vocabulary.
- Stage 1 (Talker) emits codec tokens and is a frozen intermediate stage.
- Stage 2 (Code2Wav, ``final_output_type="audio"``) produces the waveform that
  the session-level reward consumes.

Only Stage 0 receives actor weights (``weight_sync_stage_ids=[0]``); the Talker
and Code2Wav stay frozen on the rollout side, mirroring the actor's freeze.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import vllm_omni
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

_PIPELINE_ID = MINICPMO_4_5_PIPELINE.model_type  # "minicpmo_4_5"
_DEPLOY_DIR = Path(vllm_omni.__file__).resolve().parent / "deploy"


def _extract_hidden_states(output: Any) -> torch.Tensor | None:
    """Return the stage-0 Thinker last-layer hidden states, or ``None``.

    Batch-path semantics: the vLLM-Omni client multimodal channel already
    accumulated per-step ``hidden_states`` along dim 0 (``CONCAT_DIM0``) into a
    full ``[seq_len, hidden_dim]`` tensor, so this returns it as-is.

    The client multimodal channel only carries ``hidden_states`` when the
    rollout deploy enables ``hf_overrides.return_thinker_hidden_states``; when
    disabled the key is absent and this returns ``None`` (no policy surface
    changes).
    """
    multimodal = getattr(output, "multimodal_output", None)
    if not isinstance(multimodal, dict):
        return None
    hidden = multimodal.get("hidden_states")
    if hidden is None:
        return None
    tensor = torch.as_tensor(hidden)
    if tensor.numel() == 0:
        return None
    return tensor.detach().cpu()


@OmniRolloutPipelineBase.register(_PIPELINE_ID)
class MiniCPMO45RolloutAdapter(OmniRolloutPipelineBase):
    """Rollout topology for MiniCPM-o 4.5 (full 3-stage pipeline).

    Registered under ``model_type="minicpmo_4_5"``.  The stage topology comes
    unchanged from vLLM-Omni's ``MINICPMO_4_5_PIPELINE`` — no topology is
    duplicated in verl-omni.
    """

    @classmethod
    def _check_mode(cls, pipeline_mode: str) -> None:
        if pipeline_mode != "full":
            raise ValueError(
                f"MiniCPM-o 4.5 GSPO supports only pipeline_mode='full' "
                f"(Thinker + Talker + Code2Wav), got {pipeline_mode!r}."
            )

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "full") -> list:
        """Return the frozen three-stage topology (Thinker -> Talker -> Code2Wav)."""
        cls._check_mode(pipeline_mode)
        stages = list(MINICPMO_4_5_PIPELINE.stages)
        # Guard against upstream changes that silently add/remove stages.
        if len(stages) != 3:
            raise RuntimeError(
                f"Expected 3 stages in the MiniCPM-o 4.5 pipeline, got {len(stages)}. "
                "vLLM-Omni may have changed the pipeline definition."
            )
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "full") -> str:
        cls._check_mode(pipeline_mode)
        return _PIPELINE_ID

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode: str = "full") -> list[int]:
        """Sync actor weights only to Stage 0 (Thinker); Talker/Code2Wav stay frozen."""
        cls._check_mode(pipeline_mode)
        return [0]

    @classmethod
    def policy_stage_id(cls, pipeline_mode: str = "full") -> int:
        """The Thinker (stage 0) text tokens define the RL policy."""
        cls._check_mode(pipeline_mode)
        return 0

    @classmethod
    def default_deploy_config_path(cls, pipeline_mode: str = "full") -> str | None:
        """Point at vLLM-Omni's per-stage deploy profile for MiniCPM-o 4.5.

        This profile declares an independent ``devices`` / ``tensor_parallel_size``
        per stage, which the rollout resource allocator reads to compute the
        replica world size (see
        ``verl_omni.utils.rollout_device_layout``).
        """
        cls._check_mode(pipeline_mode)
        name = MINICPMO_4_5_PIPELINE.default_deploy_config_name
        if not name:
            return None
        return str(_DEPLOY_DIR / name)

    @classmethod
    def combine_engine_outputs(cls, outputs: list, prompt: dict) -> tuple[Any, dict[str, Any]]:
        """Combine Stage 0 text (policy) with Stage 2 audio (reward input).

        Stage 1 codec tokens are an intermediate artifact of the audio chain and
        are intentionally not collected: the Talker is frozen, so its codec
        logprobs are not part of the policy and the codec trajectory does not
        need to be replayed.
        """
        policy_outputs = [output for output in outputs if getattr(output, "stage_id", None) == 0]
        audio_outputs = [output for output in outputs if getattr(output, "stage_id", None) == 2]
        if not policy_outputs:
            raise RuntimeError("MiniCPM-o 4.5 rollout produced no stage-0 (Thinker) policy output.")
        if not audio_outputs:
            raise RuntimeError("MiniCPM-o 4.5 rollout produced no stage-2 (Code2Wav) audio output.")

        policy_output = policy_outputs[-1]
        audio_output = audio_outputs[-1]
        if len(getattr(policy_output, "outputs", [])) != 1:
            raise RuntimeError(
                f"MiniCPM-o 4.5 stage 0 must return exactly one completion, "
                f"got {len(getattr(policy_output, 'outputs', []))}."
            )

        try:
            multimodal_output = audio_output.multimodal_output
            # Code2Wav is a ``final_output_type="audio"`` stage; the per-request
            # output processor flattens the model-level "model_outputs" list into
            # the "audio" key, mirroring Qwen3-TTS.
            audio = multimodal_output.get("audio", multimodal_output.get("model_outputs"))
            if audio is None:
                raise KeyError("audio")
            waveform = torch.as_tensor(audio).detach().cpu().float()
            sample_rate = torch.as_tensor(multimodal_output["sr"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "MiniCPM-o 4.5 Code2Wav output does not match the pinned audio contract "
                "(expected multimodal_output['audio'] + ['sr'])."
            ) from error

        if waveform.ndim != 1:
            raise RuntimeError("MiniCPM-o 4.5 Code2Wav must return a one-dimensional mono waveform.")
        if waveform.numel() == 0:
            raise RuntimeError("MiniCPM-o 4.5 Code2Wav returned an empty waveform.")
        if sample_rate.numel() != 1:
            raise RuntimeError("MiniCPM-o 4.5 Code2Wav must return one scalar sample rate.")
        sample_rate_value = float(sample_rate.item())
        if sample_rate_value <= 0 or not sample_rate_value.is_integer():
            raise RuntimeError(f"MiniCPM-o 4.5 Code2Wav returned an invalid sample rate: {sample_rate_value!r}.")

        fields = {
            "audio": waveform,
            "audio_sample_rate": int(sample_rate_value),
        }

        # Layer-1 (optional): expose the Thinker's last-layer per-token hidden
        # states when the rollout deploy enables ``return_thinker_hidden_states``.
        thinker_hidden = _extract_hidden_states(policy_output)
        if thinker_hidden is not None:
            fields["thinker_hidden_states"] = thinker_hidden

        return policy_output, fields
