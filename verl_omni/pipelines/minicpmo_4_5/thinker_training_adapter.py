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
"""MiniCPM-o 4.5 Thinker training adapter (FSDP actor, whole-network deploy).

Trains **only the Thinker LLM** with GSPO while keeping the full three-stage
network (Thinker + Talker + Code2Wav) loaded.  Unlike the Qwen3-Omni Thinker
adapter — which strips the talker/codec to save memory — MiniCPM-o 4.5 keeps
every submodule in memory (``get_strip_modules() == []``) and instead freezes
the encoders, Talker, and Code2Wav with ``requires_grad=False`` so:

- the checkpoint stays complete (no inference-side weight back-fill), and
- actor <-> rollout parameter names stay one-to-one for weight sync.

Key MiniCPM-o 4.5 integration difference: the HF config reports
``architectures=["MiniCPMO"]`` (shared verbatim with 2.6) and is distinguished
from older checkpoints only by the top-level ``version == "4.5"``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)

# Thinker-LLM parameter prefixes that stay trainable. The HF checkpoint uses a
# *flat* layout (vLLM-Omni maps weights by top-level ``llm.``/``vpm.``/``tts.``
# prefixes); the ``thinker.*`` variants cover a nested remote-code layout.
_THINKER_LLM_PREFIXES = ("llm.", "thinker.llm.", "thinker.language_model.")

# The Thinker backbone is a Qwen2/Qwen3 LLM (vLLM-Omni picks
# ``Qwen3ForCausalLM`` when ``attention_bias is False``, else ``Qwen2ForCausalLM``).
_NO_SPLIT_MODULES = ["Qwen2DecoderLayer", "Qwen3DecoderLayer"]


def _is_thinker_llm(param_name: str) -> bool:
    """Whether *param_name* belongs to the trainable Thinker LLM backbone."""
    return param_name.startswith(_THINKER_LLM_PREFIXES)


@OmniModelBase.register("MiniCPMO", stage="thinker")
@OmniModelBase.register("MiniCPMO45OmniForConditionalGeneration", stage="thinker")
class MiniCPMO45ThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for MiniCPM-o 4.5.

    Loads the whole three-stage network, freezes the encoders / Talker /
    Code2Wav, and routes the text-logit forward through the Thinker so GSPO
    policy logprobs are computed on the single text vocabulary.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        # Whole-network deploy: keep Thinker + Talker + Code2Wav. Freezing is
        # done with requires_grad=False in configure_model, not by stripping.
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        version = str(getattr(getattr(module, "config", None) or model_config.hf_config, "version", ""))
        if version != "4.5":
            raise ValueError(
                f"MiniCPMO45ThinkerAdapter requires version=='4.5' (architectures=['MiniCPMO'] is shared "
                f"with MiniCPM-o 2.6); got version={version!r}. Point the model path at a 4.5 checkpoint."
            )

        module = super().configure_model(module, model_config)  # get_strip_modules() == [] -> no strip

        # Freeze everything except the Thinker LLM backbone. The encoders
        # (vpm/resampler/apm/audio_projection_layer), Talker (tts), and
        # Code2Wav stay in the network but receive no gradient.
        for name, param in module.named_parameters():
            param.requires_grad = _is_thinker_llm(name)

        # Route the text-logit forward and embedding accessors to the Thinker.
        thinker = getattr(module, "thinker", None)
        if thinker is not None:
            # Nested layout (Qwen3-Omni style): thinker wraps encoders + LLM.
            module.forward = thinker.forward
            module.get_input_embeddings = thinker.get_input_embeddings
            module.set_input_embeddings = thinker.set_input_embeddings
        else:
            # Flat layout (vLLM-Omni weight prefixes llm./vpm./resampler./apm./
            # audio_projection_layer./tts.): the model's own forward is the
            # Thinker text path (encoders + LLM), so keep it. Route embedding
            # accessors to the LLM backbone, which verl uses for LoRA / resize.
            llm = getattr(module, "llm", None)
            if llm is None:
                raise RuntimeError(
                    "MiniCPMO45ThinkerAdapter could not locate the Thinker text module "
                    "(neither `thinker` nor `llm` attribute found). Verify the submodule "
                    "names against the checkpoint's modeling_minicpmo45.py."
                )
            if hasattr(llm, "get_input_embeddings"):
                module.get_input_embeddings = llm.get_input_embeddings
            if hasattr(llm, "set_input_embeddings"):
                module.set_input_embeddings = llm.set_input_embeddings

        # FSDP sharding granularity for the trainable Qwen2/Qwen3 layers.
        # Add Talker / Code2Wav layer class names here after verifying them
        # against modeling_minicpmo45.py.
        module._no_split_modules = list(_NO_SPLIT_MODULES)
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the MiniCPM-o multimodal processor.

        MiniCPM-o ships a remote-code processor (``MiniCPMOProcessor``) that
        handles interleaved image/audio/video inputs and the text+speech chat
        template.  Bind any model-specific position helpers (e.g. rope/mrope
        index cast to int64, multimodal pad-token dedup) here after verifying
        them against the checkpoint's processing_minicpmo45.py / the Qwen3-Omni
        adapter's processor hook for the pattern.
        """
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        if getattr(processor, "chat_template", None) is None:
            logger.warning("MiniCPM-o processor has no chat_template; text-only prompts may not apply a template.")
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer, preferring a standalone ``chat_template.json``."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as file:
                tokenizer.chat_template = json.load(file)["chat_template"]
        return tokenizer
