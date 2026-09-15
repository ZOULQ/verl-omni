# MiniCPM-o 4.5 Thinker GSPO (duplex) — example

Train **only the Thinker LLM** of `openbmb/MiniCPM-o-4_5` with GSPO, while the
**full three-stage network** (Thinker → Talker → Code2Wav) is deployed on both
the FSDP actor and the vLLM-Omni rollout. Inputs are multimodal (text / image /
audio); outputs are text **and** speech. The Talker and Code2Wav are frozen and
receive no gradient, but stay loaded so the checkpoint remains complete.

## What is wired up

| Piece | File | Role |
|---|---|---|
| Training adapter | `verl_omni/pipelines/minicpmo_4_5/thinker_training_adapter.py` | Whole-network load + freeze, Thinker text-logit forward |
| Rollout adapter | `verl_omni/pipelines/minicpmo_4_5/omni_rollout_adapter.py` | Full 3-stage topology; policy = stage 0 text, reward audio = stage 2 |
| Rollout resource | `verl_omni/utils/rollout_device_layout.py` | Computes replica world size from per-stage deploy config |
| Launch script | `run_minicpmo_4_5_thinker_gspo_lora.sh` | LoRA-mode GSPO recipe |

The policy is the Thinker's text/turn-token sequence on a **single text
vocabulary**; logprobs use the standard `logprobs_from_logits` path, and
`actor.policy_loss.loss_mode=gspo` + `algorithm.adv_estimator=grpo` enable
`compute_policy_loss_gspo` with **zero verl-core changes**.

## Prerequisites

- Model checkpoint at `MODEL_PATH` (`openbmb/MiniCPM-o-4_5`). The training
  adapter gates on the HF config's top-level `version == "4.5"` because
  `architectures=["MiniCPMO"]` is shared verbatim with MiniCPM-o 2.6.
- `vllm-omni` pinned to the commit in `.github/vllm_omni_pin.txt` (provides
  `MINICPMO_4_5_PIPELINE` and `vllm_omni/deploy/minicpmo_4_5.yaml`).
- `VERL_USE_EXTERNAL_MODULES=verl_omni` so Ray workers can import the adapters.

## Training modes

**LoRA (script default).** The whole network is frozen at the base; LoRA is
injected only into the Thinker LLM's `q/k/v/o_proj` via
`target_modules` + `exclude_modules` (excluding `tts` / `code2wav` / encoders).

**Full-parameter.** Drop `lora_rank` / `lora_alpha` / `lora_dtype` /
`target_modules` / `exclude_modules`. The adapter's `configure_model` freezes
everything except the Thinker LLM with `requires_grad=False`, so only the LLM
backbone updates. With FSDP1 this requires `use_orig_params=true` (the omni
engine already enforces this).

## Rollout GPU placement

The rollout replica world size is computed from the per-stage deploy config, not
from flat `tp × dp × pp`. The adapter defaults to
`vllm_omni/deploy/minicpmo_4_5.yaml` (three stages colocated on one
large-memory GPU → `W_total = 1`). For a multi-GPU layout, supply your own
profile and point at it:

```text
+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config="<abs>/<path>.yaml"
```

The resource allocator raises if the per-stage device union does not evenly
divide the total GPU count.

## Data

- Interleaved multimodal duplex conversation data (user audio/image units +
  reference audio + expected turn-taking + answer text).
- The dataset must expose the reference audio used for voice cloning; it flows
  through `multi_modal_data` under `_minicpmo45_reference_audio`.
- `QwenOmniRLHFDataset` is used as the multimodal RL dataset class; adjust
  `data.custom_cls` / `+data.mm_processor_kwargs` to your task.

## Reward

A **session-level** reward consumes the text and the stage-2 audio waveform
(exposed as `extra_fields["audio"]` / `["audio_sample_rate"]`):

```text
reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex_reward.py
reward.custom_reward_function.name=compute_score
```

Implement `compute_score` (ASR + judge, turn-taking alignment, interruption
latency, audio quality — see the design doc §9). For a first text-only smoke,
`verl_omni/utils/reward_score/choice_reward.py` (used by the Qwen3-Omni AVQA
recipe) can be substituted.

## Validation checklist

1. **Load smoke** — `OmniModelConfig` resolves `MiniCPMO` + `version=="4.5"`;
   the whole network FSDP-wraps; Talker/Code2Wav remain loaded with
   `requires_grad=False`.
2. **Rollout↔actor consistency** — `rollout_actor_probs_pearson_corr > 0.995`,
   `log_ppl_diff ≈ 0` on the Thinker text segment.
3. **GSPO numerics** — `actor/loss`, `actor/grad_norm`, `actor/pg_clipfrac`
   healthy; no OOM; gradients only on Thinker LLM params.
4. **End-to-end** — reward rises over steps; audio is produced per sample.

## Path 1 vs Path 2

Two rollout data planes are implemented, both training only the Thinker LLM with
the full three-stage network deployed.

**Path 1 (batch full pipeline)** — `run_minicpmo_4_5_thinker_gspo_lora.sh`.
The standard Thinker sampler returns logprobs; near-zero vLLM-Omni changes.
Select it with the default `omni_single_turn_agent`.

**Path 2 (native duplex data plane)** — `run_minicpmo_4_5_thinker_gspo_lora_duplex.sh`.
The rollout drives the in-process engine's native duplex control plane over the
server's `/v1/realtime` WebSocket endpoint (open → stream question → commit →
collect → close) via the `minicpmo_duplex_agent` agent loop.

| Piece | File |
|---|---|
| Duplex agent loop | `verl_omni/pipelines/minicpmo_4_5/duplex_agent_loop.py` (`@register("minicpmo_duplex_agent")`) |
| WebSocket duplex client | `verl_omni/workers/rollout/vllm_rollout/vllm_omni_duplex_client.py` (`run_duplex_training_session`) |
| vLLM-Omni §6.1 | `minicpmo_4_5_omni.py` — Thinker native-duplex sampler returns `LogprobsTensors` |
| vLLM-Omni §6.2 | `duplex/data_plane.py` — `project_output` forwards `thinker_token_ids` / `thinker_logprobs` |

The dataset must expose the question audio as the **first** entry and the
reference voice as the **second** entry of `multi_modal_data["audios"]` (both as
16 kHz mono f32-le PCM). See the design doc §5.2/§6.
