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
"""Per-stage rollout resource allocation for multi-stage (omni) pipelines.

verl core computes each rollout replica's world size from the flat
``tensor_parallel_size * data_parallel_size * pipeline_parallel_size`` fields of
``RolloutConfig``.  That flat formula is wrong for multi-stage vLLM-Omni
pipelines (e.g. MiniCPM-o 4.5 thinker/talker/code2wav) whose deploy config
declares an independent ``devices`` / ``tensor_parallel_size`` per stage.

This module resolves that per-stage layout from a vLLM-Omni deploy YAML and
derives ``W_total = max(stage device id) + 1`` — the number of GPUs one full
replica needs.  Callers then fold ``W_total`` back into ``RolloutConfig`` as
``tensor_parallel_size`` (with ``data_parallel_size=1`` and
``pipeline_parallel_size=1``) so verl core's existing formula yields the correct
value without touching verl core.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

logger = logging.getLogger(__name__)


@dataclass
class StageDeviceLayout:
    """Per-stage GPU placement resolved from a vLLM-Omni deploy config.

    ``devices`` are replica-relative ids (positions within the replica's
    ``CUDA_VISIBLE_DEVICES``).  ``None`` means "let vLLM-Omni default it to
    ``range(world_size)``", which is equivalent to colocating the stage from
    device 0.
    """

    stage_id: int
    devices: list[int] | None = None
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_replicas: int = 1

    @property
    def world_size(self) -> int:
        """Per-stage engine world size (tp * dp * pp)."""
        return self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size

    @property
    def devices_str(self) -> str | None:
        """Comma-joined device ids, or ``None`` when unset."""
        if self.devices is None:
            return None
        return ",".join(str(device_id) for device_id in self.devices)


def _parse_devices(value: Any) -> list[int] | None:
    """Parse a ``devices`` value into a list of ints, or ``None`` when unset."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(device_id) for device_id in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _load_deploy_config(
    adapter_cls: "type[OmniRolloutPipelineBase]",
    pipeline_mode: str,
    deploy_config_path: str | None,
) -> Any | None:
    """Load the vLLM-Omni deploy config for *pipeline_mode*, or ``None``.

    Prefers the explicit *deploy_config_path*, then falls back to the adapter's
    ``default_deploy_config_path``.  ``None`` signals "no per-stage layout
    available; use the flat fallback".
    """
    path = deploy_config_path
    if not path:
        path = adapter_cls.default_deploy_config_path(pipeline_mode)
    if not path:
        return None

    from vllm_omni.config import load_deploy_config

    return load_deploy_config(path)


def resolve_stage_device_layouts(
    *,
    adapter_cls: "type[OmniRolloutPipelineBase]",
    pipeline_mode: str,
    deploy_config_path: str | None,
    fallback_tensor_parallel_size: int,
    stage_ids: list[int],
) -> list[StageDeviceLayout]:
    """Resolve a per-stage device layout for each id in *stage_ids*.

    When no deploy config is available, every stage falls back to the legacy
    flat layout (``devices=None``, ``tp=fallback_tensor_parallel_size``), which
    reproduces the previous "all stages share the same device range + tp"
    behavior.  When a deploy config exists but omits a stage id, that stage also
    falls back to the flat default.
    """
    deploy = _load_deploy_config(adapter_cls, pipeline_mode, deploy_config_path)

    if deploy is None:
        return [
            StageDeviceLayout(
                stage_id=stage_id,
                devices=None,
                tensor_parallel_size=fallback_tensor_parallel_size,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                num_replicas=1,
            )
            for stage_id in stage_ids
        ]

    # data_parallel_size / pipeline_parallel_size are pipeline-wide in the
    # deploy schema; tensor_parallel_size / devices / num_replicas are per-stage.
    pipeline_dp = int(getattr(deploy, "data_parallel_size", None) or 1)
    pipeline_pp = int(getattr(deploy, "pipeline_parallel_size", None) or 1)
    deploy_by_id = {int(stage.stage_id): stage for stage in deploy.stages}

    layouts: list[StageDeviceLayout] = []
    for stage_id in stage_ids:
        stage = deploy_by_id.get(stage_id)
        if stage is None:
            layouts.append(
                StageDeviceLayout(
                    stage_id=stage_id,
                    devices=None,
                    tensor_parallel_size=fallback_tensor_parallel_size,
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    num_replicas=1,
                )
            )
            continue
        layouts.append(
            StageDeviceLayout(
                stage_id=stage_id,
                devices=_parse_devices(stage.devices),
                tensor_parallel_size=int(stage.tensor_parallel_size or 1),
                data_parallel_size=pipeline_dp,
                pipeline_parallel_size=pipeline_pp,
                num_replicas=int(stage.num_replicas or 1),
            )
        )
    return layouts


def compute_rollout_world_size(layouts: list[StageDeviceLayout]) -> int:
    """Return the replica world size implied by the per-stage layouts.

    A stage without explicit ``devices`` defaults (in vLLM-Omni) to
    ``range(tp * dp * pp)``, i.e. it occupies ids ``[0, world_size)``.
    """
    max_device_id = 0
    for layout in layouts:
        if layout.devices:
            max_device_id = max(max_device_id, max(layout.devices))
        else:
            max_device_id = max(max_device_id, layout.world_size - 1)
    return max_device_id + 1


def validate_stage_device_layouts(layouts: list[StageDeviceLayout], total_gpus: int) -> int:
    """Validate per-stage device counts and return the replica world size.

    Mirrors vLLM-Omni's ``check_device_layout`` semantics: an explicit
    ``devices`` count must equal either the stage's world size (a per-replica
    template) or ``world * num_replicas`` (the full stage-replica pool).  Also
    verifies that the total world size evenly divides *total_gpus* so verl's
    ``num_replicas = world_size // rollout_world_size`` does not truncate.
    """
    for layout in layouts:
        if layout.devices is None:
            continue
        world = layout.world_size
        replicas = max(int(layout.num_replicas or 1), 1)
        count = len(layout.devices)
        if count not in (world, replicas * world):
            raise ValueError(
                f"stage {layout.stage_id}: declared {count} device(s) but the stage world size is "
                f"{world} (tp={layout.tensor_parallel_size} * dp={layout.data_parallel_size} "
                f"* pp={layout.pipeline_parallel_size}). Provide {world} (per-replica) or "
                f"{replicas * world} (num_replicas={replicas})."
            )

    world_size = compute_rollout_world_size(layouts)
    if total_gpus > 0 and total_gpus % world_size != 0:
        raise ValueError(
            f"rollout replica world size ({world_size}) must evenly divide the total GPU count "
            f"({total_gpus}); otherwise num_replicas would be truncated and leave GPUs idle."
        )
    return world_size


def normalize_multistage_rollout_world_size(config) -> int:
    """Fold a per-stage deploy layout into ``rollout.tensor_model_parallel_size``.

    Resolves the pipeline adapter from ``rollout.engine_kwargs.vllm_omni``,
    computes the replica world size from the pipeline's deploy config, and
    rewrites the flat rollout parallelism as ``tp=W_total, dp=1, pp=1`` so
    verl core's ``tp * dp * pp`` formula yields the correct per-replica size.

    Returns the original ``tensor_model_parallel_size`` unchanged when no
    multi-stage rollout adapter is configured (a no-op for plain single-stage
    and diffusion rollouts).
    """
    from omegaconf import OmegaConf

    from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

    rollout_cfg = config.actor_rollout_ref.rollout
    fallback_tp = int(rollout_cfg.tensor_model_parallel_size)

    omni_kwargs = OmegaConf.select(config, "actor_rollout_ref.rollout.engine_kwargs.vllm_omni", default=None)
    omni_kwargs = OmegaConf.to_container(omni_kwargs, resolve=True) if omni_kwargs is not None else None
    omni_kwargs = omni_kwargs or {}

    pipeline_name = omni_kwargs.get("pipeline_name")
    adapter_cls = OmniRolloutPipelineBase.get_class(pipeline_name) if pipeline_name else None
    if adapter_cls is None:
        return fallback_tp

    pipeline_mode = omni_kwargs.get("pipeline_mode") or "thinker_only"
    deploy_config_path = omni_kwargs.get("deploy_config")

    stages = adapter_cls.build_stage_configs(pipeline_mode=pipeline_mode)
    stage_ids = [stage.stage_id for stage in stages]

    layouts = resolve_stage_device_layouts(
        adapter_cls=adapter_cls,
        pipeline_mode=pipeline_mode,
        deploy_config_path=deploy_config_path,
        fallback_tensor_parallel_size=fallback_tp,
        stage_ids=stage_ids,
    )

    # Enforce the total-GPU divisibility check only when a real per-stage deploy
    # config drove the layout.  The flat fallback keeps verl core's historical
    # (silent) truncation behavior for single-stage rollouts whose tp does not
    # divide the total GPU count.
    used_deploy_config = bool(deploy_config_path or adapter_cls.default_deploy_config_path(pipeline_mode))
    total_gpus = 0
    if used_deploy_config:
        total_gpus = int(OmegaConf.select(config, "trainer.n_gpus_per_node", default=0) or 0) * int(
            OmegaConf.select(config, "trainer.nnodes", default=0) or 0
        )
    world_size = validate_stage_device_layouts(layouts, total_gpus)

    rollout_cfg.tensor_model_parallel_size = world_size
    rollout_cfg.data_parallel_size = 1
    rollout_cfg.pipeline_parallel_size = 1
    logger.info(
        "Normalized rollout world size from per-stage deploy config: pipeline=%r mode=%r "
        "tensor_model_parallel_size=%s data_parallel_size=1 pipeline_parallel_size=1.",
        pipeline_name,
        pipeline_mode,
        world_size,
    )
    return world_size
