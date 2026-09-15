# 多阶段（三阶段）模型 rollout 资源分配策略设计（verl-omni）

> 状态：设计文档（只读，未修改代码）。本方案全部落在 `verl-omni/`，不改 `verl/`、`vllm-omni/`。
> 动机：MiniCPM-o 4.5 三阶段（Thinker→Talker→Code2Wav）模型接入 GSPO 训练时，rollout 侧的 GPU 资源分配需要按 vllm-omni 的 per-stage deploy config（`vllm_omni/deploy/minicpmo_4_5.yaml`）计算，而不是按 rollout.yaml 的扁平 `tp×pp×dp`。

---

## 0. 摘要

verl-omni 现在用 rollout.yaml 的 `tensor_model_parallel_size × data_parallel_size × pipeline_model_parallel_size` 作为**每个 rollout replica 的 world size**（该公式硬编码在 verl core 的 `LLMServerManager` 与 `RolloutReplica` 里），并把**所有 stage 都铺到同一组设备 + 同一个 tp** 上。这对“每 stage 独立并行/设备”的三阶段模型是错的。

本方案在 verl-omni 侧做两件事，配合起来**零改动 verl core**：

1. **把 per-stage 布局解析出来**（来自用户提供的 vllm-omni deploy YAML，或 pipeline 默认 deploy config），并算出 `W_total = max(所有 stage 的 device id) + 1`（即一个完整三阶段 replica 需要的 GPU 数）。
2. **把 `W_total` 写回 `rollout.tensor_model_parallel_size`（`dp=1, pp=1`）**，让 verl core 里既有的 `tp×dp×pp` 公式**自然得到正确值**；同时改造 `_write_deploy_config`，让生成的 deploy config 按 stage 分别给出 `devices` / `tp` / `dp` / `pp` / `num_replicas`。

---

## 1. 现状与问题

### 1.1 verl core 的 world size 计算（不可改，但可被配置驱动）

`verl/workers/rollout/llm_server.py:409-435`（`LLMServerManager._initialize_llm_servers`）：

```python
rollout_world_size = (tp * dp * pp)          # 每个 replica 的 GPU 数
world_size = worker_group.world_size          # hybrid；standalone 则 = n_gpus_per_node * nnodes
num_replicas = world_size // rollout_world_size
# 每个 replica init_hybrid 时按 self.world_size 切片 worker_group
```

`verl/workers/rollout/replica.py:107-117`（`RolloutReplica.__init__`）：

```python
self.world_size = tp * dp * pp                # 与 manager 一致
self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
self.nnodes = self.world_size // self.gpus_per_replica_node
```

结论：**`tp × dp × pp` 是“每 replica world size”的唯一事实来源**，manager 和 replica 两处都用它，必须一致。

### 1.2 verl-omni 当前的 `_write_deploy_config`（扁平铺开）

`verl_omni/workers/rollout/vllm_rollout/vllm_omni_ar_strategy.py:178-204`：

```python
tp_size = self.server.config.tensor_model_parallel_size
device_count = len(visible_devices.split(","))
devices = ",".join(str(i) for i in range(device_count))   # 所有 stage 同一串设备
deploy_dict["stages"] = [
    {"stage_id": sid, "devices": devices, "tensor_parallel_size": tp_size,
     "text_encoder_tp_size": ..., "engine_extras": stage_extras[sid]}
    for sid in stage_ids
]
```

即：**所有 stage 共占同一组 `[0, device_count)` 设备、同一个扁平 tp**。三阶段模型每 stage 需要不同 tp/pp/设备（或显式 colocate）时，这套逻辑无法表达。

### 1.3 vllm-omni 的 per-stage deploy 语义（本方案对齐的“真值”）

`vllm_omni/deploy/minicpmo_4_5.yaml`：每个 stage 独立给出 `devices`（`"0"`）、`tensor_parallel_size`（stage0/1/2 均 `1`）、`max_num_seqs`、`gpu_memory_utilization` 等；三阶段 colocate 在 1 张卡上。

`vllm_omni/config/composable_parallel/apply.py:154-185`（`check_device_layout`）给出每 stage 的校验语义：

- 每 stage `world = tp × dp × pp`；
- 显式 `devices` 的数量必须 ∈ `{world, replicas × world}`（单 replica 模板或整 pool）；
- 不写 `devices` 时，默认 `devices = range(world)`（`config_factory.py:675`），即**从 GPU 0 开始 colocate**。

因此“一个完整三阶段 replica 需要的 GPU 数” = **所有 stage `devices` 的最大 id + 1**（`devices` 是 replica 内 `CUDA_VISIBLE_DEVICES` 的**相对、位置化**下标）。

---

## 2. 目标与非目标

### 目标
- rollout world size（每 replica GPU 数）由 **per-stage deploy config** 计算，而非扁平 `tp×dp×pp`。
- 生成的 deploy config 按 stage 分别携带 `devices`/`tp`/`dp`/`pp`/`num_replicas`。
- 全部实现位于 `verl-omni/`；不改 `verl/`、`vllm-omni/`。

### 非目标
- 不改 verl core 的 `LLMServerManager`/`RolloutReplica` 世界大小公式。
- 不支持跨节点单 replica（vllm-omni 当前单节点）；多 replica 只要求 `world_size % W_total == 0`。
- 不把 per-stage `gpu_memory_utilization`/`max_num_seqs` 等容量项回写进 rollout.yaml（它们继续由 deploy YAML 或 `engine_extras` 携带）。

---

## 3. 方案总览

```
                    ┌──────────────────────────────────────────────┐
                    │ 1. resolve_stage_device_layouts()             │
                    │    用户 deploy YAML 或 pipeline 默认 YAML       │
                    │    → 每 stage {devices, tp, dp, pp, replicas}  │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │ 2. W_total = max(stage device ids) + 1        │
                    └──────────────────────┬───────────────────────┘
                                           │
        ┌──────────────────────────────────┴──────────────────────────────────┐
        ▼                                                                      ▼
  3a. driver 侧配置归一化                                      3b. replica 侧生成 deploy config
  rollout.tensor_model_parallel_size = W_total                 `_write_deploy_config` 按 stage 写
  rollout.data_parallel_size = 1                                devices/tp/dp/pp/num_replicas
  rollout.pipeline_model_parallel_size = 1
        │                                                                      ▲
        ▼                                                                      │
  verl core `LLMServerManager`:                                     （replica 内用同一 resolve 函数，保证一致）
  num_replicas = world_size // W_total   ✓
  `RolloutReplica.world_size = W_total` ✓
```

要点：**3a 让 verl core 的既有 `tp×dp×pp` 公式“恰好”算出 `W_total`**（因为 `dp=pp=1`），因此 manager 与 replica 两处天然一致，无需触碰 verl core。3b 只需替换掉 `_write_deploy_config` 里“所有 stage 同 devices/tp”的扁平赋值。

---

## 4. 详细设计

### 4.1 per-stage 布局解析（新增 util + adapter 钩子）

新增 util（建议 `verl_omni/utils/rollout_device_layout.py`，或并入 `verl_omni/pipelines/`）：

```python
@dataclass
class StageDeviceLayout:
    stage_id: int
    devices: list[int] | None      # replica 内相对设备 id；None = 交给 vllm-omni 默认(range(world))
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    num_replicas: int              # 默认 1

def resolve_stage_device_layouts(
    *,
    pipeline_name: str,
    pipeline_mode: str,
    adapter_cls: type[OmniRolloutPipelineBase],
    deploy_config_path: str | None,
    fallback_tp: int,
) -> list[StageDeviceLayout]:
    ...
```

解析优先级（先到先得）：

1. **用户 deploy YAML**：`engine_kwargs.vllm_omni.deploy_config` 指向的路径（如 `minicpmo_4_5.yaml`）。用 vllm-omni 的 `load_deploy_config(path)` 读取 `stages[].devices / tensor_parallel_size / data_parallel_size / pipeline_parallel_size / num_replicas`。
2. **pipeline 默认 deploy YAML**：`OmniRolloutPipelineBase` 新增可选方法 `default_deploy_config_path(pipeline_mode) -> str | None`，返回该 pipeline 在 vllm-omni `deploy/` 下的默认 YAML（如 `minicpmo_4_5` 返回 `.../vllm_omni/deploy/minicpmo_4_5.yaml`）。
3. **扁平回退（向后兼容）**：都没有时，退化为现状——每个 stage `devices=None`、`tp=fallback_tp`、`dp=1`、`pp=1`（等价于“所有 stage 共占 `[0, tp)` 且同 tp”）。这样 Qwen3-Omni 等既有单/双 stage 用法行为不变。

`devices` 解析：`"0,1,2"` / `["0","1","2"]` → `[0,1,2]`；空/缺省 → `None`。

### 4.2 计算 `W_total`

```python
def compute_rollout_world_size(layouts: list[StageDeviceLayout]) -> int:
    max_id = 0
    for layout in layouts:
        if layout.devices is not None:
            max_id = max(max_id, *layout.devices)
        else:
            # 不写 devices → vllm-omni 默认 range(tp*dp*pp)，占据 [0, world)
            world = layout.tensor_parallel_size * layout.data_parallel_size * layout.pipeline_parallel_size
            max_id = max(max_id, world - 1)
    return max_id + 1
```

同时做两级校验（复用 vllm-omni 语义，避免在 spawn 时才爆）：

- 每 stage：`len(devices_i) ∈ {world_i, replicas_i × world_i}`（对齐 `check_device_layout`）。
- 全局：`world_size % W_total == 0`（否则 `num_replicas` 会截断、留空 GPU），给出清晰报错。
- （可选）提醒：`devices` 是 replica 内相对下标；跨 stage 共享设备即 colocate，计入同一 replica。

### 4.3 注入 verl core 的 world size（配置归一化）

新增 util：

```python
def normalize_multistage_rollout_world_size(config) -> int:
    """读 pipeline_name/deploy_config → 算 W_total → 写回 rollout.tp/dp/pp，返回 W_total。"""
    rcfg = config.actor_rollout_ref.rollout
    engine_kwargs = rcfg.engine_kwargs.vllm_omni
    layouts = resolve_stage_device_layouts(..., engine_kwargs.get("deploy_config"), fallback_tp=rcfg.tensor_model_parallel_size)
    w_total = compute_rollout_world_size(layouts)
    # 归一化：让 verl core 的 tp*dp*pp == W_total
    rcfg.tensor_model_parallel_size = w_total
    rcfg.data_parallel_size = 1
    rcfg.pipeline_model_parallel_size = 1
    return w_total
```

为什么这样安全（已核对 verl core 各使用点）：

- `llm_server.py:409` `rollout_world_size = W_total`，`num_replicas = world_size // W_total` ✓。
- `replica.py:107` `self.world_size = W_total`，`init_hybrid` 按 `W_total` 切片 worker ✓。
- `vllm_async_server.py:382` 的 `assert gpus_per_node % tp == 0` 仅在 `data_parallel_size > 1` 时触发；归一化后 `dp=1`，该分支被跳过 ✓。
- verl-omni 自己的 `_write_deploy_config:180` 之前读 `self.server.config.tensor_model_parallel_size` 当“每 stage tp”，这正是本次要替换掉的逻辑（见 4.4），不会再误用 `W_total` 当 stage tp ✓。

### 4.4 改造 `_write_deploy_config`

把 `vllm_omni_ar_strategy.py:178-204` 的扁平赋值替换为按 stage 输出：

```python
layouts = resolve_stage_device_layouts(
    pipeline_name=pipeline_name,
    pipeline_mode=pipeline_mode,
    adapter_cls=adapter_cls,
    deploy_config_path=engine_kwargs.get("deploy_config"),
    fallback_tp=self.server.config.tensor_model_parallel_size,  # 已是 W_total（4.3 归一化后）
)
deploy_dict["stages"] = [
    {
        "stage_id": sid,
        "devices": ",".join(str(d) for d in layout.devices) if layout.devices is not None else None,
        "tensor_parallel_size": layout.tensor_parallel_size,
        "data_parallel_size": layout.data_parallel_size,
        "pipeline_parallel_size": layout.pipeline_parallel_size,
        "num_replicas": layout.num_replicas,
        "engine_extras": stage_extras[sid],
    }
    for sid, layout in zip(stage_ids, layouts, strict=True)
]
```

要点：

- 生成的 deploy YAML 与用户/默认 deploy YAML 的 stage 布局一致；verl-omni 只是把它**重写成临时文件**并接上 `engine_extras`（`max_model_len`/`max_num_batched_tokens` 等容量项仍按现有 `stage_extras` 逻辑合并）。
- `devices` 用 replica 内相对下标，落在 `[0, W_total)`；replica 的 `CUDA_VISIBLE_DEVICES` 已由 verl 在 hybrid/standalone 里设好。
- `num_replicas` 仅在 stage 级并行（vllm-omni stage_replica）时 >1；默认写 1 或省略均可，避免与 orchestrator 的 replica 语义混淆。

### 4.5 adapter 钩子（`OmniRolloutPipelineBase`）

在 `verl_omni/pipelines/model_base.py` 的 `OmniRolloutPipelineBase` 增加两个可空/默认方法：

```python
def default_deploy_config_path(self, pipeline_mode: str) -> str | None:
    return None  # 默认无；MiniCPMO45 adapter 覆写为 vllm_omni/deploy/minicpmo_4_5.yaml
```

`MiniCPMO45RolloutAdapter`（`verl_omni/pipelines/minicpmo_4_5/omni_rollout_adapter.py`）覆写该方法返回 `minicpmo_4_5.yaml` 路径，使“用户不显式给 `deploy_config` 时”也能按三阶段布局计算。

---

## 5. 两条训练路径的接入点

### 5.1 V1 PPO（GSPO/GRPO，MiniCPM-o 4.5 主路径）

`verl_omni/trainer/main_omni.py` 的 `run_omni`（`uses_v1_trainer` 分支）在 `run_ppo(config, ...)` **之前**调用：

```python
normalize_multistage_rollout_world_size(config)   # 写回 rollout.tp/dp/pp
run_ppo(config, task_runner_class=TaskRunnerV1)
```

`run_ppo → TaskRunnerV1 → ray_trainer.py:951 LLMServerManager.create` 会读归一化后的 `tp=W_total, dp=1, pp=1`，从而 `num_replicas = world_size // W_total` 正确。

### 5.2 verl-omni 自有 trainer（diffusion / direct-preference）

这些 trainer 在 verl-omni 里直接 `LLMServerManager.create(...)`，在其**之前**调用同一 `normalize_multistage_rollout_world_size(config)`：

- `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:937`
- `verl_omni/trainer/diffusion/v1/trainer_base.py:914`
- `verl_omni/trainer/diffusion/v1/trainer_separate_async.py:178/197`

归一化集中在 util 里，两处调用同一函数，避免重复。

---

## 6. 文件清单（全部 verl-omni）

| 文件 | 动作 | 内容 |
|---|---|---|
| `verl_omni/utils/rollout_device_layout.py` | 新增 | `StageDeviceLayout`、`resolve_stage_device_layouts`、`compute_rollout_world_size`、`normalize_multistage_rollout_world_size` |
| `verl_omni/pipelines/model_base.py` | 修改 | `OmniRolloutPipelineBase` 增加 `default_deploy_config_path(pipeline_mode)`（默认 `None`） |
| `verl_omni/pipelines/minicpmo_4_5/omni_rollout_adapter.py` | 修改 | 覆写 `default_deploy_config_path` → `minicpmo_4_5.yaml` |
| `verl_omni/workers/rollout/vllm_rollout/vllm_omni_ar_strategy.py` | 修改 | `_write_deploy_config` 按 stage 输出 `devices/tp/dp/pp/num_replicas` |
| `verl_omni/trainer/main_omni.py` | 修改 | V1 分支 `run_ppo` 前调用归一化 |
| `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` / `v1/trainer_base.py` / `v1/trainer_separate_async.py` | 修改 | `LLMServerManager.create` 前调用归一化 |
| `examples/gspo_trainer/minicpmo_4_5/run_*.sh` + `README.md` | 修改 | 文档说明 `deploy_config`/`pipeline_name` 与 `W_total` 的语义；移除“扁平 tp 即 stage tp”的误导 |

> 不改 `verl/`、`vllm-omni/`。verl core 的 `tp×dp×pp` 公式原样复用。

---

## 7. 校验与测试

1. **单 stage 回归（Qwen3-Omni thinker）**：不提供 `deploy_config` 时走扁平回退，`W_total == 原 tp`，生成 YAML 与现状逐字段一致（golden diff）。
2. **三阶段 colocate（minicpmo_4_5 单卡）**：`deploy_config=minicpmo_4_5.yaml`（三 stage `devices:"0"`、tp=1）→ `W_total=1`；`num_replicas = world_size // 1`；生成 YAML 每 stage `devices:"0"`、tp=1。
3. **三阶段分卡（多卡模板）**：stage0 `devices:"0,1,2,3"`/tp=4、stage1 `devices:"4,5"`/tp=2、stage2 `devices:"6,7"`/tp=2 → `W_total=8`；生成 YAML 保留各 stage 的 devices/tp。
4. **非法布局报错**：`world_size % W_total != 0`、或某 stage `devices` 数量 ∉ {world, replicas×world} → 在配置阶段清晰报错（不等到 vllm-omni spawn）。
5. **端到端 smoke**：MiniCPM-o 4.5 GSPO Path 1（batch 全 pipeline）跑通，确认 `LLMServerManager` 打印的 replica 数 = `n_gpus_per_node * nnodes // W_total`，且 vllm-omni 按三阶段布局启动。

---

## 8. 边界与风险

- **`devices` 语义约定**：deploy YAML 里的 `devices` 必须是 **replica 内相对、0 起** 的下标；跨 stage 共享下标即 colocate。文档与报错里显式写明，避免用户按物理卡号写。
- **`W_total` 与 `gpus_per_node`**：vllm-omni 单节点，`W_total` 需 ≤ 单节点可用卡数；hybrid 切片按 `W_total` 平切，多节点跨 replica 由 `num_replicas` 承担。
- **`tp` 语义被占用**：归一化后 `rollout.tensor_model_parallel_size` 不再代表“某 stage 的 tp”，而是“每 replica 总卡数”。因此 `_write_deploy_config` 必须改用 per-stage 布局（4.4），任何仍读该字段当 stage tp 的旧代码都要清掉（当前仅 `vllm_omni_ar_strategy.py:180` 一处）。
- **stage 级 `num_replicas`（vllm-omni stage_replica）与 verl 的 replica 是两个概念**：前者是某 stage 的 intra-pipeline 副本，后者是 verl 的整 pipeline 副本。归一化只驱动后者；前者由 deploy YAML 的 `num_replicas` 字段直接透传，且要求 `devices` 数量 = `world × num_replicas`（对齐 `check_device_layout`）。
- **向后兼容**：无 `deploy_config` 时完全回退到现状（所有 stage 同 devices/tp），既有单/双 stage 模型零行为变化。
