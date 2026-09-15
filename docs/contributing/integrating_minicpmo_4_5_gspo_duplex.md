# MiniCPM-o 4.5 双工 GSPO 训练实现方案（只训 Thinker LLM，冻结 Talker/Code2Wav，整网部署）

> 状态：设计文档（只读，未修改代码）。
> 训练范围：**只训练 Thinker LLM**；**冻结** 视觉/音频编码器、Talker、Code2Wav。
> 部署范围：**训练侧（FSDP actor）与 rollout 侧都部署完整三阶段整网模型**（Thinker + Talker + Code2Wav）。输入是多模态（文本/图像/音频），输出是文本 + 语音；Talker/Code2Wav 是整网模型不可分割的部分，只冻结不剥离。
> 约束：训练后端 FSDP（`model_type="omni_model"`），推理后端 vllm-omni；verl-omni 代码全部落在 `verl-omni/`，vllm-omni 侧**允许适量修改以返回 Thinker logprob**，不改 `verl/`、`vllm/`。

---

## 0. 摘要

### 0.1 目标

用 **GSPO** 训练 MiniCPM-o 4.5（`openbmb/MiniCPM-o-4_5`，any-to-any 原生全双工模型）的 **Thinker LLM**。输入多模态，输出文本 + 语音：采样侧跑**完整三阶段双工管线**（Thinker→Talker→Code2Wav）得到文本 + 音频，reward 评估整段双工会话，梯度只回传到 Thinker LLM。

### 0.2 训练/部署范围裁定

| 组件 | 状态 | 说明 |
|---|---|---|
| 整网模型 `MiniCPMO45OmniForConditionalGeneration` | **两侧都部署** | actor 与 rollout 都加载完整三阶段，不剥离任何子模块 |
| Thinker **LLM**（Qwen2/3 主干） | **可训** | GSPO policy；产出文本 + 回合控制 token |
| 视觉/音频编码器（vpm/apm/3D resampler） | 冻结 | `requires_grad=False` |
| Talker（MiniCPMTTS codec AR） | 冻结 | **保留在整网内**，`requires_grad=False` |
| Code2Wav（声码器） | 冻结 | **保留在整网内**，`requires_grad=False` |

### 0.3 一句话结论

policy 就是 **Thinker 文本/回合 token 的单序列（单文本词表）**，`log_prob` 走标准 `logprobs_from_logits`，GSPO/`ppo_loss`/engine/agent loop **全部零改动复用**。**训练模型与 rollout 模型都部署完整三阶段整网**：rollout 跑完整三阶段产出文本 + 音频供 reward；actor 加载整网、`requires_grad=False` 冻结编码器/Talker/Code2Wav、只更新 Thinker LLM。与 Qwen3-Omni Thinker GSPO 的差异：**整网部署（不 strip）**、reward 看到音频、以及 MiniCPM-o 4.5 的 remote-code 装载 / `version` 门禁 / 子模块命名差异。

### 0.4 范围与非目标

- **范围内**：Thinker LLM GSPO、整网部署（actor + rollout）、整段双工会话 reward、vllm-omni Thinker logprob 返回、示例脚本与验证。
- **非目标**：训练 Talker/Code2Wav/编码器（本次明确冻结）；联合词表/联合策略（已废弃）；修改 verl 核心。

---

## 1. 背景

### 1.1 模型与双工要点（来自 vllm-omni 与根目录 duplex 文档）

| 项 | 值 |
|---|---|
| HF 仓库 / config | `openbmb/MiniCPM-o-4_5`；`model_type="minicpmo"`、`architectures=["MiniCPMO"]`、顶层 `version="4.5"` |
| vllm-omni 模型类 | `MiniCPMO45OmniForConditionalGeneration`；子模块 `thinker`（`model_stage="llm"`）/`talker`（`model_stage="tts"`）/`code2wav`（独立 `MiniCPMO45Code2Wav`） |
| Thinker | SigLIP 视觉塔 + 流式 Whisper 音频编码器（10 音频 token/s）+ 3D Resampler + Qwen2/3 主干；输出文本/回合 token + `tts_hidden_states` |
| Talker | MiniCPMTTS 连续 AR，单码本 `num_vq==1`，codec 词表 6562/EOS 6561；双工 chunk = 25 帧 + 终止 = 26 token |
| Code2Wav | codec → 24kHz 波形（确定性声码器） |
| 原生双工 | 模型 token 原生决定听/说；1s（16kHz）一个音频 unit append；打断靠 epoch 栅栏 |

### 1.2 为什么两侧都要部署整网

- **模型是一个 any-to-any 整体**：多模态输入经 Thinker 出文本/回合 token，再经 Talker→Code2Wav 出音频。rollout 侧必须整网才能产出语音；训练侧也必须整网，否则：
  - checkpoint 不完整，推理/上线时要回填 Talker/Code2Wav 权重；
  - actor↔rollout 参数名与结构不一致，权重同步与后续多阶段扩展都会踩坑。
- **只更新 Thinker LLM**：整网都在内存/图上，但 `requires_grad=False` 冻结其余组件，反向传播与优化器只作用于 Thinker LLM（或只把 LoRA 注入 Thinker LLM，见 §8）。

### 1.3 RL 视角：policy 与 environment

- **policy = Thinker LLM**：动作是文本/回合 token 序列（`<|listen|>/<|speak|>/<|tts_bos|>/<|chunk_eos|>/<|turn_eos|>` + 文本）。
- **environment = 冻结的 Talker + Code2Wav**（含 Talker 采样随机性）：把“说了什么”解码成音频，供 reward 评估。
- 一个双工会话 = 一个会话级 reward（GRPO 分组），优势广播到 Thinker 序列。

---

## 2. 总体架构

```
              ┌───────────────────────────────────────────────────────────────┐
              │  vllm-omni rollout（完整 3-stage：Thinker→Talker→Code2Wav）      │
              │  · duplex data plane（每秒 append）或 batch 全 pipeline           │
              │  · 产出：stage0 thinker token + logprob（policy）                 │
              │         stage1 codec、stage2 audio（reward 用）                   │
              └─────────────────────────┬─────────────────────────────────────┘
                                        │ TokenOutput(response_ids/logprobs) + extra_fields{audio,...}
                                        ▼
              ┌───────────────────────────────────────────────┐
              │  single_turn_agent（文本 policy，无需映射）       │
              │  reward(会话级) = f(text, audio, turn-taking)   │
              └─────────────────────────┬─────────────────────┘
                                        │ GRPO 分组 → advantages（广播到 thinker 序列）
                                        ▼
              ┌───────────────────────────────────────────────┐
              │  FSDP actor：整网（编码器+Thinker+Talker+Code2Wav）│
              │  冻结 Talker/Code2Wav/编码器，仅 Thinker LLM 可训 │
              │  thinker forward → text logits → log_probs      │
              └─────────────────────────┬─────────────────────┘
                                        │ ppo_loss → compute_policy_loss_gspo（零改动）
                                        ▼
              权重同步 actor → rollout（仅 stage0 Thinker；Talker/Code2Wav 冻结）
```

### 2.1 policy / advantage / loss

- `response_ids` = Thinker 生成的文本/回合 token（**单词表，无偏移、无拼接**）。
- `response_logprobs` = 对应 token 的 logprob（标准路径）。
- `response_mask` = Thinker token mask。
- `advantages` = 会话级标量广播（GRPO 组内归一化）。
- `actor.policy_loss.loss_mode=gspo` + `algorithm.adv_estimator=grpo` 直接启用 `compute_policy_loss_gspo`。

**与 Qwen3-Omni Thinker GSPO 完全同构，无任何 verl 核心改动。**

### 2.2 Talker 采样方差控制（唯一新增的建模注意点）

Talker 冻结且不在 policy 内，其采样随机性是**环境噪声**。为让 reward 更直接归因于 Thinker 决策、降低方差：

- 建议 rollout 时 Talker 用**确定性/低温采样**（`duplex_stage_sampling_params` 或 stage1 的 `temperature` 调低、`top_k/top_p` 收紧），Thinker 保持 RL 温度采样。
- 若保留 Talker 随机性，等价于给 reward 加噪声，GRPO 仍可训练，但方差更大、样本效率更低。

---

## 3. 训练适配器（FSDP actor，verl-omni 新增）

文件：`verl_omni/pipelines/minicpmo_4_5/thinker_training_adapter.py`

> 结构上基本是 Qwen3-Omni Thinker 适配器的翻版（`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`），**关键差异：不剥离任何子模块（整网部署），用 `requires_grad=False` 冻结非训练组件**。

```python
# architecture 双注册 + version 门禁（与 vllm-omni 的 hf_config_predicate 对齐）
@OmniModelBase.register("MiniCPMO", stage="thinker")
@OmniModelBase.register("MiniCPMO45OmniForConditionalGeneration", stage="thinker")
class MiniCPMO45ThinkerAdapter(OmniModelBase):

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        # 整网部署：不剥离任何子模块（Thinker + Talker + Code2Wav 全部保留）。
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        version = str(getattr(getattr(module, "config", None) or model_config.hf_config, "version", ""))
        if version != "4.5":
            raise ValueError(f"MiniCPMO45ThinkerAdapter requires version=='4.5', got {version!r}.")
        module = super().configure_model(module, model_config)   # get_strip_modules=[] 不剥离
        # 冻结除 Thinker LLM 外的一切（编码器 + Talker + Code2Wav），只更新 Thinker LLM
        for name, param in module.named_parameters():
            param.requires_grad = _is_thinker_llm(name)          # 见要点 4
        # 文本 logprob 前向走 thinker（policy=文本 token）；整网仍在内存中，checkpoint/同步完整
        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        # 三阶段都要给 FSDP 分片粒度（Thinker 层 + Talker 层 + Code2Wav 大模块）
        module._no_split_modules = ["<ThinkerDecoderLayer>", "<TalkerLayer>", "<Code2WavModule>"]
        return module

    @classmethod
    def configure_processor(cls, model_path, model_config):
        # AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # 按需绑定 mrope/get_rope_index（cast int64）、dedup_pad_tokens（AR 模式 pad 去重）
        ...

    @classmethod
    def configure_tokenizer(cls, model_path, model_config):
        # AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 若 chat_template 在独立 chat_template.json，则读入赋值给 tokenizer.chat_template
        ...
```

要点：

1. **`architecture` 双注册 + `version` 门禁**：`config.json["architectures"][0]="MiniCPMO"`（与 2.6 同名），4.5 只能靠 `version=="4.5"` 区分——这是 MiniCPM-o 与 Qwen3-Omni 最大的接入差异。
2. **`auto_model_class`**：MiniCPM-o 不在 transformers 核心库，靠 `trust_remote_code`/`auto_map` 装载；必要时在 `register_auto_classes()` 里 `AutoConfig.register / AutoModelForMultimodalLM.register` 显式绑定 remote 类。
3. **子模块名**（`thinker`/`talker`/`llm`/`tts`）与 `_no_split_modules` 层类名以 checkpoint 的 `modeling_minicpmo45.py` 为准。
4. **冻结范围（整网部署 + 只训 LLM）**：`get_strip_modules=[]` 保留完整三阶段；`configure_model` 显式 `requires_grad=False` 冻结编码器 + Talker + Code2Wav，只留 Thinker LLM 可训（`_is_thinker_llm(name)` 只对 `thinker.language_model.*` / `thinker.llm.*` 等 LLM 参数放行，具体前缀以 remote code 为准）。整网保留带来：checkpoint 完整（无需推理侧回填）、actor↔rollout 参数名一一对应、后续扩展多阶段训练不用改装载逻辑；代价是多占 Talker/Code2Wav 显存（可配合相应 offload）。
5. **前向仍是 thinker 文本 logits**：整网在内存，但 policy=文本 token，故 `module.forward` 重定向到 `module.thinker.forward`（Talker/Code2Wav 只在 rollout 执行）。

---

## 4. 推理适配器（vllm-omni rollout，verl-omni 新增）

文件：`verl_omni/pipelines/minicpmo_4_5/omni_rollout_adapter.py`

```python
@OmniRolloutPipelineBase.register("minicpmo_4_5")
class MiniCPMO45RolloutAdapter(OmniRolloutPipelineBase):

    @classmethod
    def build_stage_configs(cls, pipeline_mode="full"):
        # 完整三阶段（Thinker 文本 + Talker codec + Code2Wav 音频）
        return list(MINICPMO_4_5_PIPELINE.stages)

    @classmethod
    def get_pipeline_id(cls, pipeline_mode="full"):
        return MINICPMO_4_5_PIPELINE.model_type      # "minicpmo_4_5"，vllm-omni 已注册

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="full"):
        return [0]                                   # 只同步 Thinker；Talker/Code2Wav 冻结不同步

    @classmethod
    def policy_stage_id(cls, pipeline_mode="full"):
        return 0                                     # policy = stage0 文本

    @classmethod
    def combine_engine_outputs(cls, outputs, prompt):
        # stage0（final_output_type="text"）→ policy TokenOutput(token_ids + logprobs)
        # stage2（final_output_type="audio"）→ extra_fields["audio"] / ["audio_sample_rate"]（reward 用）
        # stage1 codec 只作音频链路的中间产物，不进 policy、不进 replay
        # → 返回 (policy_output, {"audio": waveform, "audio_sample_rate": sr, ...})
```

要点：

1. **完整三阶段**是 vllm-omni 已注册的 `MINICPMO_4_5_PIPELINE`（stage0 `final_output_type="text"` + stage2 `final_output_type="audio"`），无需在 verl-omni 派生新 pipeline。
2. `combine_engine_outputs` 只把 **stage0 文本**作为 policy，**stage2 音频**作为 reward 输入；**不收集 codec logprob**（Talker 冻结，无需）。
3. 文本 policy 用 `single_turn_agent`（无需 `postprocess_agent_loop_output` 映射）。若发现标准 `single_turn_agent` 不把 `extra_fields` 里的音频透传到 reward，退回 `omni_single_turn_agent` + 一个透传型 `postprocess_agent_loop_output`（把音频塞进 `extra_fields`）。
4. `weight_sync_stage_ids=[0]`：权重同步只搬 Thinker LLM；stage1/2 保持冻结权重（rollout 侧 Talker/Code2Wav 本来就冻结）。

---

## 5. Rollout 两条路径

> 三阶段 rollout 的 GPU 资源分配（每 replica world size 按 per-stage deploy config 计算）是独立于本方案的前提能力，见 [`multistage_rollout_resource_allocation.md`](multistage_rollout_resource_allocation.md)。本方案默认该能力已落地：`actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config` 指向 `minicpmo_4_5.yaml`，rollout world size 自动取三 stage 设备并集。

### 5.1 Path 1：batch 全 pipeline（低风险验证，首选先行）

- 一次 batch 请求跑完整 `MINICPMO_4_5_PIPELINE`，prompt = 多 unit 音频（多 `<unit>` 帧）+ 系统模板 + 参考音频，输出文本 + 音频。
- Stage0 走标准 sampler（非 duplex），`logprobs` 请求即返回 Thinker logprob；**vllm-omni 近零改动**。
- 足以验证“Thinker LLM GSPO + 音频 reward + 权重同步”端到端，再上 Path 2。

### 5.2 Path 2：native 双工 data plane（完整双工目标）

- 用 duplex data plane 做**每秒 append + 打断 + epoch 栅栏**：`open_duplex_session` → 每 1s `append_duplex_input`（`mode="append_audio_chunk"`）→ 收集 `data_plane_outputs`（listen/speak 决策 + 音频 delta + Thinker token/logprob）。
- 需新增 verl-omni **duplex agent loop**（`@register("minicpmo_duplex_agent")`）：经 rollout 服务端 `/v1/realtime` WebSocket 端点复用 `vllm_omni/clients/duplex.py` 的 `DuplexClient` 驱动整个会话（`run_duplex_training_session`，在 server actor 内执行）。
- 依赖 §6 的 Thinker logprob 改造。**actor/loss/replay 与 Path 1 完全复用**，仅 rollout 采集层不同。

---

## 6. vllm-omni 最小改造清单（只补 Thinker logprob）

> 由于**只训 Thinker LLM**，只需 Thinker 的 per-token logprob；Talker codec logprob 不需要。改动**两处**，均局部、向后兼容。

### 6.1 ① Thinker 原生双工采样器返回 logprob（`minicpmo_4_5_omni.py`）

现状：`_sample_minicpmo45_native_duplex_stage0`（约 658–710 行）返回 `SamplerOutput(sampled_token_ids=..., logprobs_tensors=None)`；`_sample_minicpmo45_native_duplex_row`（712–798 行）只返回采样 id，不返回 logprob。

改造：让行采样器返回 `(sampled_id, logprob)`，logprob 按**最终实际采用的分布**计算：

- 强制 token（`max_tokens` 触顶返回 `chunk_eos`、`force_listen`、`listen→tts_bos` 改写、greedy `argmax`）：`logprob = 0.0`（确定性）；
- 两段采样中 boundary `chunk_eos` 被接受：`logprob = log_softmax(原始 logits)[chunk_eos]`；
- 二次掩码采样：`logprob = log_softmax(经 repetition_penalty/temperature/top_k/top_p/chunk_eos 掩码后的 logits)[sampled]`。

把逐行 logprob 打包进 `SamplerOutput.logprobs_tensors`（vLLM 标准格式），既有管道即把它记到 `CompletionOutput.logprobs`。

### 6.2 ② data_plane 转发 Thinker logprob（`duplex/data_plane.py`，可选 `runtime_bridge.py`）

现状：`MiniCPMO45DataPlaneSession.project_output`（约 131 行起）提取 `completion` 的 `text/token_ids/multimodal_output`，但不提取 `logprobs`。

改造：在 `project_output` 提取 `completion.logprobs`（Thinker），按 turn/epoch 去重后附加到 `runtime_result` dict（建议键 `thinker_token_ids/thinker_logprobs`），供 verl-omni 的 duplex rollout 协议重建 policy 轨迹。

> **Path 1（batch）无需上述改动**（Thinker 标准 sampler 已返回 logprob）。Path 2 才需要 ①+②。Talker 采样参数调整（§2.2）若走 `duplex/runtime.py` 的 `configure_sampling_params`，也算一处可选小改。

---

## 7. verl-omni 新增/修改文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `verl_omni/pipelines/minicpmo_4_5/__init__.py` | 新增 | 导出训练/rollout 适配器 |
| `verl_omni/pipelines/minicpmo_4_5/thinker_training_adapter.py` | 新增 | `MiniCPMO45ThinkerAdapter`（stage="thinker"，整网加载 + 冻结） |
| `verl_omni/pipelines/minicpmo_4_5/omni_rollout_adapter.py` | 新增 | 完整三阶段 rollout：policy=stage0，audio=stage2 reward |
| `verl_omni/pipelines/minicpmo_4_5/duplex_agent_loop.py` | 新增（Path 2） | `@register("minicpmo_duplex_agent")` |
| `verl_omni/workers/rollout/vllm_rollout/vllm_omni_duplex_client.py` | 新增（Path 2） | `/v1/realtime` WebSocket duplex 训练客户端 |
| `verl_omni/pipelines/__init__.py` | 修改 | 加 `minicpmo_4_5` 导入与 `__all__` |
| `examples/gspo_trainer/minicpmo_4_5/run_minicpmo_4_5_thinker_gspo_*.sh` | 新增 | 启动脚本 |
| `examples/gspo_trainer/minicpmo_4_5/README.md` | 新增 | 数据/奖励/验证说明 |

> 除 `vllm_omni_async_server.py` 新增 WebSocket duplex 分支（`_generate_duplex`，由 agent loop 置 `sampling_params["duplex"]` 触发）外，不改 `vllm_omni_ar_strategy.py` / `engine_workers.py` / `omni_impl.py`。整网能力完全通过 `OmniModelBase`/`OmniRolloutPipelineBase` 注册钩子注入。

---

## 8. 运行脚本 sketch（仿 Qwen3-Omni AVQA GSPO）

```bash
export VERL_USE_EXTERNAL_MODULES=verl_omni
MODEL_PATH=${MODEL_PATH:-"$HOME/models/openbmb/MiniCPM-o-4_5"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="$HOME/data/<duplex_task>/train.parquet" \
    data.val_files="$HOME/data/<duplex_task>/val.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    +data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode="full" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256] \
    algorithm.adv_estimator=grpo \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/<duplex_reward>.py \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.project_name=gspo \
    trainer.experiment_name=minicpmo_4_5_thinker_duplex \
    "$@"
```

**注意（整网部署 + 只训 LLM 的两种实现）**：

- **LoRA 模式（脚本所示）**：整网基座冻结，`target_modules`/`exclude_modules` 控制 LoRA 只注入 Thinker LLM 的 `q/k/v/o_proj`（`exclude_modules` 排除 talker/code2wav/编码器）。此时“只训 Thinker LLM”= LoRA 适配器只挂在 Thinker LLM 上，Talker/Code2Wav 既不被 LoRA 也不被更新，但仍整网加载。
- **全参数模式（无 lora_rank）**：`configure_model` 的 `requires_grad=False` 冻结一切、只放行 Thinker LLM（§3 要点 4）；`exclude_modules` 不再需要。整网同样加载。
- 两种模式都不剥离子模块，checkpoint 都包含完整三阶段权重。

---

## 9. 奖励设计与数据

- **会话级奖励**：一个双工会话 → 单标量，可组合：
  - 回答质量：ASR 转写 + LLM judge 或规则；
  - 回合质量：`<|listen|>/<|speak|>/<|turn_eos|>` 决策与 ground-truth 对齐、是否抢话/漏听、打断时延；
  - 音频质量：码率/自然度（复用 Qwen3-TTS 风格指标）。
- **数据**：双工对话轨迹（用户音频 + 参考音频 + 期望回合行为 + 回答文本）。首版可用 AVQA 类 audio+text 数据打通，再换带回合标签的双工数据。
- **reward manager**：`naive` + 自定义 `compute_score`（`verl_omni/utils/reward_score/`）。

---

## 10. 验证与测试

1. **加载 smoke**：`OmniModelConfig` 解析 `architectures[0]="MiniCPMO"` + `version="4.5"`；`stage="thinker"` 适配器命中；**整网（Thinker+Talker+Code2Wav）可 FSDP 封装**；Talker/Code2Wav 保留且 `requires_grad=False`。
2. **rollout↔actor 数值一致性（首要信号）**：`rollout_actor_probs_pearson_corr > 0.995`、`rollout_corr/log_ppl_diff≈0`（Thinker 文本段）。
3. **GSPO 数值**：`actor/loss`、`actor/grad_norm`、`actor/pg_clipfrac` 正常；无 OOM；确认只有 Thinker LLM 参数有梯度。
4. **端到端**：Path 1 先跑通（看 reward 随 step 上升），再 Path 2 验证打断/epoch 下轨迹与 logprob 对齐。
5. **回归**：确认不改 `verl/`、`vllm/`；vllm-omni 改动仅 Thinker logprob 两处，附单测；`pre-commit` 全绿。

---

## 11. 风险、依赖与边界

- **vllm-omni pin**：rollout 依赖 `.github/vllm_omni_pin.txt` 锁定的 commit；Path 2 的 logprob 改造需该 pin 之后合入（或先在本地 checkout 验证）。
- **HF remote-code 装载**：子模块名（`thinker`/`talker`/`llm`/`tts`）、`_no_split_modules` 层类名、`auto_model_class` 需读 `modeling_minicpmo45.py` 确认。
- **整网显存**：actor 侧整网加载比 strip 多占 Talker/Code2Wav 显存，需用 `requires_grad=False` + 相应 offload 调平（`fsdp_config.param_offload` 等）。
- **Talker 采样噪声**：Talker 冻结但随机采样会给 reward 加噪，建议 rollout 时调低其温度（§2.2）。
- **Path 2 的 logprob 语义**：打断（epoch 前推）丢弃的旧 turn 不进入 GSPO 序列（在 duplex agent loop 里显式定义，只保留最终提交 turn）。

---

## 附录：参考实现索引

| 参考 | 路径 |
|---|---|
| 训练适配器（thinker，本方案主范式，但改为整网不 strip） | `verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py` |
| rollout 适配器（多 stage combine 范式） | `verl_omni/pipelines/qwen3_tts/omni_rollout_adapter.py` |
| GSPO loss（只读复用） | `verl/trainer/ppo/core_algos.py:1545` |
| ppo_loss（只读复用） | `verl/workers/utils/losses.py:57` |
| logprob 计算（只读） | `verl/workers/engine/fsdp/transformer_impl.py:1328`（`prepare_model_outputs`） |
| GSPO recipe（本方案脚本范式） | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh` |
| vllm-omni pipeline（只读） | `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py` |
| vllm-omni Thinker 采样（改造点 ①） | `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py:658-798` |
| vllm-omni 双工数据面（改造点 ②） | `vllm_omni/model_executor/models/minicpmo_4_5/duplex/data_plane.py:131` |
| vllm-omni 双工深度解析（只读） | 根目录 `docs/vllm-omni-minicpmo_4_5_duplex.md` |
