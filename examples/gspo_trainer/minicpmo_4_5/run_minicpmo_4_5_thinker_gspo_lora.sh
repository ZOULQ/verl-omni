#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker GSPO + LoRA training (whole-network deploy, freeze Talker/Code2Wav).
#
# Trains ONLY the Thinker LLM with GSPO. The actor loads the full three-stage
# network (Thinker + Talker + Code2Wav); the Talker, Code2Wav and encoders are
# frozen (LoRA here, or requires_grad=False in the full-parameter mode).
#
# Rollout runs the full three-stage pipeline (Thinker -> Talker -> Code2Wav),
# so each sample yields text + speech and the reward sees the whole session.
#
# Rollout GPU placement is derived from vllm-omni's per-stage deploy profile
# (vllm_omni/deploy/minicpmo_4_5.yaml) via the rollout resource allocator;
# `tensor_model_parallel_size` below is only the flat fallback and gets
# overwritten to the per-stage union (W_total). Override with
#   +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config="<path>.yaml"
# to use a custom multi-GPU stage layout.
#
# Data preparation and reward: see examples/gspo_trainer/minicpmo_4_5/README.md.
# A session-level duplex reward (text + audio + turn-taking) must be supplied via
# reward.custom_reward_function.path/name below.

set -x

# Make verl_omni available to Ray workers
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"$HOME/models/openbmb/MiniCPM-o-4_5"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/<duplex_task>/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/<duplex_task>/val.parquet"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
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
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*tts.*|.*code2wav.*|.*vpm.*|.*resampler.*|.*apm.*|.*audio_projection_layer.*|.*audio_avg_pooler.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode="full" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256] \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=minicpmo_4_5_thinker_duplex \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"
