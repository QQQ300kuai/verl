#!/usr/bin/env bash
# GRPO | Qwen3-4B-Instruct-2507 (merged SFT) | e-commerce CS tool-calling | FSDP + vLLM
# Adapted from examples/grpo_trainer/run_qwen3_4b_fsdp.sh.
#
# Defaults target TWO 24GB GPUs (e.g. two RTX 4090s). One GPU is not enough:
# FSDP's fp32 master weights (~16GB for this 4B model) can't be sharded at
# world_size=1, leaving no room for the colocated vLLM rollout engine.
#
# Trains a verl-side LoRA (the SFT LoRA is already merged into base
# weights); the actor's base weights double as the ref policy, so there's
# no second full model copy -- see docs/advance/ppo_lora.rst.
#
# train.parquet/test.parquet come from prepare_data.py. Run from the verl
# repo root:
#   MODEL_PATH=/root/autodl-tmp/code/LlamaFactory/saves/qwen3-4b/merged/sft_20260823_1124 DATA_DIR=verl_runs/qwen3-4b-instruct-2507_grpo_20260824/data bash verl_runs/qwen3-4b-instruct-2507_grpo_20260824/grpo.sh
#
# override NGPUS_PER_NODE/ROLLOUT_TP if your GPU count differs, and
# TRAIN_BATCH_SIZE/PPO_MINI_BATCH_SIZE/PPO_MICRO_BATCH_SIZE_PER_GPU to GPU memory.

set -xeuo pipefail

DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Colocated vLLM + FSDP on a single 24GB GPU is tight; expandable_segments
# reduces fragmentation-driven OOMs during the FSDP unshard in update_weights.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the merged SFT model dir}
MODEL_PATH=${MODEL_PATH%/}
DATA_DIR=${DATA_DIR:?set DATA_DIR to the dir produced by prepare_data.py}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/test.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reward_fn.py}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

# ~650 train examples after the 90/10 split -- keep batches small (both for
# dataset size and for 24GB/GPU memory) and compensate with more epochs +
# a decent group size (n) so GRPO's per-prompt baseline is estimated from
# enough samples.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# Prompts are one system(tools)+user turn; responses are either a single
# <tool_call> block or a short direct reply -- both short.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}

# LoRA needs a higher LR than full-parameter fine-tuning (roughly an order
# of magnitude, per docs/advance/ppo_lora.rst).
ACTOR_LR=${ACTOR_LR:-1e-5}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

# tensor_model_parallel_size cannot exceed the number of GPUs you have.
ROLLOUT_TP=${ROLLOUT_TP:-2}
# The vLLM rollout engine and FSDP training both stay resident on the same
# 24GB GPU (colocated, not disaggregated). At 0.3 the two together still
# left no headroom for the transient FSDP unshard of LoRA params during
# update_weights and OOM'd. Keep more slack for training-side allocations.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.2}
ROLLOUT_N=${ROLLOUT_N:-4}
# Qwen3-4B-Instruct-2507's native max_position_embeddings is 262144 (YaRN
# long-context config); without an explicit cap, vLLM sizes the KV cache
# for that full length and OOMs well before gpu_memory_utilization is the
# binding constraint. Cap it just above max_prompt_length+max_response_length.
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}

PROJECT_NAME=${PROJECT_NAME:-test}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_lora}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

case "${DEVICE}" in
    gpu) ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        NGPUS_PER_NODE=16
        ROLLOUT_GPU_MEM_UTIL=0.9
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # required so vLLM can load the base model weights for LoRA
    actor_rollout_ref.rollout.load_format=safetensors
    # avoids OOM during update_weights: summons LoRA params layer-by-layer
    # instead of unsharding the whole model at once
    actor_rollout_ref.rollout.layered_summon=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# no reward model -- rule-based reward only, see reward_fn.py
# NOTE: the v1/reward_loop trainer path (main_ppo.py -> TaskRunnerV1) reads
# reward.custom_reward_function.*, not the legacy top-level
# custom_reward_function.* -- that migration only runs in
# fully_async_main.py, so the legacy key is silently ignored here.
REWARD=(
    reward_model.enable=False
    reward.custom_reward_function.path=${REWARD_FN_PATH}
    reward.custom_reward_function.name=compute_score
)

EXTRA=(
)

########################### launch ###########################
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${INFER_BACKEND}" = vllm ] || [ "${INFER_BACKEND}" = sglang ]; }; then
    LAUNCH=(uv run --frozen --all-packages --extra "${INFER_BACKEND}" --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${INFER_BACKEND} --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
