#!/usr/bin/env bash
set -euo pipefail

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/online_alignment/sana_sprint_rwtd_hpsv2.yaml}"
PYTHON="${PYTHON:-/home/tiger/miniforge3/envs/sana/bin/python}"

# Launch topology and offline model loading.
NUM_PROCESSES="${NUM_PROCESSES:-1}"
if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer" >&2
  exit 1
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
  if [[ "${#GPU_ARRAY[@]}" -ne "${NUM_PROCESSES}" ]]; then
    echo "GPU_IDS must contain exactly NUM_PROCESSES comma-separated IDs" >&2
    exit 1
  fi
else
  GPU_ARRAY=()
  for ((rank = 0; rank < NUM_PROCESSES; rank++)); do
    GPU_ARRAY+=("${rank}")
  done
  GPU_IDS="$(IFS=,; echo "${GPU_ARRAY[*]}")"
fi
for gpu_id in "${GPU_ARRAY[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "GPU IDs must be non-negative integers, got ${gpu_id}" >&2
    exit 1
  fi
done
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# SANA-Sprint one-step rollout.
RESOLUTION="${RESOLUTION:-1024}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-1.57080}"
ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE:-2}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
VAE_GRADIENT_CHECKPOINTING="${VAE_GRADIENT_CHECKPOINTING:-true}"
CACHE_TEXT_EMBEDDINGS="${CACHE_TEXT_EMBEDDINGS:-true}"
TEXT_ENCODING_BATCH_SIZE="${TEXT_ENCODING_BATCH_SIZE:-8}"
TEXT_EMBEDDING_CACHE_PATH="${TEXT_EMBEDDING_CACHE_PATH:-${ROOT}/outputs/cache/sana_sprint_gemma_embeddings_pickapic.pt}"
REBUILD_TEXT_EMBEDDING_CACHE="${REBUILD_TEXT_EMBEDDING_CACHE:-false}"

# LoRA policy.
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_INIT="${LORA_INIT:-gaussian}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[attn.qkv,attn.proj,cross_attn.q_linear,cross_attn.kv_linear,cross_attn.proj]}"

# HPS v2.1 reward with DINOv2 transport features.
HPS_BASE_CHECKPOINT="${HPS_BASE_CHECKPOINT:-${ROOT}/Sana/reward_ckpts/open_clip_pytorch_model.bin}"
HPS_CHECKPOINT="${HPS_CHECKPOINT:-${ROOT}/Sana/reward_ckpts/HPS_v2.1_compressed.pt}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-2}"
REWARD_DTYPE="${REWARD_DTYPE:-float32}"
CURRENT_SAMPLES="${CURRENT_SAMPLES:-24}"
REFERENCE_SAMPLES="${REFERENCE_SAMPLES:-24}"
FEATURE_PROVIDER="${FEATURE_PROVIDER:-dinov2}"
FEATURE_MODEL_PATH="${FEATURE_MODEL_PATH:-${ROOT}/models/facebook-dinov2-base}"
FEATURE_NAMES="${FEATURE_NAMES:-[cls,patch_mean,patch_std]}"
FEATURE_WEIGHTS="${FEATURE_WEIGHTS:-[1.0,1.0,1.0]}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float32}"

# Independent held-out PickScore evaluation.
EVAL_ENABLED="${EVAL_ENABLED:-true}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-10}"
EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-${ROOT}/data/drawbench/alignment_eval.txt}"
EVAL_SEED="${EVAL_SEED:-1234}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-1}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-4}"
EVAL_REWARD_BATCH_SIZE="${EVAL_REWARD_BATCH_SIZE:-8}"
EVAL_PICKSCORE_MODEL="${EVAL_PICKSCORE_MODEL:-${ROOT}/models/PickScore_v1}"
EVAL_PICKSCORE_PROCESSOR="${EVAL_PICKSCORE_PROCESSOR:-${ROOT}/models/PickScore_v1}"
EVAL_PICKSCORE_DTYPE="${EVAL_PICKSCORE_DTYPE:-float32}"

# Fixed HPS statistics from 8,160 base-model Parti-Prompts generations.
COUPLING="${COUPLING:-sinkhorn}"
SINKHORN_TARGET="${SINKHORN_TARGET:-barycentric}"
STOCHASTIC_TRANSPORT="${STOCHASTIC_TRANSPORT:-false}"
REWARD_TEMPERATURE="${REWARD_TEMPERATURE:-0.5}"
REWARD_MEAN="${REWARD_MEAN:-0.3031447375430634}"
REWARD_SCALE="${REWARD_SCALE:-0.03406016468398281}"
MASS_FLOOR="${MASS_FLOOR:-0.10}"
REFERENCE_FRACTION="${REFERENCE_FRACTION:-0.15}"
TRANSPORT_STEP="${TRANSPORT_STEP:-0.20}"
OT_REGULARIZATION_SCALE="${OT_REGULARIZATION_SCALE:-0.10}"
MINIMUM_OT_EPSILON="${MINIMUM_OT_EPSILON:-1.0e-4}"
SINKHORN_ITERATIONS="${SINKHORN_ITERATIONS:-100}"
SINKHORN_TOLERANCE="${SINKHORN_TOLERANCE:-1.0e-5}"
DISPLACEMENT_CLIP="${DISPLACEMENT_CLIP:-null}"
FEATURE_STATS_PATH="${FEATURE_STATS_PATH:-null}"

# Optimization, data, and logging.
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_EPSILON="${ADAM_EPSILON:-1.0e-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT}/data/pickscore/sfw_train.txt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sana-sprint-alignment/rwtd-hpsv2-dinov2-features}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
REPORT_TO="${REPORT_TO:-tensorboard}"

for required in \
  "${CONFIG}" \
  "${PYTHON}" \
  "${PROMPT_FILE}" \
  "${EVAL_PROMPT_FILE}" \
  "${HPS_BASE_CHECKPOINT}" \
  "${HPS_CHECKPOINT}" \
  "${EVAL_PICKSCORE_MODEL}" \
  "${EVAL_PICKSCORE_PROCESSOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required RWTD/HPSv2 asset does not exist: ${required}" >&2
    exit 1
  fi
done
if [[ "${FEATURE_PROVIDER}" == "dinov2" && ! -e "${FEATURE_MODEL_PATH}" ]]; then
  echo "Required DINOv2 feature model does not exist: ${FEATURE_MODEL_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${REWARD_MEAN}" == "0.0" && "${REWARD_SCALE}" == "1.0" ]]; then
  echo "WARNING: HPS v2 reward calibration is still identity (mean=0, scale=1)." >&2
  echo "Use identity only for smoke tests; calibrate before a full training run." >&2
fi

"${PYTHON}" -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" \
  "${ROOT}/scripts/train_sana_sprint_alignment.py" \
  --config "${CONFIG}" \
  --set "model.resolution=${RESOLUTION}" \
  --set "model.guidance_scale=${GUIDANCE_SCALE}" \
  --set "model.num_inference_steps=1" \
  --set "model.max_timesteps=${MAX_TIMESTEPS}" \
  --set "model.rollout_chunk_size=${ROLLOUT_CHUNK_SIZE}" \
  --set "model.decode_chunk_size=${DECODE_CHUNK_SIZE}" \
  --set "model.gradient_checkpointing=${GRADIENT_CHECKPOINTING}" \
  --set "model.vae_gradient_checkpointing=${VAE_GRADIENT_CHECKPOINTING}" \
  --set "model.cache_text_embeddings=${CACHE_TEXT_EMBEDDINGS}" \
  --set "model.text_encoding_batch_size=${TEXT_ENCODING_BATCH_SIZE}" \
  --set "model.text_embedding_cache_path=${TEXT_EMBEDDING_CACHE_PATH}" \
  --set "model.rebuild_text_embedding_cache=${REBUILD_TEXT_EMBEDDING_CACHE}" \
  --set "lora.rank=${LORA_RANK}" \
  --set "lora.alpha=${LORA_ALPHA}" \
  --set "lora.dropout=${LORA_DROPOUT}" \
  --set "lora.init_weights=${LORA_INIT}" \
  --set "lora.target_modules=${LORA_TARGET_MODULES}" \
  --set "reward.provider=hpsv2" \
  --set "reward.base_model_path=${HPS_BASE_CHECKPOINT}" \
  --set "reward.checkpoint_path=${HPS_CHECKPOINT}" \
  --set "reward.local_files_only=true" \
  --set "reward.batch_size=${REWARD_BATCH_SIZE}" \
  --set "reward.dtype=${REWARD_DTYPE}" \
  --set "evaluation.enabled=${EVAL_ENABLED}" \
  --set "evaluation.provider=pickscore" \
  --set "evaluation.interval_steps=${EVAL_INTERVAL_STEPS}" \
  --set "evaluation.prompt_file=${EVAL_PROMPT_FILE}" \
  --set "evaluation.seed=${EVAL_SEED}" \
  --set "evaluation.samples_per_prompt=${EVAL_SAMPLES_PER_PROMPT}" \
  --set "evaluation.prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}" \
  --set "evaluation.reward_batch_size=${EVAL_REWARD_BATCH_SIZE}" \
  --set "evaluation.model_path=${EVAL_PICKSCORE_MODEL}" \
  --set "evaluation.processor_path=${EVAL_PICKSCORE_PROCESSOR}" \
  --set "evaluation.dtype=${EVAL_PICKSCORE_DTYPE}" \
  --set "features.provider=${FEATURE_PROVIDER}" \
  --set "features.model_path=${FEATURE_MODEL_PATH}" \
  --set "features.names=${FEATURE_NAMES}" \
  --set "features.dtype=${FEATURE_DTYPE}" \
  --set "objective.current_samples=${CURRENT_SAMPLES}" \
  --set "objective.reference_samples=${REFERENCE_SAMPLES}" \
  --set "objective.matched_noise=false" \
  --set "objective.feature_weights=${FEATURE_WEIGHTS}" \
  --set "rwtd.coupling=${COUPLING}" \
  --set "rwtd.sinkhorn_target=${SINKHORN_TARGET}" \
  --set "rwtd.stochastic_transport=${STOCHASTIC_TRANSPORT}" \
  --set "rwtd.reward_temperature=${REWARD_TEMPERATURE}" \
  --set "rwtd.reward_mean=${REWARD_MEAN}" \
  --set "rwtd.reward_scale=${REWARD_SCALE}" \
  --set "rwtd.mass_floor=${MASS_FLOOR}" \
  --set "rwtd.reference_fraction=${REFERENCE_FRACTION}" \
  --set "rwtd.transport_step=${TRANSPORT_STEP}" \
  --set "rwtd.ot_regularization_scale=${OT_REGULARIZATION_SCALE}" \
  --set "rwtd.minimum_ot_epsilon=${MINIMUM_OT_EPSILON}" \
  --set "rwtd.sinkhorn_iterations=${SINKHORN_ITERATIONS}" \
  --set "rwtd.sinkhorn_tolerance=${SINKHORN_TOLERANCE}" \
  --set "rwtd.displacement_clip=${DISPLACEMENT_CLIP}" \
  --set "rwtd.feature_stats_path=${FEATURE_STATS_PATH}" \
  --set "optimizer.learning_rate=${LEARNING_RATE}" \
  --set "optimizer.beta1=${ADAM_BETA1}" \
  --set "optimizer.beta2=${ADAM_BETA2}" \
  --set "optimizer.weight_decay=${WEIGHT_DECAY}" \
  --set "optimizer.epsilon=${ADAM_EPSILON}" \
  --set "optimizer.max_grad_norm=${MAX_GRAD_NORM}" \
  --set "optimizer.scheduler=${LR_SCHEDULER}" \
  --set "optimizer.warmup_steps=${LR_WARMUP_STEPS}" \
  --set "runtime.prompt_file=${PROMPT_FILE}" \
  --set "runtime.train_batch_size=${TRAIN_BATCH_SIZE}" \
  --set "runtime.max_train_steps=${MAX_TRAIN_STEPS}" \
  --set "runtime.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  --set "runtime.dataloader_num_workers=${DATALOADER_NUM_WORKERS}" \
  --set "runtime.mixed_precision=${MIXED_PRECISION}" \
  --set "runtime.seed=${SEED}" \
  --set "logging.output_dir=${OUTPUT_DIR}" \
  --set "logging.logging_steps=${LOGGING_STEPS}" \
  --set "logging.checkpointing_steps=${CHECKPOINTING_STEPS}" \
  --set "logging.report_to=${REPORT_TO}" \
  "$@"
