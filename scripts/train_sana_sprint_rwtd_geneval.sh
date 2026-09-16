#!/usr/bin/env bash
set -euo pipefail

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/configs/online_alignment/sana_sprint_rwtd_geneval.yaml"
PYTHON="${PYTHON:-/home/tiger/miniforge3/envs/sana/bin/python}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-/home/tiger/miniforge3/envs/geneval/bin/python}"
GENEVAL_DIR="${GENEVAL_DIR:-/opt/tiger/geneval-stack/geneval}"
GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR:-/opt/tiger/models/geneval}"
GENEVAL_OBJECT_NAMES="${GENEVAL_OBJECT_NAMES:-/opt/tiger/geneval-stack/geneval/evaluation/object_names.txt}"
GENEVAL_MODEL_CONFIG="${GENEVAL_MODEL_CONFIG:-/opt/tiger/geneval-stack/geneval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"

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
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
VAE_GRADIENT_CHECKPOINTING="${VAE_GRADIENT_CHECKPOINTING:-true}"
CACHE_TEXT_EMBEDDINGS="${CACHE_TEXT_EMBEDDINGS:-true}"
TEXT_ENCODING_BATCH_SIZE="${TEXT_ENCODING_BATCH_SIZE:-8}"
TEXT_EMBEDDING_CACHE_PATH="${TEXT_EMBEDDING_CACHE_PATH:-${ROOT}/outputs/cache/sana_sprint_gemma_embeddings_geneval_rebalanced.pt}"
REBUILD_TEXT_EMBEDDING_CACHE="${REBUILD_TEXT_EMBEDDING_CACHE:-false}"

# LoRA policy.
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_INIT="${LORA_INIT:-gaussian}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[attn.qkv,attn.proj,cross_attn.q_linear,cross_attn.kv_linear,cross_attn.proj]}"

# Current/reference populations and DINO blocks.
CURRENT_SAMPLES="${CURRENT_SAMPLES:-24}"
REFERENCE_SAMPLES="${REFERENCE_SAMPLES:-24}"
FEATURE_NAMES="${FEATURE_NAMES:-[cls,patch_mean,patch_std]}"
FEATURE_WEIGHTS="${FEATURE_WEIGHTS:-[1.0,1.0,1.0]}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4}"
GENEVAL_REWARD_MODE="${GENEVAL_REWARD_MODE:-hybrid}"
GENEVAL_BINARY_BONUS="${GENEVAL_BINARY_BONUS:-0.5}"
GENEVAL_TIMEOUT="${GENEVAL_TIMEOUT:-300}"
GENEVAL_STARTUP_TIMEOUT="${GENEVAL_STARTUP_TIMEOUT:-900}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float32}"

# Reproducible held-out GenEval evaluation.
EVAL_ENABLED="${EVAL_ENABLED:-true}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-25}"
EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-${GENEVAL_DIR}/prompts/evaluation_metadata.jsonl}"
EVAL_PROMPT_COUNT="${EVAL_PROMPT_COUNT:-53}"
EVAL_SEED="${EVAL_SEED:-1234}"
EVAL_SAMPLES_PER_PROMPT="${EVAL_SAMPLES_PER_PROMPT:-4}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-4}"
EVAL_REWARD_BATCH_SIZE="${EVAL_REWARD_BATCH_SIZE:-4}"
EVAL_GENEVAL_REWARD_MODE="${EVAL_GENEVAL_REWARD_MODE:-binary}"
EVAL_GENEVAL_BINARY_BONUS="${EVAL_GENEVAL_BINARY_BONUS:-0.0}"

# Fixed-temperature RWTD. No adaptive ESS temperature is used.
COUPLING="${COUPLING:-sinkhorn}"
SINKHORN_TARGET="${SINKHORN_TARGET:-barycentric}"
STOCHASTIC_TRANSPORT="${STOCHASTIC_TRANSPORT:-false}"
REWARD_TEMPERATURE="${REWARD_TEMPERATURE:-0.25}"
REWARD_MEAN="${REWARD_MEAN:-0.0}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
MASS_FLOOR="${MASS_FLOOR:-0.10}"
REFERENCE_FRACTION="${REFERENCE_FRACTION:-0.15}"
TRANSPORT_STEP="${TRANSPORT_STEP:-0.20}"
OT_REGULARIZATION_SCALE="${OT_REGULARIZATION_SCALE:-0.10}"
MINIMUM_OT_EPSILON="${MINIMUM_OT_EPSILON:-1.0e-4}"
SINKHORN_ITERATIONS="${SINKHORN_ITERATIONS:-100}"
SINKHORN_TOLERANCE="${SINKHORN_TOLERANCE:-1.0e-5}"
DISPLACEMENT_CLIP="${DISPLACEMENT_CLIP:-null}"
FEATURE_STATS_PATH="${FEATURE_STATS_PATH:-null}"

# Optimization and runtime.
LEARNING_RATE="${LEARNING_RATE:-3.0e-5}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_EPSILON="${ADAM_EPSILON:-1.0e-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT}/data/geneval/train-800-rebalanced.jsonl}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"

# Logging and persistence.
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sana-sprint-alignment/rwtd-geneval-v3-rebalanced}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-tensorboard}"

for required in \
  "${CONFIG}" \
  "${PROMPT_FILE}" \
  "${EVAL_PROMPT_FILE}" \
  "${GENEVAL_MODEL_CONFIG}" \
  "${GENEVAL_OBJECT_NAMES}" \
  "${GENEVAL_MODEL_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required RWTD/GenEval asset does not exist: ${required}" >&2
    exit 1
  fi
done
for interpreter in "${PYTHON}" "${GENEVAL_PYTHON}"; do
  if [[ ! -x "${interpreter}" ]]; then
    echo "Python interpreter is not executable: ${interpreter}" >&2
    exit 1
  fi
done

PYTHONPATH="${ROOT}" "${PYTHON}" - \
  "${PROMPT_FILE}" "${EVAL_PROMPT_FILE}" "${EVAL_PROMPT_COUNT}" "${EVAL_SEED}" <<'PY'
import sys
from src.geneval.metadata import load_metadata_rows, select_metadata_subset

training_rows = load_metadata_rows(sys.argv[1])
if len(training_rows) != 800:
    raise RuntimeError(
        f"Expected 800 GenEval training prompts, found {len(training_rows)}"
    )
evaluation_rows = load_metadata_rows(sys.argv[2])
selected = select_metadata_subset(
    evaluation_rows,
    count=int(sys.argv[3]),
    seed=int(sys.argv[4]),
)
print(f"Validated {len(training_rows)} structured GenEval training prompts")
print(
    f"Selected {len(selected)} of {len(evaluation_rows)} official GenEval "
    f"evaluation prompts with seed {sys.argv[4]}"
)
PY

mkdir -p "${OUTPUT_DIR}/geneval-scorers"
RUN_ID="$(date +%Y%m%d%H%M%S)-$$"
SOCKET_DIR="/tmp/rwtd-geneval-${USER:-user}-${RUN_ID}"
mkdir -p "${SOCKET_DIR}"
SCORER_PIDS=()
SCORER_SOCKETS=()

cleanup() {
  if [[ "${#SCORER_PIDS[@]}" -gt 0 ]]; then
    kill "${SCORER_PIDS[@]}" 2>/dev/null || true
    wait "${SCORER_PIDS[@]}" 2>/dev/null || true
  fi
  rm -rf -- "${SOCKET_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting ${NUM_PROCESSES} persistent GenEval scorer processes"
for ((rank = 0; rank < NUM_PROCESSES; rank++)); do
  gpu_id="${GPU_ARRAY[rank]}"
  socket_path="${SOCKET_DIR}/geneval-${rank}.sock"
  scorer_log="${OUTPUT_DIR}/geneval-scorers/rank-${rank}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONPATH="${ROOT}" \
  "${GENEVAL_PYTHON}" "${ROOT}/scripts/geneval_reward_server.py" \
    --socket "${socket_path}" \
    --model-path "${GENEVAL_MODEL_DIR}" \
    --model-config "${GENEVAL_MODEL_CONFIG}" \
    --object-names "${GENEVAL_OBJECT_NAMES}" \
    --timeout "${GENEVAL_TIMEOUT}" \
    >"${scorer_log}" 2>&1 &
  SCORER_PIDS+=("$!")
  SCORER_SOCKETS+=("${socket_path}")
  echo "rank=${rank} gpu=${gpu_id} socket=${socket_path} log=${scorer_log}"
done

for ((rank = 0; rank < NUM_PROCESSES; rank++)); do
  PYTHONPATH="${ROOT}" "${GENEVAL_PYTHON}" - \
    "${SCORER_SOCKETS[rank]}" "${GENEVAL_STARTUP_TIMEOUT}" <<'PY'
import sys
from src.geneval.client import GenEvalRewardClient

socket_path = sys.argv[1]
startup_timeout = float(sys.argv[2])
with GenEvalRewardClient(socket_path, timeout=300) as client:
    health = client.wait_until_ready(timeout=startup_timeout)
print(f"GenEval scorer ready: socket={socket_path} pid={health['pid']}")
PY
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
SOCKET_TEMPLATE="${SOCKET_DIR}/geneval-{local_rank}.sock"

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
  --set "reward.provider=geneval" \
  --set "reward.batch_size=${REWARD_BATCH_SIZE}" \
  --set "reward.socket_path=${SOCKET_TEMPLATE}" \
  --set "reward.timeout=${GENEVAL_TIMEOUT}" \
  --set "reward.startup_timeout=${GENEVAL_STARTUP_TIMEOUT}" \
  --set "reward.reward_mode=${GENEVAL_REWARD_MODE}" \
  --set "reward.binary_bonus=${GENEVAL_BINARY_BONUS}" \
  --set "reward.warmup=false" \
  --set "evaluation.enabled=${EVAL_ENABLED}" \
  --set "evaluation.provider=geneval" \
  --set "evaluation.interval_steps=${EVAL_INTERVAL_STEPS}" \
  --set "evaluation.prompt_file=${EVAL_PROMPT_FILE}" \
  --set "evaluation.prompt_count=${EVAL_PROMPT_COUNT}" \
  --set "evaluation.seed=${EVAL_SEED}" \
  --set "evaluation.samples_per_prompt=${EVAL_SAMPLES_PER_PROMPT}" \
  --set "evaluation.prompt_batch_size=${EVAL_PROMPT_BATCH_SIZE}" \
  --set "evaluation.reward_batch_size=${EVAL_REWARD_BATCH_SIZE}" \
  --set "evaluation.reward_mode=${EVAL_GENEVAL_REWARD_MODE}" \
  --set "evaluation.binary_bonus=${EVAL_GENEVAL_BINARY_BONUS}" \
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
