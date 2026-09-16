#!/usr/bin/env bash
set -euo pipefail

# Download the published RWTD HPSv2 LoRA and reproduce its full
# eight-GPU Parti-Prompts evaluation using the local SANA-Sprint checkpoint.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO_ID="${HF_REPO_ID:-austin-k-wang/SanaSprint1.6B-RWTD-HPSv2}"
ADAPTER_DIR="${ROOT}/models/huggingface-adapters/SanaSprint1.6B-RWTD-HPSv2"
BASE_CHECKPOINT="${ROOT}/models/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/eval-parti-prompts/hf-rwtd-hpsv2-checkpoint-400}"
SANA_PYTHON="${SANA_PYTHON:-/home/tiger/miniforge3/envs/sana/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29537}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Missing local SANA-Sprint checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -x "${SANA_PYTHON}" ]]; then
  echo "SANA Python interpreter is not executable: ${SANA_PYTHON}" >&2
  exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "Log in to access the private adapter: hf auth login" >&2
  exit 1
fi

echo "Downloading adapter: ${HF_REPO_ID}"
hf download "${HF_REPO_ID}" --local-dir "${ADAPTER_DIR}"

launch_args=(--num_processes "${NUM_GPUS}" --main_process_port "${MAIN_PROCESS_PORT}")
if (( NUM_GPUS > 1 )); then
  launch_args+=(--multi_gpu)
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${SANA_PYTHON}" -m accelerate.commands.launch \
  "${launch_args[@]}" \
  "${ROOT}/scripts/eval_parti_prompt.py" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --adapter "${ADAPTER_DIR}" \
  --prompt-file "${ROOT}/data/parti-prompts/parti-prompts.tsv" \
  --images-per-prompt 5 \
  --score-batch-size 5 \
  --resolution 1024 \
  --num-inference-steps 1 \
  --max-timesteps 1.5708 \
  --guidance-scale 4.5 \
  --base-seed 42 \
  --reward-dtype fp32 \
  --output-dir "${OUTPUT_DIR}"
