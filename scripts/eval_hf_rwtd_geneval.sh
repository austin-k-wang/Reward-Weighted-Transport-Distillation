#!/usr/bin/env bash
set -euo pipefail

# Download the published RWTD GenEval LoRA and reproduce its official
# one-step GenEval evaluation. Override NUM_GPUS/GPU_IDS to parallelize.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO_ID="${HF_REPO_ID:-austin-k-wang/SanaSprint1.6B-RWTD-GenEval}"
ADAPTER_DIR="${ROOT}/models/huggingface-adapters/SanaSprint1.6B-RWTD-GenEval"
BASE_CHECKPOINT="${ROOT}/models/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/geneval/hf-rwtd-geneval-official-maxT1.5708}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Missing local SANA-Sprint checkpoint: ${BASE_CHECKPOINT}" >&2
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

export NUM_GPUS="${NUM_GPUS:-1}"
export SEED_MODE=official
export SEED=0
export INFERENCE_STEPS=1
export MAX_TIMESTEPS=1.5708
export SKIP_SCORING=false

"${ROOT}/scripts/eval_geneval.sh" "${ADAPTER_DIR}" "${OUTPUT_DIR}"
