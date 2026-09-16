#!/usr/bin/env bash
set -euo pipefail

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANA_DIR="${SANA_DIR:-${ROOT}/Sana}"
GENEVAL_DIR="${GENEVAL_DIR:-/opt/tiger/geneval-stack/geneval}"
SANA_PYTHON="${SANA_PYTHON:-/home/tiger/miniforge3/envs/sana/bin/python}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-/home/tiger/miniforge3/envs/geneval/bin/python}"
SEED_MODE="${SEED_MODE:-invariant}"
INFERENCE_STEPS="${INFERENCE_STEPS:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [NATIVE_SANA_CHECKPOINT.pth|ALIGNMENT_CHECKPOINT_DIR] [OUTPUT_DIR]"
  echo
  echo "Environment overrides: NUM_GPUS, GPU_IDS, SCORING_GPU_ID, SEED, SEED_MODE,"
  echo "INFERENCE_STEPS, MAX_TIMESTEPS, FORCE_REGENERATE, SKIP_SCORING,"
  echo "SANA_PYTHON, GENEVAL_PYTHON,"
  echo "SANA_DIR, GENEVAL_DIR, SANA_CONFIG, EVAL_METADATA,"
  echo "DETECTOR_MODEL_DIR, and DETECTOR_CONFIG."
  echo
  echo "SEED_MODE=invariant (default) is process-count invariant."
  echo "SEED_MODE=official reproduces the official continuous RNG stream."
  echo "SKIP_SCORING=true generates the official image suite and then exits"
  echo "without running the Mask2Former evaluator."
  exit 0
fi

DEFAULT_CHECKPOINT="${ROOT}/models/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
CHECKPOINT_STEM="$(basename "${CHECKPOINT}")"
CHECKPOINT_STEM="${CHECKPOINT_STEM%.pth}"
SAFE_CHECKPOINT_STEM="${CHECKPOINT_STEM//[^a-zA-Z0-9._-]/_}"
CHECKPOINT_ID="$(printf '%s' "${CHECKPOINT}" | sha256sum | cut -c1-12)"
DEFAULT_OUTPUT_DIR="${ROOT}/outputs/geneval/${CHECKPOINT_STEM}-step${INFERENCE_STEPS}-cfg4.5"
if [[ "${SEED_MODE}" == "official" ]]; then
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}-official-seeds"
fi
OUTPUT_DIR="${2:-${DEFAULT_OUTPUT_DIR}}"
GENERATOR_CHECKPOINT="${DEFAULT_CHECKPOINT}"
ADAPTER_PATH=""
if [[ -f "${CHECKPOINT}" && "${CHECKPOINT}" == *.pth ]]; then
  GENERATOR_CHECKPOINT="${CHECKPOINT}"
elif [[ -f "${CHECKPOINT}/adapter/adapter_config.json" ]]; then
  ADAPTER_PATH="${CHECKPOINT}/adapter"
elif [[ -f "${CHECKPOINT}/adapter_config.json" ]]; then
  ADAPTER_PATH="${CHECKPOINT}"
else
  echo "Checkpoint must be a native SANA .pth file or a PEFT adapter checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
GENERATOR_CHECKPOINT="$(readlink -f "${GENERATOR_CHECKPOINT}")"

SANA_CONFIG="${SANA_CONFIG:-${SANA_DIR}/configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml}"
EVAL_METADATA="${EVAL_METADATA:-${GENEVAL_DIR}/prompts/evaluation_metadata.jsonl}"
DETECTOR_MODEL_DIR="${DETECTOR_MODEL_DIR:-/opt/tiger/models/geneval}"
DETECTOR_CONFIG="${DETECTOR_CONFIG:-${GENEVAL_DIR}/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"
OBJECT_NAMES="${OBJECT_NAMES:-${GENEVAL_DIR}/evaluation/object_names.txt}"
SEED="${SEED:-0}"
SKIP_SCORING="${SKIP_SCORING:-false}"
if [[ "${SKIP_SCORING}" == "1" ]]; then
  SKIP_SCORING=true
fi
if [[ "${SKIP_SCORING}" != true && "${SKIP_SCORING}" != false ]]; then
  echo "SKIP_SCORING must be true or false, got ${SKIP_SCORING}" >&2
  exit 1
fi
if [[ "${SEED_MODE}" != "invariant" && "${SEED_MODE}" != "official" ]]; then
  echo "SEED_MODE must be invariant or official, got ${SEED_MODE}" >&2
  exit 1
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
  NUM_GPUS="${NUM_GPUS:-${#GPU_ARRAY[@]}}"
  if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
    exit 1
  fi
  if [[ "${#GPU_ARRAY[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "GPU_IDS must contain exactly NUM_GPUS comma-separated IDs" >&2
    exit 1
  fi
else
  NUM_GPUS="${NUM_GPUS:-1}"
  if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
    exit 1
  fi
  GPU_ARRAY=()
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    GPU_ARRAY+=("${GPU_ID:-0}")
  else
    for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
      GPU_ARRAY+=("${gpu_index}")
    done
  fi
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
  exit 1
fi
for gpu_id in "${GPU_ARRAY[@]}"; do
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "GPU IDs must be non-negative integers, got ${gpu_id}" >&2
    exit 1
  fi
done
SCORING_GPU_ID="${SCORING_GPU_ID:-${GPU_ARRAY[0]}}"
if ! [[ "${SCORING_GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "SCORING_GPU_ID must be a non-negative integer" >&2
  exit 1
fi
GPU_IDS_JSON="$(IFS=,; echo "[${GPU_ARRAY[*]}]")"

NUM_PROMPTS=553
SAMPLES_PER_PROMPT=4
CFG_SCALE=4.5
if ! [[ "${INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_STEPS must be a positive integer, got ${INFERENCE_STEPS}" >&2
  exit 1
fi
MAX_TIMESTEPS="${MAX_TIMESTEPS:-1.55651}"
if [[ "${MAX_TIMESTEPS}" == *.* ]]; then
  while [[ "${MAX_TIMESTEPS}" == *0 ]]; do
    MAX_TIMESTEPS="${MAX_TIMESTEPS%0}"
  done
  if [[ "${MAX_TIMESTEPS}" == *. ]]; then
    MAX_TIMESTEPS="${MAX_TIMESTEPS}0"
  fi
fi
TIMESTEP_ARGS=()
TIMESTEPS_JSON=null
TIMESTEP_LABEL="_maxT${MAX_TIMESTEPS}"
if [[ "${INFERENCE_STEPS}" -eq 4 ]]; then
  # Appendix F.2 fixes the first two optimized boundaries, then uses 1.1 and
  # 0.6 for the final transitions. This is not the scheduler's linear default.
  PAPER_4_STEP_TIMESTEPS="[1.5682963320032104,1.3,1.1,0.6,0.0]"
  TIMESTEP_ARGS+=(--timesteps="${PAPER_4_STEP_TIMESTEPS}")
  TIMESTEPS_JSON="${PAPER_4_STEP_TIMESTEPS}"
  TIMESTEP_LABEL="_timesteps1.5682963320032104_1.3_1.1_0.6_0.0"
fi
GENERATION_LABEL="_eval_geneval_${SAFE_CHECKPOINT_STEM}_${CHECKPOINT_ID}"
if [[ "${SEED_MODE}" == "official" ]]; then
  GENERATION_LABEL="${GENERATION_LABEL}_seedmodeofficial"
fi
if [[ "${NUM_GPUS}" -gt "${NUM_PROMPTS}" ]]; then
  echo "NUM_GPUS cannot exceed the ${NUM_PROMPTS} evaluation prompts" >&2
  exit 1
fi

if [[ ! -f "${GENERATOR_CHECKPOINT}" ]]; then
  echo "Base SANA checkpoint does not exist: ${GENERATOR_CHECKPOINT}" >&2
  exit 1
fi
required_assets=(
  "${SANA_CONFIG}"
  "${EVAL_METADATA}"
)
if [[ "${SKIP_SCORING}" != true ]]; then
  required_assets+=(
    "${DETECTOR_CONFIG}"
    "${OBJECT_NAMES}"
    "${DETECTOR_MODEL_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
  )
fi
for required in "${required_assets[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required GenEval asset does not exist: ${required}" >&2
    exit 1
  fi
done
required_interpreters=("${SANA_PYTHON}")
if [[ "${SKIP_SCORING}" != true ]]; then
  required_interpreters+=("${GENEVAL_PYTHON}")
fi
for interpreter in "${required_interpreters[@]}"; do
  if [[ ! -x "${interpreter}" ]]; then
    echo "Python interpreter is not executable: ${interpreter}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(readlink -f "${OUTPUT_DIR}")"

EPOCH_NAME="unknown"
STEP_NAME="unknown"
if [[ "${CHECKPOINT}" =~ epoch_([0-9]+).*step_([0-9]+) ]]; then
  EPOCH_NAME="${BASH_REMATCH[1]}"
  STEP_NAME="${BASH_REMATCH[2]}"
fi
MODEL_ROOT="$(dirname "$(dirname "${GENERATOR_CHECKPOINT}")")"
GENERATION_NAME="GenEval_epoch${EPOCH_NAME}_step${STEP_NAME}_scale${CFG_SCALE}_step${INFERENCE_STEPS}_size1024_bs${SAMPLES_PER_PROMPT}_sampscm_seed${SEED}_bfloat16${TIMESTEP_LABEL}_imgnums${NUM_PROMPTS}${GENERATION_LABEL}"
GENERATION_DIR="${MODEL_ROOT}/vis/${GENERATION_NAME}"

cat >"${OUTPUT_DIR}/evaluation_config.json" <<EOF
{
  "checkpoint": "${CHECKPOINT}",
  "base_checkpoint": "${GENERATOR_CHECKPOINT}",
  "adapter_path": "${ADAPTER_PATH}",
  "sana_config": "${SANA_CONFIG}",
  "evaluation_metadata": "${EVAL_METADATA}",
  "generation_dir": "${GENERATION_DIR}",
  "num_prompts": ${NUM_PROMPTS},
  "samples_per_prompt": ${SAMPLES_PER_PROMPT},
  "num_inference_steps": ${INFERENCE_STEPS},
  "cfg_scale": ${CFG_SCALE},
  "max_timesteps": ${MAX_TIMESTEPS},
  "timesteps": ${TIMESTEPS_JSON},
  "seed": ${SEED},
  "seed_mode": "${SEED_MODE}",
  "skip_scoring": ${SKIP_SCORING},
  "num_generation_gpus": ${NUM_GPUS},
  "generation_gpu_ids": ${GPU_IDS_JSON},
  "scoring_gpu_id": ${SCORING_GPU_ID}
}
EOF

echo "Generating ${SAMPLES_PER_PROMPT} images for each of ${NUM_PROMPTS} prompts"
echo "Checkpoint: ${CHECKPOINT}"
echo "Generation directory: ${GENERATION_DIR}"
EXPECTED_IMAGES=$((NUM_PROMPTS * SAMPLES_PER_PROMPT))
SKIP_GENERATION=false
if [[ -d "${GENERATION_DIR}" && "${FORCE_REGENERATE:-false}" == true ]]; then
  echo "FORCE_REGENERATE=true; removing existing generated suite"
  rm -rf -- "${GENERATION_DIR}"
elif [[ -d "${GENERATION_DIR}" ]]; then
  shopt -s nullglob
  existing_metadata=("${GENERATION_DIR}"/*/metadata.jsonl)
  existing_samples=("${GENERATION_DIR}"/*/samples/*.png)
  if [[ "${#existing_metadata[@]}" -eq "${NUM_PROMPTS}" && "${#existing_samples[@]}" -eq "${EXPECTED_IMAGES}" ]]; then
    SKIP_GENERATION=true
    echo "Reusing complete existing generation"
  else
    echo "Removing incomplete generated suite before retrying"
    rm -rf -- "${GENERATION_DIR}"
  fi
fi

if [[ "${SKIP_GENERATION}" == false ]]; then
  ADAPTER_ARGS=()
  if [[ -n "${ADAPTER_PATH}" ]]; then
    ADAPTER_ARGS+=(--adapter_path="${ADAPTER_PATH}")
  fi
  echo "Generation GPUs: ${GPU_ARRAY[*]}" | tee "${OUTPUT_DIR}/generation.log"
  generation_pids=()
  generation_logs=()
  for ((rank = 0; rank < NUM_GPUS; rank++)); do
    start_index=$((rank * NUM_PROMPTS / NUM_GPUS))
    end_index=$(((rank + 1) * NUM_PROMPTS / NUM_GPUS))
    gpu_id="${GPU_ARRAY[rank]}"
    rank_log="${OUTPUT_DIR}/generation-rank-${rank}.log"
    generation_logs+=("${rank_log}")
    echo "rank=${rank} gpu=${gpu_id} prompts=[${start_index},${end_index})" | tee -a "${OUTPUT_DIR}/generation.log"
    (
      cd "${SANA_DIR}"
      GENEVAL_DATA_URL="${EVAL_METADATA}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${SANA_PYTHON}" scripts/inference_sana_sprint_geneval.py \
        --config="${SANA_CONFIG}" \
        --model_path="${GENERATOR_CHECKPOINT}" \
        "${ADAPTER_ARGS[@]}" \
        --sampling_algo=scm \
        --step="${INFERENCE_STEPS}" \
        --cfg_scale="${CFG_SCALE}" \
        --max_timesteps="${MAX_TIMESTEPS}" \
        "${TIMESTEP_ARGS[@]}" \
        --sample_nums="${NUM_PROMPTS}" \
        --n_samples="${SAMPLES_PER_PROMPT}" \
        --batch_size=1 \
        --gpu_id="${rank}" \
        --start_index="${start_index}" \
        --end_index="${end_index}" \
        --seed="${SEED}" \
        --seed_mode="${SEED_MODE}" \
        --skip_grid=true \
        --add_label="${GENERATION_LABEL}"
    ) >"${rank_log}" 2>&1 &
    generation_pids+=("$!")
  done
  if ! "${SANA_PYTHON}" "${ROOT}/scripts/monitor_geneval_progress.py" \
    --image-dir "${GENERATION_DIR}" \
    --total "${EXPECTED_IMAGES}" \
    --pids "${generation_pids[@]}"; then
    echo "Generation processes exited before all images were written" >&2
  fi
  generation_failed=false
  for ((rank = 0; rank < NUM_GPUS; rank++)); do
    if ! wait "${generation_pids[rank]}"; then
      generation_failed=true
      echo "Generation rank ${rank} failed; last log lines:" >&2
      rank_log="${generation_logs[rank]:-${OUTPUT_DIR}/generation-rank-${rank}.log}"
      if [[ -f "${rank_log}" ]]; then
        tail -n 100 "${rank_log}" >&2
      else
        echo "Generation log not found: ${rank_log}" >&2
      fi
    fi
  done
  if [[ "${generation_failed}" == true ]]; then
    exit 1
  fi
else
  echo "Generation reused from ${GENERATION_DIR}" >"${OUTPUT_DIR}/generation.log"
fi

shopt -s nullglob
metadata_files=("${GENERATION_DIR}"/*/metadata.jsonl)
sample_files=("${GENERATION_DIR}"/*/samples/*.png)
if [[ "${#metadata_files[@]}" -ne "${NUM_PROMPTS}" ]]; then
  echo "Expected ${NUM_PROMPTS} generated prompt directories, found ${#metadata_files[@]}" >&2
  exit 1
fi
if [[ "${#sample_files[@]}" -ne "${EXPECTED_IMAGES}" ]]; then
  echo "Expected ${EXPECTED_IMAGES} generated images, found ${#sample_files[@]}" >&2
  exit 1
fi
sha256sum "${sample_files[@]}" >"${OUTPUT_DIR}/image_manifest.sha256"

if [[ "${SKIP_SCORING}" == true ]]; then
  echo "SKIP_SCORING=true; skipping official GenEval scoring"
  echo "Generated images: ${GENERATION_DIR}"
  echo "Image manifest: ${OUTPUT_DIR}/image_manifest.sha256"
  exit 0
fi

RESULTS_JSONL="${OUTPUT_DIR}/results.jsonl"
echo "Running the official GenEval evaluator"
CUDA_VISIBLE_DEVICES="${SCORING_GPU_ID}" \
"${GENEVAL_PYTHON}" "${GENEVAL_DIR}/evaluation/evaluate_images.py" \
  "${GENERATION_DIR}" \
  --outfile "${RESULTS_JSONL}" \
  --model-config "${DETECTOR_CONFIG}" \
  --model-path "${DETECTOR_MODEL_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/evaluation.log"

echo "Computing the official text summary"
"${GENEVAL_PYTHON}" "${GENEVAL_DIR}/evaluation/summary_scores.py" \
  "${RESULTS_JSONL}" \
  | tee "${OUTPUT_DIR}/summary.txt"

echo "Writing prompt_results.json and summary.json"
"${GENEVAL_PYTHON}" - \
  "${RESULTS_JSONL}" \
  "${OUTPUT_DIR}/prompt_results.json" \
  "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import sys
from collections import OrderedDict
from pathlib import Path

results_path = Path(sys.argv[1])
prompt_results_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

rows = [
    json.loads(line)
    for line in results_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 553 * 4:
    raise RuntimeError(f"Expected 2212 image results, found {len(rows)}")

groups = OrderedDict()
for row in rows:
    raw_metadata = row["metadata"]
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    key = json.dumps(metadata, sort_keys=True)
    if key not in groups:
        filename = Path(row["filename"])
        groups[key] = {
            "prompt_index": filename.parent.parent.name,
            "prompt": row["prompt"],
            "tag": row["tag"],
            "metadata": metadata,
            "samples": [],
        }
    groups[key]["samples"].append(
        {
            "filename": row["filename"],
            "correct": bool(row["correct"]),
            "reason": row.get("reason"),
        }
    )

if len(groups) != 553:
    raise RuntimeError(f"Expected 553 prompt results, found {len(groups)}")

prompt_results = []
for group in groups.values():
    correct_images = sum(sample["correct"] for sample in group["samples"])
    group["image_count"] = len(group["samples"])
    group["correct_images"] = correct_images
    group["image_accuracy"] = correct_images / len(group["samples"])
    group["any_correct"] = correct_images > 0
    prompt_results.append(group)

tag_totals = OrderedDict()
for row in rows:
    values = tag_totals.setdefault(row["tag"], {"correct": 0, "count": 0})
    values["correct"] += int(bool(row["correct"]))
    values["count"] += 1
task_scores = OrderedDict(
    (tag, values["correct"] / values["count"])
    for tag, values in tag_totals.items()
)
correct_images = sum(bool(row["correct"]) for row in rows)
correct_prompts = sum(group["any_correct"] for group in prompt_results)
summary = {
    "total_images": len(rows),
    "total_prompts": len(prompt_results),
    "correct_images": correct_images,
    "image_accuracy": correct_images / len(rows),
    "correct_prompts": correct_prompts,
    "prompt_accuracy_any_sample": correct_prompts / len(prompt_results),
    "task_scores": task_scores,
    "overall_score": sum(task_scores.values()) / len(task_scores),
}

prompt_results_path.write_text(
    json.dumps(prompt_results, indent=2) + "\n",
    encoding="utf-8",
)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

echo "GenEval complete"
echo "Per-image results: ${RESULTS_JSONL}"
echo "Per-prompt results: ${OUTPUT_DIR}/prompt_results.json"
echo "Final averages: ${OUTPUT_DIR}/summary.json"
