# RWTD for One-Step SANA-Sprint

This repository implements Reward-Weighted Transport Distillation (RWTD) for
online alignment of the one-step SANA-Sprint 1.6B text-to-image model. It
includes reproducible GenEval and Parti-Prompts evaluation, GenEval-reward
training, and HPS v2.1-reward training.

The HuggingFace collection containing the GenEval and HPSv2 post-trained checkpoints is [here](https://huggingface.co/collections/austin-k-wang/reward-weighted-transport-distillation-models). Instructions below show how to load in the checkpoints and run inference.

Run all commands below from the repository root.

## 1. Set up the environments

The modified SANA implementation is vendored under `Sana/`; no submodule
initialization is required. Create the main Python 3.11 environment:

```bash
cd Sana
SANA_SKIP_TE=1 bash environment_setup.sh sana
conda activate sana
cd ..

export SANA_PYTHON="$(conda run -n sana which python)"
export PYTHON="$SANA_PYTHON"
```

The installer uses CUDA 12.8 and installs the project in editable mode.
Transformer Engine is not required for BF16 RWTD training or evaluation.

## 2. Download model assets

Install and authenticate the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Keep the Hugging Face cache inside the project so the evaluation scripts can
resolve all snapshots consistently:

```bash
export HF_HOME="$PWD/models/hf-cache"
mkdir -p "$HF_HOME" models Sana/reward_ckpts
```

Download the SANA-Sprint generator and its supporting models:

```bash
hf download Efficient-Large-Model/Sana_Sprint_1.6B_1024px \
  --local-dir models/Sana_Sprint_1.6B_1024px

hf download Efficient-Large-Model/gemma-2-2b-it
hf download mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers

hf download facebook/dinov2-base \
  --local-dir models/facebook-dinov2-base
```

Download the HPS v2.1 and PickScore reward models used during training:

```bash
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  open_clip_pytorch_model.bin \
  --local-dir Sana/reward_ckpts

hf download xswu/HPSv2 \
  HPS_v2.1_compressed.pt \
  --local-dir Sana/reward_ckpts

hf download yuvalkirstain/PickScore_v1 \
  --local-dir models/PickScore_v1
```

The full Parti-Prompts evaluation also reports CLIP, LAION Aesthetics, and
ImageReward. Download their assets:

```bash
hf download openai/clip-vit-large-patch14
hf download bert-base-uncased

mkdir -p models/laion-aesthetic "$HOME/.cache/ImageReward"

curl -L \
  https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth \
  -o models/laion-aesthetic/sa_0_4_vit_l_14_linear.pth

hf download zai-org/ImageReward \
  ImageReward.pt med_config.json \
  --local-dir "$HOME/.cache/ImageReward"
```

## 3. Set up GenEval

GenEval uses a separate Python 3.10 environment because its MMCV 1.x and
MMDetection 2.x dependencies conflict with the SANA environment.

### Check out the tested revisions

```bash
export GENEVAL_STACK="$HOME/geneval-stack"
export GENEVAL_DIR="$GENEVAL_STACK/geneval"
mkdir -p "$GENEVAL_STACK"

git clone https://github.com/djghosh13/geneval.git "$GENEVAL_DIR"
git -C "$GENEVAL_DIR" checkout af4902f24d3ca90ebbb446dd9891a59e0f82725f

git clone https://github.com/open-mmlab/mmcv.git "$GENEVAL_DIR/mmcv"
git -C "$GENEVAL_DIR/mmcv" checkout 4e85793e58fb220ed91be8c075396b16a385f349

git clone https://github.com/open-mmlab/mmdetection.git \
  "$GENEVAL_DIR/mmdetection"
git -C "$GENEVAL_DIR/mmdetection" checkout \
  e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1
```

### Create the evaluator environment

```bash
conda create -n geneval -y \
  -c nvidia -c conda-forge \
  python=3.10 \
  "pip>=23.2,<25" \
  setuptools=69.5.1 \
  wheel ninja numpy=1.23.5 \
  cuda-nvcc=12.1 \
  cuda-cudart-dev=12.1 \
  cuda-cccl=12.1 \
  cuda-libraries-dev=12.1

conda activate geneval

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2

python -m pip install \
  addict==2.4.0 \
  albumentations==1.3.1 \
  clip-benchmark==1.4.0 \
  "cython<3" \
  einops==0.7.0 \
  matplotlib==3.7.5 \
  mmengine==0.7.3 \
  open-clip-torch==2.20.0 \
  opencv-python-headless==4.8.1.78 \
  packaging==23.2 \
  pandas==1.5.3 \
  pillow==10.1.0 \
  pycocotools==2.0.7 \
  pyyaml==6.0.1 \
  scipy==1.10.1 \
  six==1.16.0 \
  terminaltables==3.1.10 \
  timm==0.9.12 \
  tqdm==4.66.1 \
  yapf==0.40.1
```

Compile MMCV for the GPU architecture. This example targets H20/H100
(`sm_90`); use the appropriate architecture for other GPUs.

```bash
cd "$GENEVAL_DIR/mmcv"
MAX_JOBS=16 \
MMCV_WITH_OPS=1 \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST=9.0 \
MMCV_CUDA_ARGS="-arch=sm_90" \
python -m pip install -v -e . --no-build-isolation

cd "$GENEVAL_DIR/mmdetection"
python -m pip install -v -e . --no-build-isolation
cd -

export GENEVAL_PYTHON="$(conda run -n geneval which python)"
```

### Download the detector

```bash
export GENEVAL_MODEL_DIR="$PWD/models/geneval"
export GENEVAL_MODEL_CONFIG="$GENEVAL_DIR/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
export GENEVAL_OBJECT_NAMES="$GENEVAL_DIR/evaluation/object_names.txt"
mkdir -p "$GENEVAL_MODEL_DIR"

curl -L \
  https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
  -o "$GENEVAL_MODEL_DIR/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
```

Warm the OpenCLIP cache used by GenEval:

```bash
"$GENEVAL_PYTHON" - <<'PY'
import open_clip
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
print("GenEval OpenCLIP weights are ready")
PY
```

## 4. Reproduce the published model results

The published adapters are:

- [RWTD GenEval](https://huggingface.co/austin-k-wang/SanaSprint1.6B-RWTD-GenEval)
- [RWTD HPSv2](https://huggingface.co/austin-k-wang/SanaSprint1.6B-RWTD-HPSv2)

The reproduction scripts download only the LoRA adapters. They use the local
SANA-Sprint checkpoint downloaded in section 2.

### GenEval

```bash
export SANA_PYTHON="$(conda run -n sana which python)"
export GENEVAL_PYTHON="$(conda run -n geneval which python)"
export GENEVAL_DIR="$HOME/geneval-stack/geneval"
export DETECTOR_MODEL_DIR="$PWD/models/geneval"
export DETECTOR_CONFIG="$GENEVAL_DIR/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
export OBJECT_NAMES="$GENEVAL_DIR/evaluation/object_names.txt"

NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 SCORING_GPU_ID=0 \
scripts/eval_hf_rwtd_geneval.sh
```

This runs the official 553-prompt benchmark with four images per prompt,
official seeds, one inference step, CFG 4.5, and maximum timestep 1.5708.
Results are written to:

```text
outputs/geneval/hf-rwtd-geneval-official-maxT1.5708/summary.json
```

### Parti-Prompts

```bash
export SANA_PYTHON="$(conda run -n sana which python)"

NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 \
scripts/eval_hf_rwtd_hpsv2_parti.sh
```

This evaluates 1,632 prompts with five images per prompt and reports PickScore,
HPSv2, ImageReward, CLIP, and Aesthetics. Results are written to:

```text
outputs/eval-parti-prompts/hf-rwtd-hpsv2-checkpoint-400/results.json
```

## 5. Run RWTD training

The checked-in YAML files retain the original experiment paths. The commands
below override the model paths so training also works when the repository is
cloned elsewhere:

```bash
export HF_HOME="$PWD/models/hf-cache"

MODEL_OVERRIDES=(
  --set "model.config_path=$PWD/Sana/configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
  --set "model.checkpoint_path=$PWD/models/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
  --set "model.text_encoder_path=Efficient-Large-Model/gemma-2-2b-it"
  --set "model.vae_path=mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
  --set "features.model_path=$PWD/models/facebook-dinov2-base"
)
```

### Train with GenEval rewards

```bash
export PYTHON="$(conda run -n sana which python)"
export GENEVAL_PYTHON="$(conda run -n geneval which python)"
export GENEVAL_DIR="$HOME/geneval-stack/geneval"
export GENEVAL_MODEL_DIR="$PWD/models/geneval"
export GENEVAL_MODEL_CONFIG="$GENEVAL_DIR/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
export GENEVAL_OBJECT_NAMES="$GENEVAL_DIR/evaluation/object_names.txt"

NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
scripts/train_sana_sprint_rwtd_geneval.sh \
  "${MODEL_OVERRIDES[@]}"
```

The launcher starts one persistent GenEval scorer per training rank, trains a
LoRA adapter, runs periodic held-out evaluation, and writes checkpoints and
TensorBoard logs under:

```text
outputs/sana-sprint-alignment/rwtd-geneval-v3-rebalanced/
```

### Train with HPS v2.1 rewards

The following settings reproduce the published HPSv2 model configuration:
transport step 0.2, reference mass 0.15, and reward temperature 0.2.

```bash
export PYTHON="$(conda run -n sana which python)"

NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
REWARD_TEMPERATURE=0.2 \
TRANSPORT_STEP=0.2 \
REFERENCE_FRACTION=0.15 \
MAX_TRAIN_STEPS=400 \
CHECKPOINTING_STEPS=100 \
EVAL_PROMPT_FILE="$PWD/data/drawbench/alignment_eval.txt" \
OUTPUT_DIR="$PWD/outputs/sana-sprint-alignment/rwtd-hpsv2-temperature-0p2" \
scripts/train_sana_sprint_rwtd_hpsv2.sh \
  "${MODEL_OVERRIDES[@]}"
```

Training and evaluation runs write persistent logs and expose tqdm progress
bars. To run the test suite:

```bash
PYTHONPATH="$PWD" "$SANA_PYTHON" -m pytest tests
```

## Portability

Some launch scripts and YAML files retain setup-specific proxy, interpreter,
and absolute-path defaults, including `http_proxy`, `https_proxy`,
`/opt/tiger/...`, and `/home/tiger/...`. Users must replace these values with
paths and network settings for their own environment, or provide the
corresponding environment-variable and `--set` overrides shown above.
