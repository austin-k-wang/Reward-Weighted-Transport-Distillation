#!/usr/bin/env python3
"""Generate a PNG grid from the first prompts in a file with SANA-Sprint."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANA_ROOT = PROJECT_ROOT / "Sana"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "data/parti-prompts/parti-prompts.tsv"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
)
DEFAULT_CONFIG = (
    SANA_ROOT
    / "configs/sana_sprint_config/1024ms/"
    "SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/quick_viz.png"
LOCAL_PICK_SCORE_MODEL = PROJECT_ROOT / "models/PickScore_v1"
DEFAULT_PICK_SCORE_MODEL = (
    str(LOCAL_PICK_SCORE_MODEL)
    if LOCAL_PICK_SCORE_MODEL.is_dir()
    else "yuvalkirstain/PickScore_v1"
)
DEFAULT_PICK_SCORE_PROCESSOR = (
    str(LOCAL_PICK_SCORE_MODEL)
    if LOCAL_PICK_SCORE_MODEL.is_dir()
    else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
DEFAULT_HPS_BASE_CHECKPOINT = SANA_ROOT / "reward_ckpts/open_clip_pytorch_model.bin"
DEFAULT_HPS_CHECKPOINT = SANA_ROOT / "reward_ckpts/HPS_v2.1_compressed.pt"


def local_huggingface_snapshot(repository: str) -> str | None:
    """Resolve a complete model snapshot from the project Hugging Face cache.

    Args:
        repository: Hugging Face repository ID in ``organization/name`` form.

    Returns:
        Absolute snapshot directory selected by the cached ``refs/main`` file,
        or ``None`` when the repository is not cached locally.
    """
    repository_dir = (
        PROJECT_ROOT
        / "models/hf-cache/hub"
        / f"models--{repository.replace('/', '--')}"
    )
    main_ref = repository_dir / "refs/main"
    if not main_ref.is_file():
        return None
    revision = main_ref.read_text(encoding="utf-8").strip()
    snapshot = repository_dir / "snapshots" / revision
    return str(snapshot) if snapshot.is_dir() else None


DEFAULT_CLIP_MODEL = (
    local_huggingface_snapshot("openai/clip-vit-large-patch14")
    or "openai/clip-vit-large-patch14"
)
DEFAULT_IMAGEREWARD_TOKENIZER = (
    local_huggingface_snapshot("bert-base-uncased") or "bert-base-uncased"
)
DEFAULT_AESTHETIC_CHECKPOINT = (
    PROJECT_ROOT / "models/laion-aesthetic/sa_0_4_vit_l_14_linear.pth"
)
DEFAULT_IMAGEREWARD_ROOT = Path.home() / ".cache/ImageReward"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse quick-visualization command-line options.

    Args:
        argv: Optional sequence of command-line tokens without the executable
            name. When omitted, arguments are read from ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` containing resolved paths, generation
        settings, the random seed, and the requested prompt count.
    """
    parser = argparse.ArgumentParser(
        description="Generate a grid with SANA-Sprint 1.6B from a prompt file."
    )
    parser.add_argument(
        "-n",
        "--num-prompts",
        type=int,
        default=16,
        help="Number of prompts to generate (default: 16).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="TSV with a Prompt column, or a text file with one prompt per line.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the native SANA-Sprint .pth checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the SANA-Sprint 1.6B YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination grid PNG (default: outputs/quick_viz.png).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of sCM inference steps (default: 1).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale (default: 4.5).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument(
        "--pickscore-model",
        default=DEFAULT_PICK_SCORE_MODEL,
        help="PickScore model ID or local model directory.",
    )
    parser.add_argument(
        "--pickscore-processor",
        default=DEFAULT_PICK_SCORE_PROCESSOR,
        help="PickScore CLIP processor ID or local directory.",
    )
    parser.add_argument(
        "--pickscore-batch-size",
        "--reward-batch-size",
        dest="reward_batch_size",
        type=int,
        default=8,
        help="Number of generated images scored per reward batch (default: 8).",
    )
    parser.add_argument(
        "--hps-base-checkpoint",
        type=Path,
        default=DEFAULT_HPS_BASE_CHECKPOINT,
        help="Local OpenCLIP ViT-H/14 checkpoint used by HPS v2.1.",
    )
    parser.add_argument(
        "--hps-checkpoint",
        type=Path,
        default=DEFAULT_HPS_CHECKPOINT,
        help="Local HPS v2.1 preference checkpoint.",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="Hugging Face CLIP model ID or local directory.",
    )
    parser.add_argument(
        "--aesthetic-checkpoint",
        type=Path,
        default=DEFAULT_AESTHETIC_CHECKPOINT,
        help="Local LAION aesthetic predictor checkpoint.",
    )
    parser.add_argument(
        "--imagereward-root",
        type=Path,
        default=DEFAULT_IMAGEREWARD_ROOT,
        help="Local ImageReward checkpoint root.",
    )
    parser.add_argument(
        "--imagereward-tokenizer",
        default=DEFAULT_IMAGEREWARD_TOKENIZER,
        help="BERT tokenizer ID or local snapshot used by ImageReward.",
    )
    return parser.parse_args(argv)


def load_prompts(prompt_file: Path, limit: int) -> list[str]:
    """Load the first non-empty prompts from a TSV or line-oriented text file.

    Args:
        prompt_file: Input file path. A tab-separated file whose header
            contains ``Prompt`` is read by column; any other file is treated
            as one prompt per line.
        limit: Maximum number of prompts to return. Must be greater than zero.

    Returns:
        A list of at most ``limit`` prompt strings in source-file order.

    Raises:
        ValueError: If ``limit`` is not positive or the file contains no
            usable prompts.
        FileNotFoundError: If ``prompt_file`` does not exist.
    """
    if limit <= 0:
        raise ValueError("--num-prompts must be greater than zero")

    with prompt_file.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        header = [column.strip() for column in first_line.rstrip("\r\n").split("\t")]
        prompt_column = next(
            (column for column in header if column.casefold() == "prompt"),
            None,
        )

        if prompt_column is not None:
            reader = csv.DictReader(handle, delimiter="\t")
            prompts = [
                value
                for row in reader
                if (value := (row.get(prompt_column) or "").strip())
            ]
        else:
            prompts = [line.strip() for line in handle if line.strip()]

    prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def setup_logging(output_file: Path) -> logging.Logger:
    """Configure console and persistent logging for an inference run.

    Args:
        output_file: Destination PNG path. Its parent directory is created,
            and a log with the same stem and a ``.log`` suffix is written
            alongside it.

    Returns:
        The configured module logger. Messages are emitted to standard error
        and to the run-specific log file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = output_file.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )
    return logging.getLogger("quick_viz")


def save_grid(images: Sequence[Any], output_file: Path) -> tuple[int, int, int, int]:
    """Move generated image tensors to CPU and save a near-square PNG grid.

    Args:
        images: Sequence of image batches. Every item must have tensor shape
            ``(1, C, H, W)`` and may reside on any device or use any floating
            dtype supported by PyTorch.
        output_file: Destination PNG path. Its parent directory must already
            exist.

    Returns:
        The concatenated CPU float32 batch shape ``(N, C, H, W)``.

    Raises:
        RuntimeError: If no images are supplied or an item does not have shape
            ``(1, C, H, W)``.
    """
    import torch
    from torchvision.utils import save_image

    if not images:
        raise RuntimeError("Cannot create a grid without generated images")

    cpu_images = []
    for image in images:
        if image.ndim != 4 or image.shape[0] != 1:
            raise RuntimeError(
                f"Expected pipeline output shape (1, C, H, W), got {tuple(image.shape)}"
            )
        cpu_images.append(image.detach().to(device="cpu", dtype=torch.float32))

    batch = torch.cat(cpu_images, dim=0)
    columns = math.ceil(math.sqrt(len(cpu_images)))
    save_image(
        batch,
        output_file,
        nrow=columns,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    return tuple(batch.shape)


def score_and_print_images(
    prompts: Sequence[str],
    images: Sequence[Any],
    model_name_or_path: str,
    processor_name_or_path: str,
    batch_size: int,
    logger: logging.Logger,
) -> list[float]:
    """Compute and print one raw PickScore value per generated image.

    Args:
        prompts: Prompt strings corresponding one-to-one with ``images``.
        images: Generated image tensors with individual shape ``(1, C, H, W)``
            and values in SANA's ``[-1, 1]`` range.
        model_name_or_path: Hugging Face model ID or local PickScore directory.
        processor_name_or_path: Hugging Face ID or local directory for the
            PickScore CLIP-H processor.
        batch_size: Maximum number of prompt-image pairs scored per batch.
        logger: Run logger receiving individual and mean score records.

    Returns:
        Raw PickScore values as a Python list in prompt and image order.

    Raises:
        ValueError: If the prompt and image counts differ or ``batch_size`` is
            not positive.
    """
    if len(prompts) != len(images):
        raise ValueError(
            f"Expected one prompt per image, got {len(prompts)} prompts and "
            f"{len(images)} images"
        )
    if batch_size <= 0:
        raise ValueError("--pickscore-batch-size must be greater than zero")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.pickscore import PickScore

    scorer = PickScore(
        model_name_or_path=model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        tensor_value_range=(-1.0, 1.0),
        logger=logger,
    )
    score_tensor = scorer.score(
        prompts=prompts,
        images=[image[0] for image in images],
        batch_size=batch_size,
        show_progress=True,
    )
    scores = score_tensor.tolist()
    for index, (prompt, score) in enumerate(zip(prompts, scores)):
        line = f"[{index:04d}] PickScore={score:.6f} | {prompt}"
        print(line)
        logger.info(line)

    mean_score = score_tensor.mean().item()
    print(f"Mean PickScore={mean_score:.6f}")
    logger.info("Mean PickScore=%.6f", mean_score)
    return scores


def score_all_rewards(
    prompts: Sequence[str],
    images: Sequence[Any],
    *,
    pickscore_model: str,
    pickscore_processor: str,
    hps_base_checkpoint: Path,
    hps_checkpoint: Path,
    clip_model: str,
    aesthetic_checkpoint: Path,
    imagereward_root: Path,
    imagereward_tokenizer: str,
    batch_size: int,
    output_file: Path,
    logger: logging.Logger,
) -> dict[str, list[float]]:
    """Score generated images sequentially with five frozen reward models.

    Args:
        prompts: Prompt strings aligned one-to-one with generated images.
        images: Sequence of tensors shaped ``[1,3,H,W]`` in SANA's ``[-1,1]``
            range.
        pickscore_model: Local directory or Hugging Face ID for PickScore.
        pickscore_processor: Local directory or Hugging Face ID for its
            processor.
        hps_base_checkpoint: Local OpenCLIP ViT-H/14 backbone checkpoint.
        hps_checkpoint: Local HPS v2.1 preference checkpoint.
        clip_model: Local directory or Hugging Face ID for CLIP ViT-L/14.
        aesthetic_checkpoint: Local LAION aesthetic predictor checkpoint.
        imagereward_root: Local ImageReward checkpoint/cache directory.
        imagereward_tokenizer: BERT tokenizer ID or complete local snapshot.
        batch_size: Maximum prompt-image pairs scored in one forward call.
        output_file: Grid output path used to derive the score JSON path.
        logger: Run logger receiving per-image and aggregate reward records.

    Returns:
        Mapping from reward name to raw score values in prompt order. The same
        values and prompts are written beside the grid as ``*.scores.json``.

    Raises:
        ValueError: If prompt/image cardinalities differ or the batch size is
            not positive.
    """
    import torch

    if len(prompts) != len(images):
        raise ValueError(
            f"Expected one prompt per image, got {len(prompts)} prompts and "
            f"{len(images)} images"
        )
    if batch_size <= 0:
        raise ValueError("--reward-batch-size must be greater than zero")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.rewards import (
        CLIPReward,
        HPSv21Reward,
        ImageRewardEvaluator,
        LAIONAestheticReward,
        PickScoreReward,
    )

    image_batch = torch.cat(
        [image.detach().to(device="cpu", dtype=torch.float32) for image in images],
        dim=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builders = (
        (
            "PickScore",
            lambda: PickScoreReward(
                pickscore_model,
                pickscore_processor,
                device=device,
                dtype=torch.float32,
                local_files_only=True,
            ),
        ),
        (
            "HPSv2",
            lambda: HPSv21Reward(
                hps_checkpoint,
                model_checkpoint_path=hps_base_checkpoint,
                device=device,
                dtype=torch.float32,
            ),
        ),
        (
            "Aesthetics",
            lambda: LAIONAestheticReward(
                aesthetic_checkpoint,
                clip_model_name_or_path=clip_model,
                processor_name_or_path=clip_model,
                device=device,
                dtype=torch.float32,
                local_files_only=True,
            ),
        ),
        (
            "CLIP",
            lambda: CLIPReward(
                clip_model,
                processor_name_or_path=clip_model,
                device=device,
                dtype=torch.float32,
                local_files_only=True,
            ),
        ),
        (
            "ImageReward",
            lambda: ImageRewardEvaluator(
                checkpoint_root=imagereward_root,
                tokenizer_name_or_path=imagereward_tokenizer,
                device=device,
                dtype=torch.float32,
                local_files_only=True,
            ),
        ),
    )

    results: dict[str, list[float]] = {}
    for reward_name, build_reward in builders:
        logger.info("Loading and scoring with %s", reward_name)
        reward = build_reward()
        values = reward.score(prompts, image_batch, batch_size=batch_size)
        scores = values.tolist()
        results[reward_name] = scores
        for index, (prompt, score) in enumerate(zip(prompts, scores)):
            line = f"[{index:04d}] {reward_name}={score:.6f} | {prompt}"
            print(line)
            logger.info(line)
        mean_score = values.mean().item()
        print(f"Mean {reward_name}={mean_score:.6f}")
        logger.info("Mean %s=%.6f", reward_name, mean_score)
        del reward
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_path = output_file.with_suffix(".scores.json")
    records = [
        {
            "index": index,
            "prompt": prompt,
            "scores": {
                reward_name: values[index]
                for reward_name, values in results.items()
            },
        }
        for index, prompt in enumerate(prompts)
    ]
    score_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved raw reward scores to %s", score_path)
    return results


def generate_grid(
    prompts: Sequence[str],
    checkpoint: Path,
    config: Path,
    output_file: Path,
    seed: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    pickscore_model: str,
    pickscore_processor: str,
    hps_base_checkpoint: Path,
    hps_checkpoint: Path,
    clip_model: str,
    aesthetic_checkpoint: Path,
    imagereward_root: Path,
    imagereward_tokenizer: str,
    reward_batch_size: int,
    logger: logging.Logger,
) -> tuple[int, int, int, int]:
    """Generate and score one image per prompt, then save one PNG grid.

    Args:
        prompts: Ordered prompt strings, with one output image generated per
            item.
        checkpoint: Native SANA-Sprint checkpoint path.
        config: SANA-Sprint model YAML configuration path.
        output_file: Destination path for the grid PNG.
        seed: Base seed; prompt at index ``i`` uses ``seed + i``.
        steps: Number of sCM denoising steps per image.
        guidance_scale: Classifier-free guidance strength.
        height: Requested output height in pixels.
        width: Requested output width in pixels.
        pickscore_model: Hugging Face model ID or local PickScore directory.
        pickscore_processor: Hugging Face ID or local PickScore processor.
        hps_base_checkpoint: Local OpenCLIP ViT-H/14 backbone checkpoint.
        hps_checkpoint: Local HPS v2.1 preference checkpoint.
        clip_model: Hugging Face CLIP model ID or local directory.
        aesthetic_checkpoint: Local LAION aesthetic predictor checkpoint.
        imagereward_root: Local ImageReward checkpoint/cache directory.
        imagereward_tokenizer: BERT tokenizer ID or complete local snapshot.
        reward_batch_size: Maximum prompt-image pairs per scoring batch.
        logger: Logger receiving model-load and output-shape information.

    Returns:
        The generated batch shape ``(N, C, H, W)``. ``N`` equals the number
        of prompts, ``C`` is normally 3, and ``H``/``W`` are the requested
        image dimensions after SANA resolution binning and cropping.

    Raises:
        RuntimeError: If CUDA is unavailable or the pipeline returns a tensor
            with an unexpected shape.
    """
    if str(SANA_ROOT) not in sys.path:
        sys.path.insert(0, str(SANA_ROOT))

    import torch
    from app.sana_sprint_pipeline import SanaSprintPipeline
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("SANA-Sprint 1.6B requires a CUDA GPU for this script")

    logger.info("Loading SANA-Sprint config=%s checkpoint=%s", config, checkpoint)
    pipeline = SanaSprintPipeline(str(config))
    pipeline.from_pretrained(str(checkpoint))

    images = []
    for index, prompt in enumerate(tqdm(prompts, desc="Generating images", unit="image")):
        generator = torch.Generator(device=pipeline.device).manual_seed(seed + index)
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )
        images.append(image)

    images = [image.detach().float().cpu() for image in images]
    del pipeline
    torch.cuda.empty_cache()

    shape = save_grid(images, output_file)
    logger.info("Saved grid with batch shape %s to %s", shape, output_file)
    score_all_rewards(
        prompts=prompts,
        images=images,
        pickscore_model=pickscore_model,
        pickscore_processor=pickscore_processor,
        hps_base_checkpoint=hps_base_checkpoint,
        hps_checkpoint=hps_checkpoint,
        clip_model=clip_model,
        aesthetic_checkpoint=aesthetic_checkpoint,
        imagereward_root=imagereward_root,
        imagereward_tokenizer=imagereward_tokenizer,
        batch_size=reward_batch_size,
        output_file=output_file,
        logger=logger,
    )
    return shape


def main(argv: Sequence[str] | None = None) -> int:
    """Run prompt loading, SANA-Sprint inference, and grid serialization.

    Args:
        argv: Optional command-line tokens without the executable name. When
            omitted, arguments are read from ``sys.argv``.

    Returns:
        Process status code ``0`` after the grid and corresponding log file
        are saved successfully.

    Raises:
        FileNotFoundError: If the prompt file, model config, checkpoint, or
            cloned SANA repository is missing.
        ValueError: If numeric generation options are invalid.
    """
    args = parse_args(argv)
    logger = setup_logging(args.output)

    required_paths = {
        "SANA repository": SANA_ROOT,
        "prompt file": args.prompt_file,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "HPS base checkpoint": args.hps_base_checkpoint,
        "HPS v2.1 checkpoint": args.hps_checkpoint,
        "LAION aesthetic checkpoint": args.aesthetic_checkpoint,
    }
    for description, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be greater than zero")
    if args.reward_batch_size <= 0:
        raise ValueError("--reward-batch-size must be greater than zero")

    prompts = load_prompts(args.prompt_file, args.num_prompts)
    logger.info(
        "Generating %d images from %s with seed=%d, steps=%d, guidance_scale=%g",
        len(prompts),
        args.prompt_file,
        args.seed,
        args.steps,
        args.guidance_scale,
    )
    generate_grid(
        prompts=prompts,
        checkpoint=args.checkpoint,
        config=args.config,
        output_file=args.output,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        pickscore_model=args.pickscore_model,
        pickscore_processor=args.pickscore_processor,
        hps_base_checkpoint=args.hps_base_checkpoint,
        hps_checkpoint=args.hps_checkpoint,
        clip_model=args.clip_model,
        aesthetic_checkpoint=args.aesthetic_checkpoint,
        imagereward_root=args.imagereward_root,
        imagereward_tokenizer=args.imagereward_tokenizer,
        reward_batch_size=args.reward_batch_size,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
