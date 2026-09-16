#!/usr/bin/env python3
"""Evaluate SANA-Sprint on Parti-Prompts with image rewards."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from tqdm.auto import tqdm
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SANA_ROOT = ROOT / "Sana"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))

from app.sana_sprint_pipeline import SanaSprintPipeline  # noqa: E402
from diffusion.model.builder import get_vae  # noqa: E402
from src.alignment.interfaces import RewardEvaluator  # noqa: E402
from src.rewards import (  # noqa: E402
    CLIPReward,
    HPSv21Reward,
    ImageRewardEvaluator,
    LAIONAestheticReward,
    PickScoreReward,
)


DEFAULT_IMAGES_PER_PROMPT = 5
DEFAULT_PROMPT_FILE = ROOT / "data/parti-prompts/parti-prompts.tsv"
DEFAULT_CONFIG = (
    SANA_ROOT
    / "configs/sana_sprint_config/1024ms/"
    "SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "models/Sana_Sprint_1.6B_1024px/checkpoints/"
    "Sana_Sprint_1.6B_1024px.pth"
)
DEFAULT_PICK_SCORE = ROOT / "models/PickScore_v1"
DEFAULT_HPS_BASE_CHECKPOINT = SANA_ROOT / "reward_ckpts/open_clip_pytorch_model.bin"
DEFAULT_HPS_CHECKPOINT = SANA_ROOT / "reward_ckpts/HPS_v2.1_compressed.pt"
DEFAULT_AESTHETIC_CHECKPOINT = (
    ROOT / "models/laion-aesthetic/sa_0_4_vit_l_14_linear.pth"
)
DEFAULT_IMAGEREWARD_ROOT = Path.home() / ".cache/ImageReward"
logger = logging.getLogger("eval_parti_prompt")


def local_huggingface_snapshot(repository: str) -> str | None:
    """Resolve a usable snapshot from the project-local Hugging Face cache.

    Args:
        repository: Hugging Face repository ID in ``organization/name`` form.

    Returns:
        Absolute snapshot path selected by ``refs/main``, or ``None`` when the
        repository is absent or an indexed model snapshot is missing shards.
    """
    repository_dir = (
        ROOT / "models/hf-cache/hub" / f"models--{repository.replace('/', '--')}"
    )
    main_ref = repository_dir / "refs/main"
    if not main_ref.is_file():
        return None
    revision = main_ref.read_text(encoding="utf-8").strip()
    snapshot = repository_dir / "snapshots" / revision
    if not snapshot.is_dir():
        return None
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        required_shards = {
            str(filename)
            for filename in weight_map.values()
            if isinstance(filename, str)
        }
        if not required_shards or any(
            not (snapshot / shard).is_file() for shard in required_shards
        ):
            return None
    return str(snapshot)


DEFAULT_CLIP_MODEL = (
    local_huggingface_snapshot("openai/clip-vit-large-patch14")
    or "openai/clip-vit-large-patch14"
)
DEFAULT_TEXT_ENCODER = (
    local_huggingface_snapshot("Efficient-Large-Model/gemma-2-2b-it")
    or "Efficient-Large-Model/gemma-2-2b-it"
)
DEFAULT_VAE = (
    local_huggingface_snapshot("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    or "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
)
DEFAULT_IMAGEREWARD_TOKENIZER = (
    local_huggingface_snapshot("bert-base-uncased") or "bert-base-uncased"
)


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    """Resolve reward-model precision requested for an evaluation device.

    Args:
        name: Precision name, one of ``fp16``, ``bf16``, or ``fp32``.
        device: Device on which reward inference will run.

    Returns:
        Requested floating dtype on CUDA, or ``torch.float32`` on CPU.

    Raises:
        ValueError: If ``name`` is not a supported precision.
    """
    if device.type != "cuda":
        return torch.float32
    values = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype: {name}") from exc


def sample_seed(prompt_index: int, image_index: int, base_seed: int) -> int:
    """Create a deterministic seed for one global prompt/sample pair.

    Args:
        prompt_index: Zero-based index in the ordered Parti-Prompts split.
        image_index: Zero-based image index for the prompt.
        base_seed: Non-negative user-selected evaluation seed.

    Returns:
        A non-negative PyTorch seed independent of rank and world size.

    Raises:
        ValueError: If any input index or seed is negative.
    """
    if prompt_index < 0:
        raise ValueError("prompt_index must be non-negative")
    if image_index < 0:
        raise ValueError("image_index must be non-negative")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    return (base_seed + prompt_index * 1_000_003 + image_index) % (2**63 - 1)


def shard_prompt_indices(
    total_prompts: int,
    process_index: int,
    num_processes: int,
) -> list[int]:
    """Assign unique global prompt indices to one distributed process.

    Args:
        total_prompts: Number of prompts in the complete evaluation split.
        process_index: Zero-based rank of the current process.
        num_processes: Total number of distributed processes.

    Returns:
        Strided global indices whose union across ranks covers every prompt.

    Raises:
        ValueError: If counts or the process index are invalid.
    """
    if total_prompts < 0:
        raise ValueError("total_prompts must be non-negative")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    if not 0 <= process_index < num_processes:
        raise ValueError("process_index must be in [0, num_processes)")
    return list(range(process_index, total_prompts, num_processes))


def require_local_file(path: str | Path, description: str) -> Path:
    """Resolve and validate one required local file.

    Args:
        path: User-supplied filesystem path.
        description: Human-readable label included in validation errors.

    Returns:
        Absolute resolved path to an existing regular file.

    Raises:
        FileNotFoundError: If the resolved path is not a regular file.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def load_parti_prompts(
    prompt_file: str | Path,
    *,
    max_prompts: int | None = None,
    randomize: bool = False,
    random_seed: int = 42,
) -> list[str]:
    """Load and optionally deterministically shuffle prompts from a local TSV.

    Args:
        prompt_file: Parti-Prompts TSV whose first column is ``Prompt``.
        max_prompts: Optional positive number of prompts to evaluate.
        randomize: Whether to shuffle all prompts before applying the cap.
        random_seed: Non-negative seed controlling the deterministic shuffle.

    Returns:
        Selected non-empty prompt strings without the TSV header. Prompts retain
        TSV order unless ``randomize`` is true.

    Raises:
        ValueError: If the cap or seed is invalid, or the TSV has no prompts.
        FileNotFoundError: If ``prompt_file`` is missing.
    """
    if max_prompts is not None and max_prompts < 1:
        raise ValueError("max_prompts must be positive when provided")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    path = require_local_file(prompt_file, "Parti-Prompts TSV file")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        prompts = [row[0].strip() for row in rows if row and row[0].strip()]
    if prompts and prompts[0].casefold() == "prompt":
        prompts = prompts[1:]
    if randomize:
        random.Random(random_seed).shuffle(prompts)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"Parti-Prompts TSV contained no prompts: {path}")
    return prompts


def resolve_output_dir(output_dir: str | None, checkpoint: str | Path) -> Path:
    """Resolve an explicit or checkpoint-derived evaluation directory.

    Args:
        output_dir: Optional user-selected output directory.
        checkpoint: SANA-Sprint checkpoint used to derive the default label.

    Returns:
        Absolute path receiving logs, metrics, and optional images.
    """
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    label = Path(checkpoint).expanduser().stem
    return (ROOT / "outputs/eval-parti-prompts" / label).resolve()


def setup_logging(output_dir: Path, accelerator: Accelerator) -> None:
    """Configure rank-aware console output and a persistent main-rank log.

    Args:
        output_dir: Existing directory receiving ``eval.log``.
        accelerator: Runtime identifying process rank and main-process status.

    Returns:
        Nothing. Python's root logging configuration is replaced.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(output_dir / "eval.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format=(
            "%(asctime)s | %(levelname)s | "
            f"rank={accelerator.process_index}/{accelerator.num_processes} | "
            "%(message)s"
        ),
        handlers=handlers,
        force=True,
    )


def load_sana_pipeline(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    max_timesteps: float,
    adapter_path: str | Path | None = None,
    text_encoder_path: str | Path | None = None,
    vae_path: str | Path | None = None,
) -> SanaSprintPipeline:
    """Load native SANA-Sprint weights and an optional PEFT LoRA adapter.

    Args:
        config_path: Native SANA-Sprint 1.6B YAML configuration.
        checkpoint_path: Native ``.pth`` SANA-Sprint checkpoint.
        device: Accelerator-selected CPU or CUDA device for this rank.
        max_timesteps: Initial trigonometric-flow timestep used by generation.
        adapter_path: Optional PEFT adapter directory containing its
            configuration and serialized weights.
        text_encoder_path: Optional local Gemma model snapshot used instead of
            the repository ID embedded in the native SANA configuration.
        vae_path: Optional local DC-AE snapshot used instead of the repository
            ID embedded in the native SANA configuration.

    Returns:
        Frozen SANA-Sprint pipeline, optionally wrapped with the requested
        adapter, with internal denoising bars disabled. Model and VAE dtypes
        follow the native YAML configuration.

    Raises:
        FileNotFoundError: If ``adapter_path`` is not an existing directory.
    """
    class LocalAssetPipeline(SanaSprintPipeline):
        """Resolve optional SANA auxiliary models from explicit local paths."""

        def build_vae(self, native_config: Any) -> torch.nn.Module:
            """Load the configured VAE from the explicit local snapshot.

            Args:
                native_config: Native SANA VAE configuration containing the
                    architecture type and fallback pretrained source.

            Returns:
                VAE module moved to the pipeline device and configured dtype.
            """
            source = vae_path or native_config.vae_pretrained
            return get_vae(
                native_config.vae_type,
                str(source),
                self.device,
            ).to(self.vae_dtype)

        def build_text_encoder(
            self,
            native_config: Any,
        ) -> tuple[Any, torch.nn.Module]:
            """Load the Gemma tokenizer and decoder from a local snapshot.

            Args:
                native_config: Native SANA text configuration accepted for API
                    compatibility; model selection comes from
                    ``text_encoder_path``.

            Returns:
                Right-padding tokenizer and Gemma decoder module on the
                process-local evaluation device.

            Raises:
                ValueError: If no explicit local text encoder path was given.
            """
            del native_config
            if text_encoder_path is None:
                raise ValueError("text_encoder_path is required for local loading")
            local_files_only = Path(text_encoder_path).expanduser().is_dir()
            tokenizer = AutoTokenizer.from_pretrained(
                str(text_encoder_path),
                local_files_only=local_files_only,
            )
            tokenizer.padding_side = "right"
            text_encoder = (
                AutoModelForCausalLM.from_pretrained(
                    str(text_encoder_path),
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )
                .get_decoder()
                .to(self.device)
            )
            return tokenizer, text_encoder

    pipeline_class = (
        LocalAssetPipeline
        if text_encoder_path is not None
        else SanaSprintPipeline
    )
    pipe = pipeline_class(str(config_path), device=device)
    pipe.config.max_timesteps = max_timesteps
    pipe.from_pretrained(str(checkpoint_path))
    if adapter_path is not None:
        resolved_adapter = Path(adapter_path).expanduser().resolve()
        if not resolved_adapter.is_dir():
            raise FileNotFoundError(f"Missing PEFT adapter directory: {resolved_adapter}")
        pipe.model = PeftModel.from_pretrained(
            pipe.model,
            str(resolved_adapter),
            is_trainable=False,
        )
    pipe.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def reward_statistics(values: list[float]) -> tuple[float, float]:
    """Compute population mean and standard deviation for reward values.

    Args:
        values: Non-empty sequence of scalar rewards from one evaluator.

    Returns:
        Pair ``(mean, population_std)``; singleton standard deviation is zero.

    Raises:
        ValueError: If no rewards are provided.
    """
    if not values:
        raise ValueError("At least one reward is required")
    return float(statistics.fmean(values)), float(statistics.pstdev(values))


@torch.inference_mode()
def evaluate_prompts(
    *,
    prompts: list[str],
    pipe: SanaSprintPipeline,
    scorers: Mapping[str, RewardEvaluator],
    accelerator: Accelerator,
    output_dir: Path,
    images_per_prompt: int,
    score_batch_size: int,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    base_seed: int,
    save_images: bool,
) -> dict[str, object]:
    """Generate and score deterministic SANA-Sprint samples across all ranks.

    Args:
        prompts: Complete ordered Parti-Prompts split shared by every rank.
        pipe: Native SANA-Sprint pipeline on the process-local device.
        scorers: Named canonical reward evaluators on the process-local device.
        accelerator: Distributed runtime used for sharding and gathering.
        output_dir: Shared directory for metrics and optional images.
        images_per_prompt: Number of independently seeded images per prompt.
        score_batch_size: Maximum generated images per reward forward pass.
        resolution: Requested square image size, divisible by 32.
        num_inference_steps: Number of SANA sCM denoising steps.
        guidance_scale: SANA-Sprint classifier-free guidance scale.
        base_seed: Seed from which global prompt/sample seeds are derived.
        save_images: Whether individual generated images are written to disk.

    Returns:
        Main-rank JSON-serializable global and per-prompt metrics. Other ranks
        return an empty-record placeholder after all collectives complete.

    Raises:
        ValueError: If prompts or numeric evaluation settings are invalid.

    Notes:
        Samples are generated one at a time so each result is invariant to
        process count and scoring batch size. Reward evaluation is batched.
    """
    if not prompts:
        raise ValueError("At least one prompt is required")
    if not scorers:
        raise ValueError("At least one reward scorer is required")
    if images_per_prompt < 1:
        raise ValueError("images_per_prompt must be positive")
    if score_batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    if resolution < 32 or resolution % 32:
        raise ValueError("resolution must be positive and divisible by 32")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")

    image_dir = output_dir / "images"
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    local_indices = shard_prompt_indices(
        len(prompts),
        accelerator.process_index,
        accelerator.num_processes,
    )
    max_local_prompts = (
        len(prompts) + accelerator.num_processes - 1
    ) // accelerator.num_processes
    records: list[dict[str, object]] = []
    running_rewards: dict[str, list[float]] = {
        name: [] for name in scorers
    }
    progress = tqdm(
        total=len(prompts),
        desc="Parti-Prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    for local_position in range(max_local_prompts):
        local_records: list[dict[str, object]] = []
        if local_position < len(local_indices):
            prompt_index = local_indices[local_position]
            prompt = prompts[prompt_index]
            seeds: list[int] = []
            images: list[torch.Tensor] = []
            image_paths: list[str] = []

            for image_index in range(images_per_prompt):
                seed = sample_seed(prompt_index, image_index, base_seed)
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                sample = pipe(
                    prompt=prompt,
                    height=resolution,
                    width=resolution,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
                image = sample[0].detach().float().cpu()
                images.append(image)
                seeds.append(seed)

                if save_images:
                    image_path = (
                        image_dir
                        / f"prompt-{prompt_index:05d}-sample-{image_index:02d}.png"
                    )
                    save_image(
                        image,
                        image_path,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    image_paths.append(str(image_path))

            image_batch = torch.stack(images)
            scores_by_reward: dict[str, list[float]] = {}
            means_by_reward: dict[str, float] = {}
            stds_by_reward: dict[str, float] = {}
            for reward_name, scorer in scorers.items():
                score_tensor = scorer.score(
                    [prompt] * len(images),
                    image_batch,
                    batch_size=score_batch_size,
                )
                scores = [float(score) for score in score_tensor]
                prompt_mean, prompt_std = reward_statistics(scores)
                scores_by_reward[reward_name] = scores
                means_by_reward[reward_name] = prompt_mean
                stds_by_reward[reward_name] = prompt_std
            local_records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "average_rewards": means_by_reward,
                    "reward_stds": stds_by_reward,
                    "rewards": scores_by_reward,
                    "seeds": seeds,
                    "image_paths": image_paths,
                }
            )

        gathered_records = gather_object(local_records)
        if accelerator.is_main_process:
            records.extend(gathered_records)
            for record in gathered_records:
                record_rewards = cast(dict[str, list[float]], record["rewards"])
                for reward_name in scorers:
                    running_rewards[reward_name].extend(record_rewards[reward_name])
            progress.update(len(gathered_records))
            if any(running_rewards.values()):
                progress.set_postfix(
                    {
                        name: f"{statistics.fmean(values):.4f}"
                        for name, values in running_rewards.items()
                        if values
                    }
                )

    progress.close()
    if not accelerator.is_main_process:
        return {
            "reward_metrics": {},
            "results": [],
        }

    records.sort(key=lambda record: cast(int, record["prompt_index"]))
    reward_metrics: dict[str, dict[str, float | int]] = {}
    for reward_name in scorers:
        all_rewards = [
            float(reward)
            for record in records
            for reward in cast(
                dict[str, list[float]], record["rewards"]
            )[reward_name]
        ]
        prompt_means = [
            cast(dict[str, float], record["average_rewards"])[reward_name]
            for record in records
        ]
        average_reward, reward_std = reward_statistics(all_rewards)
        prompt_average, prompt_average_std = reward_statistics(prompt_means)
        reward_metrics[reward_name] = {
            "average_reward": average_reward,
            "reward_std": reward_std,
            "average_prompt_reward": prompt_average,
            "prompt_reward_std": prompt_average_std,
            "num_generations": len(all_rewards),
        }
    return {
        "reward_metrics": reward_metrics,
        "num_prompts": len(prompts),
        "images_per_prompt": images_per_prompt,
        "num_generations": len(prompts) * images_per_prompt,
        "base_seed": base_seed,
        "results": records,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the SANA-Sprint Parti-Prompts evaluation command line.

    Args:
        None.

    Returns:
        Parser covering model, dataset, generation, scoring, and output
        settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible SANA-Sprint 1.6B images for Parti-Prompts "
            "and report reward distributions."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT LoRA adapter directory applied to the base checkpoint.",
    )
    parser.add_argument(
        "--text-encoder-path",
        default=DEFAULT_TEXT_ENCODER,
        help="Local Gemma snapshot used by the SANA prompt encoder.",
    )
    parser.add_argument(
        "--vae-path",
        default=DEFAULT_VAE,
        help="Local DC-AE snapshot used by the SANA image decoder.",
    )
    parser.add_argument("--pickscore-path", default=str(DEFAULT_PICK_SCORE))
    parser.add_argument(
        "--pickscore-processor-path",
        default=str(DEFAULT_PICK_SCORE),
        help="Local processor directory or Hugging Face processor ID.",
    )
    parser.add_argument(
        "--hps-base-checkpoint",
        default=str(DEFAULT_HPS_BASE_CHECKPOINT),
        help="Local OpenCLIP ViT-H/14 base checkpoint.",
    )
    parser.add_argument(
        "--hps-checkpoint",
        default=str(DEFAULT_HPS_CHECKPOINT),
        help="Local HPS v2.1 checkpoint.",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="Local CLIP snapshot or Hugging Face model ID.",
    )
    parser.add_argument(
        "--aesthetic-checkpoint",
        default=str(DEFAULT_AESTHETIC_CHECKPOINT),
        help="Local LAION aesthetic predictor checkpoint.",
    )
    parser.add_argument(
        "--imagereward-root",
        default=str(DEFAULT_IMAGEREWARD_ROOT),
        help="ImageReward checkpoint/cache directory.",
    )
    parser.add_argument(
        "--imagereward-tokenizer",
        default=DEFAULT_IMAGEREWARD_TOKENIZER,
        help="Local BERT tokenizer snapshot or Hugging Face model ID.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Local TSV file whose first column contains prompts.",
    )
    parser.add_argument(
        "--num-prompts",
        "--max-prompts",
        dest="max_prompts",
        type=int,
        default=None,
        help="Number of prompts to evaluate; defaults to the complete TSV.",
    )
    parser.add_argument(
        "--randomize-prompts",
        action="store_true",
        help="Deterministically shuffle prompts before selecting --num-prompts.",
    )
    parser.add_argument(
        "--prompt-order-seed",
        type=int,
        default=42,
        help="Seed for --randomize-prompts, independent of generation seeds.",
    )
    parser.add_argument(
        "--images-per-prompt",
        type=int,
        default=DEFAULT_IMAGES_PER_PROMPT,
    )
    parser.add_argument(
        "--score-batch-size",
        "--sample-batch-size",
        dest="score_batch_size",
        type=int,
        default=5,
        help="Maximum images per reward batch; generation is sequential.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--max-timesteps", type=float, default=1.57080)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--reward-dtype",
        "--pickscore-dtype",
        dest="reward_dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp32",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load models and prompts, run distributed evaluation, and save JSON.

    Args:
        argv: Optional explicit CLI arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code zero after successful evaluation.
    """
    args = build_parser().parse_args(argv)
    if args.max_timesteps <= 0:
        raise ValueError("--max-timesteps must be positive")
    accelerator = Accelerator(cpu=args.device == "cpu")
    if args.device == "cuda" and accelerator.device.type != "cuda":
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")

    output_dir = resolve_output_dir(args.output_dir, args.checkpoint)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    setup_logging(output_dir, accelerator)

    config_path = require_local_file(args.config, "SANA-Sprint config")
    checkpoint_path = require_local_file(args.checkpoint, "SANA-Sprint checkpoint")
    hps_base_checkpoint = require_local_file(
        args.hps_base_checkpoint, "HPS OpenCLIP base checkpoint"
    )
    hps_checkpoint = require_local_file(args.hps_checkpoint, "HPS v2.1 checkpoint")
    aesthetic_checkpoint = require_local_file(
        args.aesthetic_checkpoint, "LAION aesthetic checkpoint"
    )
    logger.info(
        "Loading prompts from %s with max_prompts=%s, randomize=%s, seed=%d",
        args.prompt_file,
        args.max_prompts,
        args.randomize_prompts,
        args.prompt_order_seed,
    )
    prompts = load_parti_prompts(
        args.prompt_file,
        max_prompts=args.max_prompts,
        randomize=args.randomize_prompts,
        random_seed=args.prompt_order_seed,
    )
    pipe = load_sana_pipeline(
        config_path,
        checkpoint_path,
        device=accelerator.device,
        max_timesteps=args.max_timesteps,
        adapter_path=args.adapter,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
    )
    reward_dtype = dtype_from_name(args.reward_dtype, accelerator.device)
    logger.info("Loading PickScore, HPS v2, CLIP, Aesthetics, and ImageReward")
    clip_reward = CLIPReward(
        args.clip_model,
        processor_name_or_path=args.clip_model,
        device=accelerator.device,
        dtype=reward_dtype,
        local_files_only=True,
    )
    scorers: dict[str, RewardEvaluator] = {
        "pickscore": PickScoreReward(
            model_name_or_path=args.pickscore_path,
            processor_name_or_path=args.pickscore_processor_path,
            device=accelerator.device,
            dtype=reward_dtype,
            local_files_only=True,
        ),
        "hpsv2": HPSv21Reward(
            hps_checkpoint,
            model_checkpoint_path=hps_base_checkpoint,
            device=accelerator.device,
            dtype=reward_dtype,
        ),
        "clip": clip_reward,
        "aesthetics": LAIONAestheticReward(
            aesthetic_checkpoint,
            clip_model_name_or_path=args.clip_model,
            processor_name_or_path=args.clip_model,
            device=accelerator.device,
            dtype=reward_dtype,
            local_files_only=True,
        ),
        "imagereward": ImageRewardEvaluator(
            checkpoint_root=args.imagereward_root,
            tokenizer_name_or_path=args.imagereward_tokenizer,
            device=accelerator.device,
            dtype=reward_dtype,
            local_files_only=True,
        ),
    }
    logger.info(
        "Evaluating %d prompts with %d images each on %d process(es)",
        len(prompts),
        args.images_per_prompt,
        accelerator.num_processes,
    )
    summary = evaluate_prompts(
        prompts=prompts,
        pipe=pipe,
        scorers=scorers,
        accelerator=accelerator,
        output_dir=output_dir,
        images_per_prompt=args.images_per_prompt,
        score_batch_size=args.score_batch_size,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.base_seed,
        save_images=args.save_images,
    )

    if accelerator.is_main_process:
        summary.update(
            {
                "prompt_file": str(Path(args.prompt_file).expanduser().resolve()),
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "adapter": (
                    str(Path(args.adapter).expanduser().resolve())
                    if args.adapter is not None
                    else None
                ),
                "text_encoder_path": args.text_encoder_path,
                "vae_path": args.vae_path,
                "pickscore_path": args.pickscore_path,
                "pickscore_processor_path": args.pickscore_processor_path,
                "reward_dtype": args.reward_dtype,
                "hps_base_checkpoint": str(hps_base_checkpoint),
                "hps_checkpoint": str(hps_checkpoint),
                "clip_model": args.clip_model,
                "aesthetic_checkpoint": str(aesthetic_checkpoint),
                "imagereward_root": args.imagereward_root,
                "num_processes": accelerator.num_processes,
                "resolution": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "max_timesteps": args.max_timesteps,
                "guidance_scale": args.guidance_scale,
                "score_batch_size": args.score_batch_size,
                "randomize_prompts": args.randomize_prompts,
                "prompt_order_seed": args.prompt_order_seed,
            }
        )
        results_path = output_dir / "results.json"
        results_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        reward_metrics = cast(
            dict[str, dict[str, float | int]], summary["reward_metrics"]
        )
        for reward_name, metrics in reward_metrics.items():
            print(
                f"{reward_name}: {float(metrics['average_reward']):.6f} "
                f"± {float(metrics['reward_std']):.6f} over "
                f"{int(metrics['num_generations'])} generations"
            )
            print(
                f"{reward_name} per-prompt mean: "
                f"{float(metrics['average_prompt_reward']):.6f} ± "
                f"{float(metrics['prompt_reward_std']):.6f}"
            )
        print(f"Results: {results_path}")
        logger.info("Evaluation complete; results written to %s", results_path)
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
