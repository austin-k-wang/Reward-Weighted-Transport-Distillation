"""Train SANA-Sprint LoRA adapters with a modular online objective."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alignment.config import (  # noqa: E402
    apply_alignment_overrides,
    load_alignment_config,
)
from src.alignment.data import PromptFileDataset  # noqa: E402
from src.alignment.enrichers import PopulationEnricher  # noqa: E402
from src.alignment.generators import SanaSprintPolicy  # noqa: E402
from src.alignment.objectives import build_objective  # noqa: E402
from src.alignment.trainer import AlignmentTrainer  # noqa: E402
from src.dinov2 import DINOv2FeatureEncoder  # noqa: E402
from src.geneval import GenEvalReward  # noqa: E402
from src.geneval.metadata import (  # noqa: E402
    load_metadata_rows,
    select_metadata_subset,
)
from src.pickscore import PickScore  # noqa: E402
from src.rewards import (  # noqa: E402
    HPSv21FeatureEncoder,
    HPSv21Reward,
    create_reward_from_config,
)


logger = logging.getLogger(__name__)


def _dtype(name: str) -> torch.dtype:
    """Resolve a user-facing dtype name into a PyTorch dtype.

    Args:
        name: One of ``float32``, ``fp32``, ``float16``, ``fp16``, ``bfloat16``,
            or ``bf16``.

    Returns:
        Corresponding floating-point PyTorch dtype.

    Raises:
        ValueError: If ``name`` is unsupported.
    """
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def _setup_logging(output_dir: Path, accelerator: Accelerator) -> None:
    """Configure rank-aware console and persistent training logs.

    Args:
        output_dir: Run directory that receives ``train.log``.
        accelerator: Runtime identifying the main process.

    Returns:
        Nothing. The root logger is configured for immediate output.
    """
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if accelerator.is_main_process:
        handlers.append(logging.FileHandler(output_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _geneval_socket_path(template: str, accelerator: Accelerator) -> str:
    """Resolve a rank-aware GenEval Unix-socket path.

    Args:
        template: Configured path supporting ``{local_rank}`` and ``{rank}``
            placeholders.
        accelerator: Distributed runtime providing local and global indices.

    Returns:
        Concrete socket path for the current training process.

    Raises:
        ValueError: If the template contains an unsupported placeholder.
    """
    try:
        return template.format(
            local_rank=accelerator.local_process_index,
            rank=accelerator.process_index,
        )
    except KeyError as exc:
        raise ValueError(
            "reward.socket_path supports only {local_rank} and {rank}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the alignment training command-line parser.

    Returns:
        Parser requiring one YAML configuration path.
    """
    parser = argparse.ArgumentParser(
        description="Train one-step SANA-Sprint with a modular online objective."
    )
    parser.add_argument("--config", required=True, help="Alignment YAML path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override one YAML value; may be repeated.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional adapter checkpoint override for this launch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load configured components and execute online-alignment training.

    Args:
        argv: Optional explicit command-line arguments.

    Returns:
        Process exit code zero after successful training.
    """
    args = build_parser().parse_args(argv)
    config = load_alignment_config(args.config)
    config = apply_alignment_overrides(config, args.overrides)
    if args.resume_from_checkpoint is not None:
        config = replace(
            config,
            logging=replace(
                config.logging,
                resume_from_checkpoint=args.resume_from_checkpoint,
            ),
        )
    accelerator = Accelerator(
        mixed_precision=config.runtime.mixed_precision,
        gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
        log_with=None if config.logging.report_to == "none" else config.logging.report_to,
        project_dir=config.logging.output_dir,
    )
    output_dir = Path(config.logging.output_dir)
    _setup_logging(output_dir, accelerator)
    set_seed(config.runtime.seed + accelerator.process_index)
    if config.logging.report_to != "none":
        accelerator.init_trackers(
            "sana-sprint-alignment",
            config={
                name: json.dumps(values, sort_keys=True)
                for name, values in config.to_dict().items()
            },
        )

    logger.info("Loading native SANA-Sprint policy")
    policy = SanaSprintPolicy.from_config(config, device=accelerator.device)
    if config.model.cache_text_embeddings:
        prompt_dataset = PromptFileDataset(config.runtime.prompt_file)
        cached_prompts = [prompt for prompt, _ in prompt_dataset.items]
        if config.evaluation.enabled:
            evaluation_path = Path(config.evaluation.prompt_file).expanduser()
            if config.evaluation.provider == "geneval":
                evaluation_rows = load_metadata_rows(evaluation_path)
                selected_rows = select_metadata_subset(
                    evaluation_rows,
                    count=(
                        config.evaluation.prompt_count
                        or len(evaluation_rows)
                    ),
                    seed=config.evaluation.seed,
                )
                cached_prompts.extend(
                    str(row["prompt"]) for _, row in selected_rows
                )
            else:
                cached_prompts.extend(
                    line.strip()
                    for line in evaluation_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
        logger.info(
            "Pre-encoding %d training/evaluation prompts before loading auxiliaries",
            len(cached_prompts),
        )
        cache_signature = json.dumps(
            {
                "text_encoder_path": config.model.text_encoder_path,
                "chi_prompt": policy.config.text_encoder.chi_prompt,
                "model_max_length": policy.config.text_encoder.model_max_length,
            },
            sort_keys=True,
        )
        if accelerator.is_main_process:
            policy.cache_prompt_embeddings(
                cached_prompts,
                batch_size=config.model.text_encoding_batch_size,
                cache_path=config.model.text_embedding_cache_path,
                cache_signature=cache_signature,
                rebuild=config.model.rebuild_text_embedding_cache,
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            policy.cache_prompt_embeddings(
                cached_prompts,
                batch_size=config.model.text_encoding_batch_size,
                cache_path=config.model.text_embedding_cache_path,
                cache_signature=cache_signature,
                rebuild=False,
            )
        del prompt_dataset, cached_prompts
    reward = None
    if config.reward.enabled:
        socket_path = None
        if config.reward.provider == "geneval":
            socket_path = _geneval_socket_path(
                config.reward.socket_path,
                accelerator,
            )
            logger.info("Connecting to GenEval scorer at %s", socket_path)
        else:
            logger.info("Loading %s reward evaluator", config.reward.provider)
        reward = create_reward_from_config(
            config.reward,
            device=accelerator.device,
            dtype=_dtype(config.reward.dtype),
            socket_path=socket_path,
        )
    features = None
    if config.features.enabled:
        if config.features.provider == "dinov2":
            features = DINOv2FeatureEncoder(
                config.features.model_path,
                device=accelerator.device,
                dtype=_dtype(config.features.dtype),
                local_files_only=True,
            )
        elif config.features.provider == "hpsv2":
            if not isinstance(reward, HPSv21Reward):
                raise TypeError(
                    "features.provider=hpsv2 requires the reward evaluator to be "
                    "HPSv21Reward"
                )
            features = HPSv21FeatureEncoder(reward)
        else:
            raise ValueError(f"Unsupported feature provider: {config.features.provider}")
    evaluation_reward = None
    if config.evaluation.enabled and accelerator.is_main_process:
        if config.evaluation.provider == "geneval":
            if not isinstance(reward, GenEvalReward):
                raise RuntimeError(
                    "GenEval evaluation requires the GenEval training reward"
                )
            logger.info(
                "Reusing rank-zero GenEval scorer for %s",
                config.evaluation.prompt_file,
            )
            evaluation_reward = reward
        else:
            logger.info(
                "Loading held-out PickScore evaluator for %s",
                config.evaluation.prompt_file,
            )
            evaluation_reward = PickScore(
                model_name_or_path=config.evaluation.model_path,
                processor_name_or_path=config.evaluation.processor_path,
                device=accelerator.device,
                dtype=_dtype(config.evaluation.dtype),
            )
    enricher = PopulationEnricher(
        reward=reward,
        features=features,
        reward_batch_size=config.reward.batch_size,
        feature_batch_size=config.model.rollout_chunk_size,
    )
    objective = build_objective(config)
    configure_runtime = getattr(objective, "configure_runtime", None)
    if callable(configure_runtime):
        configure_runtime(policy=policy, reward=reward)
    trainer = AlignmentTrainer(
        config=config,
        accelerator=accelerator,
        policy=policy,
        objective=objective,
        enricher=enricher,
        evaluation_reward=evaluation_reward,
    )
    final_state = trainer.train()
    logger.info("Training complete at global_step=%d", final_state.global_step)
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
