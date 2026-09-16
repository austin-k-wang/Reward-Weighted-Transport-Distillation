"""Integration tests for request collection, training, and adapter checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from src.alignment.checkpoint import AlignmentCheckpointManager
from src.alignment.config import (
    AlignmentConfig,
    EvaluationConfig,
    FeatureConfig,
    LoggingConfig,
    ModelConfig,
    ObjectiveConfig,
    RWTDConfig,
    RewardConfig,
    RuntimeConfig,
)
from src.alignment.enrichers import PopulationEnricher
from src.alignment.objectives import build_objective
from src.alignment.trainer import AlignmentTrainer
from src.alignment.types import RolloutResult, TrainerState
from src.geneval.metadata import load_metadata_rows, select_metadata_subset


class TinyPeftBase(torch.nn.Module):
    """Tiny linear base model compatible with PEFT checkpoint tests."""

    def __init__(self) -> None:
        """Initialize one two-dimensional linear projection."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project a tensor through the tiny linear layer.

        Args:
            values: Tensor whose final dimension is two.

        Returns:
            Tensor with the same shape as ``values``.
        """
        return self.linear(values)


class ScalarPolicyModel(torch.nn.Module):
    """Single-parameter model used by the generic trainer test."""

    def __init__(self) -> None:
        """Initialize one trainable scalar adapter."""
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Scale generated noise by the trainable adapter.

        Args:
            noise: Population tensor of any shape.

        Returns:
            Scaled tensor with the same shape.
        """
        return noise * self.adapter


class FakePolicy:
    """Minimal request-driven policy implementing the trainer protocol."""

    def __init__(self) -> None:
        """Initialize a scalar policy and rollout-noise trace."""
        self.model = ScalarPolicyModel()
        self.seen_noise: list[torch.Tensor] = []

    @property
    def device(self) -> torch.device:
        """Return the scalar model device.

        Returns:
            CPU device used by this test policy.
        """
        return next(self.model.parameters()).device

    def set_prepared_model(self, model: torch.nn.Module) -> None:
        """Install the model returned by Accelerate.

        Args:
            model: Prepared scalar policy model.

        Returns:
            Nothing. Future rollouts use ``model``.
        """
        self.model = model

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return the scalar adapter parameter.

        Returns:
            List containing one trainable parameter.
        """
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def sample_initial_noise(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample deterministic fake rollout noise.

        Args:
            count: Number of scalar image particles.
            generator: Optional CPU generator controlling exact values.

        Returns:
            Standard-normal noise shaped ``[count,1,1,1]``.
        """
        return torch.randn(count, 1, 1, 1, generator=generator)

    def rollout(
        self,
        prompts: tuple[str, ...],
        *,
        samples_per_prompt: int,
        trainable: bool,
        initial_noise: torch.Tensor | None = None,
    ) -> RolloutResult:
        """Generate scalar fake images with optional matched initial noise.

        Args:
            prompts: Prompt batch.
            samples_per_prompt: Particle count per prompt.
            trainable: Whether to use the active scalar adapter.
            initial_noise: Optional shared population tensor.

        Returns:
            Fake rollout with tensors shaped ``[N,1,1,1]``.
        """
        count = len(prompts) * samples_per_prompt
        noise = torch.ones(count, 1, 1, 1) if initial_noise is None else initial_noise
        self.seen_noise.append(noise.detach().clone())
        if trainable:
            images = self.model(noise)
        else:
            with torch.no_grad():
                images = torch.zeros_like(noise)
        repeated = tuple(prompt for prompt in prompts for _ in range(samples_per_prompt))
        return RolloutResult(
            name="current" if trainable else "reference",
            prompts=repeated,
            initial_noise=noise,
            denoised_latents=images,
            images=images,
        )


class MeanFeatureEncoder:
    """Convert each fake image into one differentiable scalar CLS block."""

    def vector_features(
        self,
        images: torch.Tensor,
        features: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Average spatial/channel dimensions for requested feature names.

        Args:
            images: Fake population tensor shaped ``[N,C,H,W]``.
            features: Requested names; this fake supports ``cls``.

        Returns:
            Mapping from each requested name to ``[N,1,1]`` values.
        """
        names = ("cls",) if features is None else tuple(features)
        value = images.flatten(1).mean(dim=1).view(-1, 1, 1)
        return {name: value for name in names}


class RecordingReward:
    """Return image means and retain inputs for reproducibility assertions."""

    def __init__(self) -> None:
        """Initialize an empty image-call trace.

        Returns:
            Nothing.
        """
        self.images: list[torch.Tensor] = []

    def score(
        self,
        prompts: tuple[str, ...] | list[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: object = None,
    ) -> torch.Tensor:
        """Score fake images by scalar mean.

        Args:
            prompts: Prompt strings aligned with ``images``.
            images: Fake images shaped ``[N,1,1,1]``.
            batch_size: Positive compatibility argument.
            metadata: Unused optional structured metadata.

        Returns:
            Detached scalar image means shaped ``[N]``.
        """
        del prompts, batch_size, metadata
        self.images.append(images.detach().clone())
        return images.flatten(1).mean(dim=1)


class RecordingGenEvalReward(RecordingReward):
    """Record structured metadata passed to periodic GenEval evaluation."""

    def __init__(self) -> None:
        """Initialize image and metadata call traces."""
        super().__init__()
        self.metadata: list[list[dict[str, object] | None]] = []

    def score_with_mode(
        self,
        prompts: tuple[str, ...] | list[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: list[dict[str, object] | None],
        reward_mode: str,
        binary_bonus: float,
    ) -> torch.Tensor:
        """Return deterministic binary scores and retain structured rows.

        Args:
            prompts: Prompt strings aligned with ``images``.
            images: Fake image tensor shaped ``[N,1,1,1]``.
            batch_size: Positive compatibility argument.
            metadata: Structured GenEval rows aligned with images.
            reward_mode: Requested evaluation reward mode.
            binary_bonus: Requested hybrid bonus.

        Returns:
            Binary score tensor shaped ``[N]``.
        """
        del prompts, batch_size, binary_bonus
        assert reward_mode == "binary"
        self.images.append(images.detach().clone())
        self.metadata.append(list(metadata))
        return (images.flatten(1).mean(dim=1) > 0).float()


def test_checkpoint_round_trip_restores_lora_and_state(tmp_path: Path) -> None:
    """Verify adapter-only checkpoints restore weights and trainer counters.

    Args:
        tmp_path: Pytest temporary checkpoint directory.

    Returns:
        Nothing. The test mutates and restores a tiny PEFT adapter.
    """
    accelerator = Accelerator(cpu=True)
    model = get_peft_model(
        TinyPeftBase(),
        PeftLoraConfig(r=1, lora_alpha=1, target_modules=["linear"]),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = AlignmentConfig(logging=LoggingConfig(output_dir=str(tmp_path)))
    manager = AlignmentCheckpointManager(tmp_path, accelerator, config)
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainerState(global_step=3, micro_step=5),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(10)
    state = manager.load(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert state == TrainerState(global_step=3, micro_step=5)
    for name, parameter in model.named_parameters():
        if name in original:
            torch.testing.assert_close(parameter, original[name])


def test_rwtd_trainer_updates_policy(tmp_path: Path) -> None:
    """Verify the generic trainer runs one RWTD optimizer step.

    Args:
        tmp_path: Pytest temporary directory for prompts and logs.

    Returns:
        Nothing. The test checks progress, metrics, and a policy update.
    """
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a test image\n", encoding="utf-8")
    config = AlignmentConfig(
        features=FeatureConfig(names=("cls",)),
        evaluation=EvaluationConfig(enabled=False),
        objective=ObjectiveConfig(
            current_samples=1,
            reference_samples=1,
        ),
        runtime=RuntimeConfig(
            prompt_file=str(prompt_file),
            max_train_steps=1,
            mixed_precision="no",
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "run"),
            checkpointing_steps=99,
            report_to="none",
        ),
    )
    policy = FakePolicy()
    initial_value = policy.model.adapter.detach().clone()
    accelerator = Accelerator(cpu=True)
    trainer = AlignmentTrainer(
        config=config,
        accelerator=accelerator,
        policy=policy,
        objective=build_objective(config),
        enricher=PopulationEnricher(
            reward=RecordingReward(),
            features=MeanFeatureEncoder(),
            feature_batch_size=1,
        ),
    )

    state = trainer.train()

    assert state.global_step == 1
    assert len(policy.seen_noise) == 3
    assert not torch.equal(policy.model.adapter.detach(), initial_value)
    assert (tmp_path / "run/train_metrics.jsonl").is_file()


def test_rwtd_trainer_regenerates_saved_noise_in_backward_chunks(
    tmp_path: Path,
) -> None:
    """Verify two-pass RWTD replays current noise and updates chunkwise.

    Args:
        tmp_path: Temporary directory for prompts and training outputs.

    Returns:
        Nothing. The test checks exact noise replay across two chunks and a
        nonzero policy update.
    """
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a test image\n", encoding="utf-8")
    config = AlignmentConfig(
        model=ModelConfig(rollout_chunk_size=2),
        reward=RewardConfig(enabled=True, batch_size=1),
        features=FeatureConfig(names=("cls",)),
        evaluation=EvaluationConfig(enabled=False),
        objective=ObjectiveConfig(
            name="rwtd",
            current_samples=3,
            reference_samples=2,
            feature_weights=(1.0,),
        ),
        runtime=RuntimeConfig(
            prompt_file=str(prompt_file),
            max_train_steps=1,
            mixed_precision="no",
            gradient_accumulation_steps=1,
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "run"),
            checkpointing_steps=99,
            report_to="none",
        ),
    )
    policy = FakePolicy()
    initial_value = policy.model.adapter.detach().clone()
    trainer = AlignmentTrainer(
        config=config,
        accelerator=Accelerator(cpu=True),
        policy=policy,
        objective=build_objective(config),
        enricher=PopulationEnricher(
            reward=RecordingReward(),
            features=MeanFeatureEncoder(),
            reward_batch_size=1,
            feature_batch_size=1,
        ),
    )

    state = trainer.train()

    assert state.global_step == 1
    replayed_noise = torch.cat((policy.seen_noise[2], policy.seen_noise[3]))
    torch.testing.assert_close(replayed_noise, policy.seen_noise[0])
    assert not torch.equal(policy.model.adapter.detach(), initial_value)


def test_periodic_pickscore_evaluation_reuses_exact_noise(tmp_path: Path) -> None:
    """Verify held-out evaluation recreates identical images and scores.

    Args:
        tmp_path: Temporary directory receiving training/evaluation prompt
            files and metric output.

    Returns:
        Nothing. Two evaluations of an unchanged model must match exactly.
    """
    train_prompts = tmp_path / "train.txt"
    eval_prompts = tmp_path / "eval.txt"
    train_prompts.write_text("training prompt\n", encoding="utf-8")
    eval_prompts.write_text("first held-out\nsecond held-out\n", encoding="utf-8")
    config = AlignmentConfig(
        features=FeatureConfig(names=("cls",)),
        evaluation=EvaluationConfig(
            enabled=True,
            interval_steps=1,
            prompt_file=str(eval_prompts),
            seed=17,
            prompt_batch_size=1,
        ),
        runtime=RuntimeConfig(
            prompt_file=str(train_prompts),
            max_train_steps=1,
            mixed_precision="no",
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "run"),
            checkpointing_steps=99,
            report_to="none",
        ),
    )
    policy = FakePolicy()
    reward = RecordingReward()
    trainer = AlignmentTrainer(
        config=config,
        accelerator=Accelerator(cpu=True),
        policy=policy,
        objective=build_objective(config),
        enricher=PopulationEnricher(
            reward=None,
            features=MeanFeatureEncoder(),
            feature_batch_size=1,
        ),
        evaluation_reward=reward,
    )
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)

    first = trainer._evaluate_periodic()
    second = trainer._evaluate_periodic()

    torch.testing.assert_close(reward.images[0], reward.images[1], rtol=0, atol=0)
    assert first == second
    records = (
        tmp_path / "run/heldout_pickscore.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    image_directory = tmp_path / "run/periodic-eval/step-00000000"
    assert len(tuple(image_directory.glob("*.png"))) == 2
    manifest = json.loads(
        (image_directory / "results.json").read_text(encoding="utf-8")
    )
    assert manifest["pickscore_count"] == 2
    assert manifest["pickscore_mean"] == first["eval/pickscore_mean"]
    assert manifest["pickscore_std"] == first["eval/pickscore_std"]
    assert [row["prompt"] for row in manifest["images"]] == [
        "first held-out",
        "second held-out",
    ]


def test_periodic_geneval_uses_reproducible_metadata_subset(
    tmp_path: Path,
) -> None:
    """Verify periodic GenEval selects 53 fixed rows and passes their metadata.

    Args:
        tmp_path: Temporary directory receiving JSONL prompts and outputs.

    Returns:
        Nothing. Selected source indices, metadata, metrics, and manifest
        contents are checked.
    """
    train_prompts = tmp_path / "train.txt"
    evaluation_rows = tmp_path / "evaluation.jsonl"
    train_prompts.write_text("training prompt\n", encoding="utf-8")
    rows = [
        {
            "prompt": f"a photo of object {index}",
            "tag": "single_object",
            "include": [{"class": f"object-{index}", "count": 1}],
        }
        for index in range(60)
    ]
    evaluation_rows.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = AlignmentConfig(
        reward=RewardConfig(enabled=True, provider="geneval"),
        features=FeatureConfig(names=("cls",)),
        evaluation=EvaluationConfig(
            enabled=True,
            provider="geneval",
            prompt_file=str(evaluation_rows),
            prompt_count=53,
            seed=99,
            prompt_batch_size=7,
            reward_mode="binary",
        ),
        runtime=RuntimeConfig(
            prompt_file=str(train_prompts),
            max_train_steps=1,
            mixed_precision="no",
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "run"),
            checkpointing_steps=99,
            report_to="none",
        ),
    )
    reward = RecordingGenEvalReward()
    trainer = AlignmentTrainer(
        config=config,
        accelerator=Accelerator(cpu=True),
        policy=FakePolicy(),
        objective=build_objective(config),
        enricher=PopulationEnricher(
            reward=reward,
            features=MeanFeatureEncoder(),
            reward_batch_size=1,
            feature_batch_size=1,
        ),
        evaluation_reward=reward,
    )

    metrics = trainer._evaluate_periodic()
    expected = select_metadata_subset(
        load_metadata_rows(evaluation_rows),
        count=53,
        seed=99,
    )

    assert trainer.evaluation_source_indices == tuple(index for index, _ in expected)
    assert len(reward.metadata[0]) == 53
    assert reward.metadata[0] == [row for _, row in expected]
    assert metrics["eval/geneval_count"] == 53
    manifest = json.loads(
        (
            tmp_path / "run/periodic-eval/step-00000000/results.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["provider"] == "geneval"
    assert manifest["geneval_count"] == 53
    assert manifest["images"][0]["metadata"]["tag"] == "single_object"
    assert (
        tmp_path / "run/heldout_geneval.jsonl"
    ).is_file()


def test_training_runs_initial_pickscore_evaluation_at_step_zero(
    tmp_path: Path,
) -> None:
    """Verify enabled held-out evaluation runs before the first optimizer step.

    Args:
        tmp_path: Temporary directory receiving prompts, images, and metrics.

    Returns:
        Nothing. Step-zero images and one held-out score record must exist even
        when the periodic interval is larger than the training run.
    """
    train_prompts = tmp_path / "train.txt"
    eval_prompts = tmp_path / "eval.txt"
    train_prompts.write_text("training prompt\n", encoding="utf-8")
    eval_prompts.write_text("held-out prompt\n", encoding="utf-8")
    config = AlignmentConfig(
        features=FeatureConfig(names=("cls",)),
        evaluation=EvaluationConfig(
            enabled=True,
            interval_steps=100,
            prompt_file=str(eval_prompts),
            prompt_batch_size=1,
        ),
        objective=ObjectiveConfig(
            current_samples=1,
            reference_samples=1,
        ),
        runtime=RuntimeConfig(
            prompt_file=str(train_prompts),
            max_train_steps=1,
            mixed_precision="no",
            gradient_accumulation_steps=1,
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "run"),
            checkpointing_steps=99,
            report_to="none",
        ),
    )
    trainer = AlignmentTrainer(
        config=config,
        accelerator=Accelerator(cpu=True),
        policy=FakePolicy(),
        objective=build_objective(config),
        enricher=PopulationEnricher(
            reward=RecordingReward(),
            features=MeanFeatureEncoder(),
            feature_batch_size=1,
        ),
        evaluation_reward=RecordingReward(),
    )

    trainer.train()

    assert (
        tmp_path / "run/periodic-eval/step-00000000/00000.png"
    ).is_file()
    records = (
        tmp_path / "run/heldout_pickscore.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
