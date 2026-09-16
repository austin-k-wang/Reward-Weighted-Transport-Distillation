"""Algorithm-agnostic Accelerate trainer for one-step online alignment."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.geneval.metadata import load_metadata_rows, select_metadata_subset

from .checkpoint import AlignmentCheckpointManager
from .config import AlignmentConfig
from .data import PromptFileDataset, collate_prompts
from .enrichers import PopulationEnricher
from .interfaces import OnlineObjective, OneStepPolicy, RewardEvaluator
from .types import ObjectiveResult, Population, PromptBatch, RolloutRequest, TrainerState


logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """Run request-driven online objectives over one-step policy populations."""

    def __init__(
        self,
        *,
        config: AlignmentConfig,
        accelerator: Accelerator,
        policy: OneStepPolicy,
        objective: OnlineObjective,
        enricher: PopulationEnricher,
        evaluation_reward: RewardEvaluator | None = None,
    ) -> None:
        """Build optimizer, scheduler, prompt loader, and persistence helpers.

        Args:
            config: Validated complete training configuration.
            accelerator: Configured distributed Accelerate runtime.
            policy: SANA-Sprint or compatible one-step policy adapter.
            objective: Registered algorithm plug-in.
            enricher: Reward/feature services used according to requests.
            evaluation_reward: Configured held-out reward evaluator used by the
                main process, or ``None`` on other ranks.

        Returns:
            Nothing. Trainable components are prepared for distributed use.
        """
        self.config = config
        self.accelerator = accelerator
        self.policy = policy
        self.objective = objective
        self.enricher = enricher
        self.evaluation_reward = evaluation_reward
        parameters = policy.trainable_parameters()
        if not parameters:
            raise ValueError("Alignment policy exposes no trainable parameters")
        optimizer_config = config.optimizer
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
        dataset = PromptFileDataset(config.runtime.prompt_file)
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.runtime.train_batch_size,
            shuffle=True,
            num_workers=config.runtime.dataloader_num_workers,
            collate_fn=collate_prompts,
            drop_last=True,
        )
        # Accelerate advances a prepared scheduler once per process when the
        # dataloader batches are sharded. Scale scheduler steps so configured
        # warmup and training lengths remain expressed in global updates.
        scheduler_step_scale = accelerator.num_processes
        self.scheduler = get_scheduler(
            optimizer_config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=(
                optimizer_config.warmup_steps * scheduler_step_scale
            ),
            num_training_steps=(
                config.runtime.max_train_steps * scheduler_step_scale
            ),
        )
        model = getattr(policy, "model")
        model, self.optimizer, self.dataloader, self.scheduler = accelerator.prepare(
            model,
            self.optimizer,
            self.dataloader,
            self.scheduler,
        )
        set_prepared_model = getattr(policy, "set_prepared_model")
        set_prepared_model(model)
        self.model = model
        self.state = TrainerState()
        self.checkpoints = AlignmentCheckpointManager(
            config.logging.output_dir,
            accelerator,
            config,
        )
        self.metrics_path = Path(config.logging.output_dir) / "train_metrics.jsonl"
        self.evaluation_metrics_path = (
            Path(config.logging.output_dir)
            / f"heldout_{config.evaluation.provider}.jsonl"
        )
        self.evaluation_prompts: tuple[str, ...] = ()
        self.evaluation_metadata: tuple[Mapping[str, object] | None, ...] = ()
        self.evaluation_source_indices: tuple[int | None, ...] = ()
        if config.evaluation.enabled:
            evaluation_prompt_path = Path(config.evaluation.prompt_file).expanduser()
            if not evaluation_prompt_path.is_file():
                raise FileNotFoundError(
                    f"Evaluation prompt file does not exist: {evaluation_prompt_path}"
                )
            if config.evaluation.provider == "geneval":
                rows = load_metadata_rows(evaluation_prompt_path)
                count = config.evaluation.prompt_count or len(rows)
                selected = select_metadata_subset(
                    rows,
                    count=count,
                    seed=config.evaluation.seed,
                )
                self.evaluation_source_indices = tuple(
                    index for index, _ in selected
                )
                self.evaluation_metadata = tuple(row for _, row in selected)
                self.evaluation_prompts = tuple(
                    str(row["prompt"]) for _, row in selected
                )
            else:
                self.evaluation_prompts = tuple(
                    line.strip()
                    for line in evaluation_prompt_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
                self.evaluation_metadata = tuple(
                    None for _ in self.evaluation_prompts
                )
                self.evaluation_source_indices = tuple(
                    None for _ in self.evaluation_prompts
                )
            if not self.evaluation_prompts:
                raise ValueError(
                    f"Evaluation prompt file is empty: {evaluation_prompt_path}"
                )
        if config.logging.resume_from_checkpoint:
            self.state = self.checkpoints.load(
                config.logging.resume_from_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

    def collect_populations(
        self,
        batch: PromptBatch,
    ) -> dict[str, Population]:
        """Execute objective requests and enrich each generated population.

        Args:
            batch: Prompt conditions for one trainer micro-step.

        Returns:
            Population mapping keyed by each unique request name.

        Raises:
            ValueError: If request names repeat or shared-noise cardinalities
                disagree.
        """
        requests = self.objective.rollout_requests(self.config)
        populations: dict[str, Population] = {}
        shared_noise: dict[str, torch.Tensor] = {}
        for request in requests:
            if request.name in populations:
                raise ValueError(f"Duplicate rollout request name: {request.name}")
            repeated_metadata = (
                tuple(
                    row
                    for row in batch.metadata
                    for _ in range(request.samples_per_prompt)
                )
                if batch.metadata
                else None
            )
            noise = None
            if request.shared_noise_group is not None:
                noise = shared_noise.get(request.shared_noise_group)
                if noise is not None:
                    expected = len(batch.prompts) * request.samples_per_prompt
                    if noise.shape[0] != expected:
                        raise ValueError(
                            f"Shared noise group {request.shared_noise_group!r} "
                            "has incompatible population counts"
                        )
            rollout = self.policy.rollout(
                batch.prompts,
                samples_per_prompt=request.samples_per_prompt,
                trainable=request.trainable,
                initial_noise=noise,
            )
            rollout.name = request.name
            if request.shared_noise_group is not None and noise is None:
                shared_noise[request.shared_noise_group] = (
                    rollout.initial_noise.detach().clone()
                )
            populations[request.name] = self.enricher.enrich(
                rollout,
                request,
                metadata=repeated_metadata,
            )
        return populations

    def _backward_two_pass_objective(self, batch: PromptBatch) -> ObjectiveResult:
        """Prepare detached objective targets, then backpropagate by chunk.

        The first pass samples complete current/reference populations without
        retaining autograd graphs. The second pass regenerates the active-policy
        current population from exactly the same noise and immediately
        backpropagates each feature-loss chunk.

        Args:
            batch: Prompt conditions for one trainer micro-step.

        Returns:
            Detached aggregate loss and full-population objective diagnostics.
            Model gradients have already accumulated when this method returns.

        Raises:
            RuntimeError: If the selected objective does not expose the
                two-pass target and chunk-loss methods.
        """
        prepare_targets = getattr(self.objective, "prepare_targets", None)
        chunk_loss = getattr(self.objective, "chunk_loss", None)
        chunk_rollout_loss = getattr(self.objective, "chunk_rollout_loss", None)
        if not callable(prepare_targets) or not (
            callable(chunk_loss) or callable(chunk_rollout_loss)
        ):
            raise RuntimeError("Objective does not implement two-pass training")

        requests = self.objective.rollout_requests(self.config)
        current_requests = [
            request
            for request in requests
            if request.name == "current" and request.trainable
        ]
        if len(current_requests) != 1:
            raise RuntimeError(
                "Two-pass training requires exactly one trainable current request"
            )
        current_request = current_requests[0]

        with torch.no_grad():
            populations = self.collect_populations(batch)
            prepared = prepare_targets(batch, populations)
            current_rollout = populations["current"].rollout
            initial_noise = current_rollout.initial_noise.detach()
            repeated_prompts = current_rollout.prompts
        del populations, current_rollout

        total_samples = len(repeated_prompts)
        if initial_noise.shape[0] != total_samples:
            raise RuntimeError("Current prompts and saved noise do not align")
        chunk_size = self.config.model.rollout_chunk_size
        total_loss = initial_noise.new_zeros(())
        feature_request = RolloutRequest(
            name="current",
            samples_per_prompt=1,
            trainable=True,
            requires_rewards=False,
            feature_names=current_request.feature_names,
        )
        for start in range(0, total_samples, chunk_size):
            stop = min(start + chunk_size, total_samples)
            rollout = self.policy.rollout(
                repeated_prompts[start:stop],
                samples_per_prompt=1,
                trainable=True,
                initial_noise=initial_noise[start:stop],
            )
            population = self.enricher.enrich(
                rollout,
                feature_request,
                metadata=None,
            )
            if callable(chunk_rollout_loss):
                partial_loss = chunk_rollout_loss(
                    population.rollout,
                    prepared,
                    start=start,
                    stop=stop,
                )
            else:
                partial_loss = chunk_loss(
                    population.features,
                    prepared,
                    start=start,
                    stop=stop,
                )
            self.accelerator.backward(partial_loss)
            total_loss = total_loss + partial_loss.detach()
            del rollout, population, partial_loss
        return ObjectiveResult(loss=total_loss, metrics=prepared.metrics)

    def _evaluate_periodic(self) -> dict[str, float]:
        """Evaluate the active policy on a fixed held-out sample and noise.

        The main process temporarily uses the unwrapped active-LoRA model while
        other ranks wait at a metrics broadcast. A device-local generator
        recreates the same complete latent population at every evaluation step
        without perturbing training RNG state.

        Returns:
            Provider-prefixed aggregate metrics broadcast to every rank.

        Raises:
            RuntimeError: If the main process lacks a configured evaluator or
                the policy cannot sample explicit initial noise.
        """
        payload: list[dict[str, float] | None] = [None]
        if self.accelerator.is_main_process:
            if self.evaluation_reward is None:
                raise RuntimeError(
                    "Main process requires evaluation_reward when evaluation is enabled"
                )
            sample_noise = getattr(self.policy, "sample_initial_noise", None)
            if not callable(sample_noise):
                raise RuntimeError(
                    "Periodic evaluation requires policy.sample_initial_noise"
                )
            evaluation = self.config.evaluation
            provider = evaluation.provider
            prompt_count = len(self.evaluation_prompts)
            samples_per_prompt = evaluation.samples_per_prompt
            total_images = prompt_count * samples_per_prompt
            generator = torch.Generator(device=self.policy.device).manual_seed(
                evaluation.seed
            )
            initial_noise = sample_noise(total_images, generator=generator)
            wrapped_model = self.model
            unwrapped_model = self.accelerator.unwrap_model(wrapped_model)
            was_training = unwrapped_model.training
            images: list[torch.Tensor] = []
            repeated_prompts: list[str] = []
            repeated_metadata: list[Mapping[str, object] | None] = []
            repeated_source_indices: list[int | None] = []
            try:
                set_prepared_model = getattr(self.policy, "set_prepared_model")
                set_prepared_model(unwrapped_model)
                unwrapped_model.eval()
                starts = range(0, prompt_count, evaluation.prompt_batch_size)
                progress = tqdm(
                    starts,
                    total=(
                        prompt_count + evaluation.prompt_batch_size - 1
                    )
                    // evaluation.prompt_batch_size,
                    desc=f"{provider} eval step {self.state.global_step}",
                    unit="batch",
                )
                with torch.inference_mode():
                    for start in progress:
                        stop = min(
                            start + evaluation.prompt_batch_size,
                            prompt_count,
                        )
                        noise_start = start * samples_per_prompt
                        noise_stop = stop * samples_per_prompt
                        rollout = self.policy.rollout(
                            self.evaluation_prompts[start:stop],
                            samples_per_prompt=samples_per_prompt,
                            trainable=True,
                            initial_noise=initial_noise[noise_start:noise_stop],
                        )
                        images.append(rollout.images.detach().cpu())
                        repeated_prompts.extend(rollout.prompts)
                        for metadata, source_index in zip(
                            self.evaluation_metadata[start:stop],
                            self.evaluation_source_indices[start:stop],
                            strict=True,
                        ):
                            repeated_metadata.extend(
                                [metadata] * samples_per_prompt
                            )
                            repeated_source_indices.extend(
                                [source_index] * samples_per_prompt
                            )
                image_batch = torch.cat(images, dim=0)
                if provider == "geneval":
                    score_with_mode = getattr(
                        self.evaluation_reward,
                        "score_with_mode",
                        None,
                    )
                    if not callable(score_with_mode):
                        raise RuntimeError(
                            "GenEval evaluation requires score_with_mode"
                        )
                    scores = score_with_mode(
                        repeated_prompts,
                        image_batch,
                        batch_size=evaluation.reward_batch_size,
                        metadata=repeated_metadata,
                        reward_mode=evaluation.reward_mode,
                        binary_bonus=evaluation.binary_bonus,
                    ).detach().float().cpu()
                else:
                    scores = self.evaluation_reward.score(
                        repeated_prompts,
                        image_batch,
                        batch_size=evaluation.reward_batch_size,
                        metadata=None,
                    ).detach().float().cpu()
            finally:
                set_prepared_model = getattr(self.policy, "set_prepared_model")
                set_prepared_model(wrapped_model)
                unwrapped_model.train(was_training)
            if scores.shape != (total_images,) or not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"Held-out {provider} returned invalid reward shape or values"
                )
            image_directory = self._save_evaluation_images(
                image_batch,
                repeated_prompts,
                scores,
                repeated_metadata,
                repeated_source_indices,
            )
            prefix = f"eval/{provider}"
            metrics = {
                f"{prefix}_mean": float(scores.mean()),
                f"{prefix}_std": float(scores.std(unbiased=False)),
                f"{prefix}_min": float(scores.min()),
                f"{prefix}_max": float(scores.max()),
                f"{prefix}_count": float(scores.numel()),
                f"{prefix}_seed": float(evaluation.seed),
            }
            if provider == "geneval":
                for tag in sorted(
                    {
                        str(row["tag"])
                        for row in repeated_metadata
                        if row is not None
                    }
                ):
                    tag_scores = torch.tensor(
                        [
                            float(score)
                            for score, row in zip(
                                scores,
                                repeated_metadata,
                                strict=True,
                            )
                            if row is not None and row["tag"] == tag
                        ],
                        dtype=torch.float32,
                    )
                    metrics[f"{prefix}_{tag}_mean"] = float(tag_scores.mean())
            record = {
                "step": self.state.global_step,
                "provider": provider,
                "prompts": list(repeated_prompts),
                "source_indices": repeated_source_indices,
                "scores": scores.tolist(),
                "image_directory": str(image_directory),
                **metrics,
            }
            with self.evaluation_metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            payload[0] = metrics
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        metrics = payload[0]
        if metrics is None:
            raise RuntimeError("Held-out evaluation metrics were not broadcast")
        return metrics

    def _save_evaluation_images(
        self,
        images: torch.Tensor,
        prompts: list[str],
        scores: torch.Tensor,
        metadata: list[Mapping[str, object] | None],
        source_indices: list[int | None],
    ) -> Path:
        """Save one deterministic held-out image directory and manifest.

        Args:
            images: CPU image tensor shaped ``[N,C,H,W]`` in approximate
                ``[-1,1]`` range.
            prompts: Prompt strings aligned with the leading image dimension.
            scores: CPU reward values shaped ``[N]``.
            metadata: Optional structured rows aligned with images.
            source_indices: Optional source-file indices aligned with images.

        Returns:
            Created directory
            ``OUTPUT_DIR/periodic-eval/step-{global_step:08d}``.

        Raises:
            ValueError: If image, prompt, score, or channel cardinalities differ.
        """
        if (
            images.ndim != 4
            or images.shape[0] != len(prompts)
            or scores.shape != (images.shape[0],)
            or len(metadata) != images.shape[0]
            or len(source_indices) != images.shape[0]
        ):
            raise ValueError("Periodic evaluation inputs must align")
        image_directory = (
            Path(self.config.logging.output_dir)
            / "periodic-eval"
            / f"step-{self.state.global_step:08d}"
        )
        image_directory.mkdir(parents=True, exist_ok=True)
        manifest = []
        progress = tqdm(
            enumerate(
                zip(
                    images,
                    prompts,
                    scores,
                    metadata,
                    source_indices,
                    strict=True,
                )
            ),
            total=images.shape[0],
            desc=(
                "Saving "
                f"{self.config.evaluation.provider} eval step "
                f"{self.state.global_step}"
            ),
            unit="image",
        )
        provider = self.config.evaluation.provider
        for index, (
            image,
            prompt,
            score,
            metadata_row,
            source_index,
        ) in progress:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            if image.shape[0] != 3:
                raise ValueError(
                    f"Periodic evaluation image {index} has {image.shape[0]} channels"
                )
            pixels = (
                image.detach()
                .float()
                .clamp(-1, 1)
                .add(1)
                .mul(127.5)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )
            filename = f"{index:05d}.png"
            Image.fromarray(pixels).save(image_directory / filename)
            image_record = {
                "index": index,
                "filename": filename,
                "prompt": prompt,
                f"{provider}_reward": float(score),
            }
            if source_index is not None:
                image_record["source_index"] = source_index
            if metadata_row is not None:
                image_record["metadata"] = dict(metadata_row)
            if provider == "geneval" and self.config.evaluation.reward_mode == "binary":
                image_record["official_correct"] = bool(score)
            manifest.append(image_record)
        result = {
            "step": self.state.global_step,
            "provider": provider,
            "seed": self.config.evaluation.seed,
            f"{provider}_mean": float(scores.mean()),
            f"{provider}_std": float(scores.std(unbiased=False)),
            f"{provider}_min": float(scores.min()),
            f"{provider}_max": float(scores.max()),
            f"{provider}_count": int(scores.numel()),
            "images": manifest,
        }
        (image_directory / "results.json").write_text(
            json.dumps(result, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return image_directory

    def train(self) -> TrainerState:
        """Run optimizer steps until ``max_train_steps`` and save checkpoints.

        Returns:
            Final global/micro-step state after all synchronized updates.
        """
        output_dir = Path(self.config.logging.output_dir)
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        progress = tqdm(
            total=self.config.runtime.max_train_steps,
            initial=self.state.global_step,
            disable=not self.accelerator.is_local_main_process,
            desc="SANA alignment",
        )
        self.model.train()
        if self.config.evaluation.enabled and self.state.global_step == 0:
            initial_evaluation_metrics = self._evaluate_periodic()
            self._log(initial_evaluation_metrics)
        while self.state.global_step < self.config.runtime.max_train_steps:
            for batch in self.dataloader:
                assert isinstance(batch, PromptBatch)
                with self.accelerator.accumulate(self.model):
                    if (
                        callable(getattr(self.objective, "prepare_targets", None))
                        and (
                            callable(getattr(self.objective, "chunk_loss", None))
                            or callable(
                                getattr(self.objective, "chunk_rollout_loss", None)
                            )
                        )
                    ):
                        result = self._backward_two_pass_objective(batch)
                    else:
                        populations = self.collect_populations(batch)
                        result = self.objective.compute(batch, populations)
                        self.accelerator.backward(result.loss)
                    grad_norm = result.loss.new_zeros(())
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.policy.trainable_parameters(),
                            self.config.optimizer.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.state.micro_step += 1
                if not self.accelerator.sync_gradients:
                    continue
                self.state.global_step += 1
                progress.update(1)
                metrics = dict(result.metrics)
                metrics.update(
                    {
                        "train/loss": result.loss.detach(),
                        "train/grad_norm": grad_norm.detach()
                        if torch.is_tensor(grad_norm)
                        else float(grad_norm),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )
                if self.accelerator.device.type == "cuda":
                    metrics["train/peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated(self.accelerator.device)
                        / 1024**3
                    )
                evaluated = (
                    self.config.evaluation.enabled
                    and self.state.global_step
                    % self.config.evaluation.interval_steps
                    == 0
                )
                if evaluated:
                    metrics.update(self._evaluate_periodic())
                if (
                    self.state.global_step % self.config.logging.logging_steps == 0
                    or evaluated
                ):
                    self._log(metrics)
                if (
                    self.state.global_step
                    % self.config.logging.checkpointing_steps
                    == 0
                ):
                    self.checkpoints.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        state=self.state,
                    )
                if self.state.global_step >= self.config.runtime.max_train_steps:
                    break
        progress.close()
        return self.state

    def _log(self, metrics: dict[str, torch.Tensor | float]) -> None:
        """Reduce, track, and append one synchronized metric record.

        Args:
            metrics: Scalar local tensors or Python numbers.

        Returns:
            Nothing. Main-process metrics are appended to JSONL and all
            configured Accelerate trackers receive the same values.
        """
        reduced: dict[str, float] = {}
        for name, value in metrics.items():
            tensor = (
                value.detach().float().reshape(())
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=self.accelerator.device)
            )
            reduced[name] = float(self.accelerator.reduce(tensor, reduction="mean").cpu())
        self.accelerator.log(reduced, step=self.state.global_step)
        if self.accelerator.is_main_process:
            record = {"step": self.state.global_step, **reduced}
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            logger.info("step=%d metrics=%s", self.state.global_step, json.dumps(reduced, sort_keys=True))
