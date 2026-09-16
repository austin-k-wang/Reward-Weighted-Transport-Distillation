"""Dataclass configuration and YAML loading for online alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    """Native SANA-Sprint model and differentiable rollout settings."""

    config_path: str = str(
        ROOT
        / "Sana/configs/sana_sprint_config/1024ms/"
        "SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
    )
    checkpoint_path: str = str(
        ROOT
        / "models/Sana_Sprint_1.6B_1024px/checkpoints/"
        "Sana_Sprint_1.6B_1024px.pth"
    )
    text_encoder_path: str | None = "Efficient-Large-Model/gemma-2-2b-it"
    vae_path: str | None = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
    local_files_only: bool = True
    resolution: int = 1024
    guidance_scale: float = 4.5
    num_inference_steps: int = 1
    max_timesteps: float = 1.57080
    rollout_chunk_size: int = 4
    decode_chunk_size: int = 1
    gradient_checkpointing: bool = True
    vae_gradient_checkpointing: bool = True
    cache_text_embeddings: bool = True
    text_encoding_batch_size: int = 8
    text_embedding_cache_path: str = str(
        ROOT / "outputs/cache/sana_sprint_gemma_embeddings.pt"
    )
    rebuild_text_embedding_cache: bool = False


@dataclass(frozen=True)
class LoRAConfig:
    """PEFT LoRA setup for the trainable SANA attention projections."""

    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "attn.qkv",
        "attn.proj",
        "cross_attn.q_linear",
        "cross_attn.kv_linear",
        "cross_attn.proj",
    )
    init_weights: str | bool = "gaussian"


@dataclass(frozen=True)
class RewardConfig:
    """Optional black-box reward model and local checkpoint settings."""

    enabled: bool = True
    provider: str = "pickscore"
    model_path: str | None = None
    processor_path: str | None = None
    checkpoint_path: str | None = None
    base_model_path: str | None = None
    local_files_only: bool = True
    components: tuple[dict[str, Any], ...] = ()
    batch_size: int = 8
    dtype: str = "float32"
    socket_path: str = "/tmp/geneval-{local_rank}.sock"
    timeout: float = 300.0
    startup_timeout: float = 900.0
    reward_mode: str = "hybrid"
    binary_bonus: float = 0.25
    warmup: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    """Periodic deterministic held-out reward evaluation settings."""

    enabled: bool = True
    provider: str = "pickscore"
    interval_steps: int = 10
    prompt_file: str = str(ROOT / "data/drawbench/alignment_eval.txt")
    prompt_count: int | None = None
    seed: int = 1234
    samples_per_prompt: int = 1
    prompt_batch_size: int = 4
    reward_batch_size: int = 8
    reward_mode: str = "binary"
    binary_bonus: float = 0.0
    model_path: str = str(ROOT / "models/PickScore_v1")
    processor_path: str = str(ROOT / "models/PickScore_v1")
    dtype: str = "float32"


@dataclass(frozen=True)
class FeatureConfig:
    """Frozen differentiable image-feature encoder settings."""

    enabled: bool = True
    provider: str = "dinov2"
    model_path: str = str(ROOT / "models/facebook-dinov2-base")
    names: tuple[str, ...] = ("cls", "patch_mean", "patch_std")
    dtype: str = "float32"


@dataclass(frozen=True)
class ObjectiveConfig:
    """Algorithm selection and generic population settings."""

    name: str = "rwtd"
    current_samples: int = 2
    reference_samples: int = 2
    matched_noise: bool = False
    feature_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class RWTDConfig:
    """Reward weighting, coupling, and partial-transport hyperparameters."""

    coupling: str = "sinkhorn"
    sinkhorn_target: str = "barycentric"
    stochastic_transport: bool = False
    reward_temperature: float = 0.5
    reward_mean: float = 0.0
    reward_scale: float = 1.0
    mass_floor: float = 0.10
    reference_fraction: float = 0.15
    transport_step: float = 0.20
    ot_regularization_scale: float = 0.10
    minimum_ot_epsilon: float = 1e-4
    sinkhorn_iterations: int = 100
    sinkhorn_tolerance: float | None = 1e-5
    displacement_clip: float | None = None
    feature_stats_path: str | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW and gradient-control settings."""

    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler: str = "constant"
    warmup_steps: int = 0


@dataclass(frozen=True)
class RuntimeConfig:
    """Distributed execution, prompt loading, and step-count settings."""

    prompt_file: str = str(ROOT / "data/pickscore/sfw_train.txt")
    train_batch_size: int = 1
    max_train_steps: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 0
    mixed_precision: str = "bf16"
    seed: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Output, progress, tracker, and checkpoint cadence settings."""

    output_dir: str = str(ROOT / "outputs/sana-sprint-alignment/rwtd")
    logging_steps: int = 1
    checkpointing_steps: int = 1
    report_to: str = "tensorboard"
    resume_from_checkpoint: str | None = None


@dataclass(frozen=True)
class AlignmentConfig:
    """Complete resolved configuration for an online-alignment run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    rwtd: RWTDConfig = field(default_factory=RWTDConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Validate cross-component invariants before loading large models.

        Raises:
            ValueError: If a resolution, count, dtype, feature selection,
                objective, LoRA, optimizer, or logging option is invalid.
        """
        if self.model.resolution < 32 or self.model.resolution % 32:
            raise ValueError("model.resolution must be positive and divisible by 32")
        if self.model.num_inference_steps != 1:
            raise ValueError("The initial SANA-Sprint alignment scaffold requires one inference step")
        for name in (
            "rollout_chunk_size",
            "decode_chunk_size",
            "text_encoding_batch_size",
        ):
            if getattr(self.model, name) < 1:
                raise ValueError(f"model.{name} must be positive")
        if self.lora.rank < 1 or self.lora.alpha < 1 or not self.lora.target_modules:
            raise ValueError("LoRA rank, alpha, and target_modules must be non-empty and positive")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("lora.dropout must be in [0, 1)")
        if self.objective.current_samples < 1 or self.objective.reference_samples < 1:
            raise ValueError("Objective population counts must be positive")
        if self.objective.name == "rwtd":
            if not self.features.enabled or not self.reward.enabled:
                raise ValueError("rwtd requires both features and reward to be enabled")
            if self.rwtd.coupling not in {"sinkhorn", "random"}:
                raise ValueError("rwtd.coupling must be sinkhorn or random")
            if self.rwtd.sinkhorn_target not in {"barycentric", "sampled"}:
                raise ValueError(
                    "rwtd.sinkhorn_target must be barycentric or sampled"
                )
            if self.rwtd.reward_temperature <= 0:
                raise ValueError("rwtd.reward_temperature must be positive")
            if self.rwtd.reward_scale <= 0:
                raise ValueError("rwtd.reward_scale must be positive")
            if not 0 <= self.rwtd.mass_floor < 1:
                raise ValueError("rwtd.mass_floor must be in [0, 1)")
            if not 0 <= self.rwtd.reference_fraction <= 1:
                raise ValueError("rwtd.reference_fraction must be in [0, 1]")
            if not 0 < self.rwtd.transport_step <= 1:
                raise ValueError("rwtd.transport_step must be in (0, 1]")
            if self.rwtd.coupling == "sinkhorn":
                if self.rwtd.ot_regularization_scale <= 0:
                    raise ValueError("rwtd.ot_regularization_scale must be positive")
                if self.rwtd.minimum_ot_epsilon <= 0:
                    raise ValueError("rwtd.minimum_ot_epsilon must be positive")
                if self.rwtd.sinkhorn_iterations < 1:
                    raise ValueError("rwtd.sinkhorn_iterations must be positive")
                if (
                    self.rwtd.sinkhorn_tolerance is not None
                    and self.rwtd.sinkhorn_tolerance <= 0
                ):
                    raise ValueError("rwtd.sinkhorn_tolerance must be positive")
            if self.rwtd.displacement_clip is not None and self.rwtd.displacement_clip <= 0:
                raise ValueError("rwtd.displacement_clip must be positive")
        if self.features.enabled and not self.features.names:
            raise ValueError("features.names must not be empty")
        if self.features.provider not in {"dinov2", "hpsv2"}:
            raise ValueError("features.provider must be dinov2 or hpsv2")
        if self.features.enabled and self.features.provider == "hpsv2":
            if self.objective.name != "rwtd":
                raise ValueError("hpsv2 features are currently supported only by rwtd")
            if not self.reward.enabled or self.reward.provider != "hpsv2":
                raise ValueError(
                    "hpsv2 features require an enabled hpsv2 reward to reuse"
                )
            if self.features.names != ("hps",):
                raise ValueError("hpsv2 features require features.names=[hps]")
        if self.reward.batch_size < 1:
            raise ValueError("reward.batch_size must be positive")
        if self.reward.dtype not in {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
        }:
            raise ValueError("reward.dtype must be a supported floating-point dtype")
        reward_providers = {
            "pickscore",
            "geneval",
            "imagereward",
            "hpsv2",
            "clip",
            "laion_aesthetic",
            "composite",
        }
        if self.reward.provider not in reward_providers:
            choices = ", ".join(sorted(reward_providers))
            raise ValueError(f"reward.provider must be one of: {choices}")
        if self.reward.provider == "geneval":
            if not self.reward.socket_path:
                raise ValueError("reward.socket_path must not be empty for GenEval")
            if self.reward.timeout <= 0 or self.reward.startup_timeout <= 0:
                raise ValueError("GenEval reward timeouts must be positive")
            if self.reward.reward_mode not in {"binary", "dense", "hybrid"}:
                raise ValueError(
                    "reward.reward_mode must be binary, dense, or hybrid"
                )
            if self.reward.binary_bonus < 0:
                raise ValueError("reward.binary_bonus must be non-negative")
        if self.reward.provider == "composite":
            if not self.reward.components:
                raise ValueError("reward.components must not be empty for composite rewards")
            component_providers = reward_providers - {"composite", "geneval"}
            for index, component in enumerate(self.reward.components):
                if not isinstance(component, dict):
                    raise ValueError(f"reward.components[{index}] must be a mapping")
                provider = component.get("provider")
                if provider not in component_providers:
                    choices = ", ".join(sorted(component_providers))
                    raise ValueError(
                        f"reward.components[{index}].provider must be one of: {choices}"
                    )
                for field_name in ("weight", "mean", "scale"):
                    value = component.get(field_name)
                    if not isinstance(value, (int, float)):
                        raise ValueError(
                            f"reward.components[{index}].{field_name} must be numeric"
                        )
                if component["scale"] <= 0:
                    raise ValueError(
                        f"reward.components[{index}].scale must be positive"
                    )
        if self.evaluation.interval_steps < 1:
            raise ValueError("evaluation.interval_steps must be positive")
        if self.evaluation.provider not in {"pickscore", "geneval"}:
            raise ValueError("evaluation.provider must be pickscore or geneval")
        if (
            self.evaluation.prompt_count is not None
            and self.evaluation.prompt_count < 1
        ):
            raise ValueError("evaluation.prompt_count must be positive when set")
        if self.evaluation.seed < 0:
            raise ValueError("evaluation.seed must be non-negative")
        if (
            self.evaluation.samples_per_prompt < 1
            or self.evaluation.prompt_batch_size < 1
            or self.evaluation.reward_batch_size < 1
        ):
            raise ValueError("Evaluation sample and batch sizes must be positive")
        if self.evaluation.enabled and self.evaluation.provider == "geneval":
            if not self.reward.enabled or self.reward.provider != "geneval":
                raise ValueError(
                    "GenEval periodic evaluation requires an enabled GenEval reward"
                )
            if self.evaluation.reward_mode not in {"binary", "dense", "hybrid"}:
                raise ValueError(
                    "evaluation.reward_mode must be binary, dense, or hybrid"
                )
            if self.evaluation.binary_bonus < 0:
                raise ValueError(
                    "evaluation.binary_bonus must be non-negative"
                )
        if self.optimizer.learning_rate <= 0 or self.optimizer.max_grad_norm <= 0:
            raise ValueError("Optimizer learning rate and max_grad_norm must be positive")
        if self.runtime.train_batch_size < 1 or self.runtime.max_train_steps < 1:
            raise ValueError("Runtime batch size and max_train_steps must be positive")
        if self.runtime.gradient_accumulation_steps < 1:
            raise ValueError("runtime.gradient_accumulation_steps must be positive")
        if self.runtime.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("runtime.mixed_precision must be no, fp16, or bf16")
        if self.logging.logging_steps < 1 or self.logging.checkpointing_steps < 1:
            raise ValueError("Logging and checkpointing intervals must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved nested dataclasses into plain mappings.

        Returns:
            Nested dictionary suitable for YAML, JSON, or tracker metadata.
        """
        return asdict(self)


def _section(cls: type[Any], value: Any) -> Any:
    """Construct one nested dataclass section from YAML-compatible data.

    Args:
        cls: Dataclass type to instantiate.
        value: Mapping loaded from YAML or ``None`` for section defaults.

    Returns:
        Instantiated dataclass section.

    Raises:
        TypeError: If a section is not represented by a mapping.
    """
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} configuration must be a mapping")
    converted = dict(value)
    for key in (
        "target_modules",
        "names",
        "feature_weights",
        "components",
    ):
        if key in converted and isinstance(converted[key], list):
            converted[key] = tuple(converted[key])
    return cls(**converted)


def load_alignment_config(path: str | Path) -> AlignmentConfig:
    """Load, resolve, and validate an online-alignment YAML configuration.

    Args:
        path: YAML file containing any subset of ``AlignmentConfig`` sections.

    Returns:
        Fully populated and validated immutable alignment configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root or a section is not a mapping.
        ValueError: If resolved configuration values violate invariants.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Alignment config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("Alignment YAML root must be a mapping")
    known = {
        "model",
        "lora",
        "reward",
        "evaluation",
        "features",
        "objective",
        "rwtd",
        "optimizer",
        "runtime",
        "logging",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown alignment config sections: {unknown}")
    config = AlignmentConfig(
        model=_section(ModelConfig, raw.get("model")),
        lora=_section(LoRAConfig, raw.get("lora")),
        reward=_section(RewardConfig, raw.get("reward")),
        evaluation=_section(EvaluationConfig, raw.get("evaluation")),
        features=_section(FeatureConfig, raw.get("features")),
        objective=_section(ObjectiveConfig, raw.get("objective")),
        rwtd=_section(RWTDConfig, raw.get("rwtd")),
        optimizer=_section(OptimizerConfig, raw.get("optimizer")),
        runtime=_section(RuntimeConfig, raw.get("runtime")),
        logging=_section(LoggingConfig, raw.get("logging")),
    )
    config.validate()
    return config


def apply_alignment_overrides(
    config: AlignmentConfig,
    overrides: list[str],
) -> AlignmentConfig:
    """Apply validated ``section.field=YAML_VALUE`` command-line overrides.

    Args:
        config: Fully resolved base configuration.
        overrides: Ordered dotted assignments, for example
            ``rwtd.reward_temperature=0.5``.

    Returns:
        New validated immutable alignment configuration.

    Raises:
        ValueError: If syntax, section names, or field names are invalid.
    """
    updated = config
    for assignment in overrides:
        if "=" not in assignment:
            raise ValueError(f"Invalid override {assignment!r}; expected section.field=value")
        path, raw_value = assignment.split("=", 1)
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid override path {path!r}; expected section.field")
        section_name, field_name = parts
        if not hasattr(updated, section_name):
            raise ValueError(f"Unknown config section in override: {section_name}")
        section = getattr(updated, section_name)
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown config field in override: {path}")
        value = yaml.safe_load(raw_value)
        current = getattr(section, field_name)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        updated = replace(
            updated,
            **{section_name: replace(section, **{field_name: value})},
        )
    updated.validate()
    return updated
