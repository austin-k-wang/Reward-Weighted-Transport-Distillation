"""Lazy registry and construction helpers for unified rewards."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


RewardLoader = Callable[..., object] | str

_REGISTRY: dict[str, RewardLoader] = {
    "pickscore": "src.rewards.pickscore:PickScoreReward",
    "geneval": "src.rewards.geneval:GenEvalReward",
    "imagereward": "src.rewards.imagereward:ImageRewardEvaluator",
    "hpsv2": "src.rewards.hps:HPSv21Reward",
    "hpsv2.1": "src.rewards.hps:HPSv21Reward",
    "hps_v2.1": "src.rewards.hps:HPSv21Reward",
    "clip": "src.rewards.clip:CLIPReward",
    "laion_aesthetic": "src.rewards.aesthetics:LAIONAestheticReward",
    "composite": "src.rewards.composite:StandardizedWeightedReward",
}


def register_reward(name: str, loader: RewardLoader, *, replace: bool = False) -> None:
    """Register a reward constructor or lazy import target.

    Args:
        name: Case-insensitive non-empty registry key.
        loader: Constructor callable or ``"module:attribute"`` import target.
        replace: Whether an existing registration may be overwritten.

    Returns:
        Nothing. The process-local reward registry is updated.

    Raises:
        ValueError: If the name or import target is invalid, or the key exists
            while ``replace`` is false.
        TypeError: If ``loader`` is neither callable nor a string.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("reward name must not be empty")
    if not callable(loader) and not isinstance(loader, str):
        raise TypeError("loader must be callable or a 'module:attribute' string")
    if isinstance(loader, str) and loader.count(":") != 1:
        raise ValueError("lazy loader must use 'module:attribute' syntax")
    if key in _REGISTRY and not replace:
        raise ValueError(f"reward {key!r} is already registered")
    _REGISTRY[key] = loader


def _resolve_loader(loader: RewardLoader) -> Callable[..., object]:
    """Resolve one callable or lazy module target to a constructor.

    Args:
        loader: Direct callable or ``"module:attribute"`` string.

    Returns:
        A callable reward constructor.

    Raises:
        TypeError: If the resolved attribute is not callable.
    """
    if callable(loader):
        return loader
    module_name, attribute = loader.split(":", maxsplit=1)
    resolved = getattr(importlib.import_module(module_name), attribute)
    if not callable(resolved):
        raise TypeError(f"registered target {loader!r} is not callable")
    return resolved


def create_reward(name: str, **kwargs: Any) -> object:
    """Construct a registered reward without importing unrelated backends.

    Args:
        name: Case-insensitive registered reward name.
        **kwargs: Keyword arguments forwarded unchanged to its constructor.

    Returns:
        A newly constructed reward evaluator.

    Raises:
        KeyError: If no reward is registered under ``name``.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        choices = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown reward {name!r}; available rewards: {choices}")
    return _resolve_loader(_REGISTRY[key])(**kwargs)


def available_rewards() -> tuple[str, ...]:
    """List currently registered reward names in deterministic order.

    Returns:
        Sorted tuple of lowercase registry keys.
    """
    return tuple(sorted(_REGISTRY))


def create_reward_from_config(
    config: object,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    socket_path: str | None = None,
) -> object:
    """Construct one canonical evaluator from reward configuration data.

    Args:
        config: Dataclass or mapping containing at least a ``provider`` key and
            optional model, checkpoint, socket, and composite component fields.
        device: Device used by in-process neural reward models.
        dtype: Floating dtype used by in-process neural reward models.
        socket_path: Optional resolved rank-local GenEval socket overriding the
            configured socket template.

    Returns:
        A registered reward evaluator configured for the selected provider.
        Composite configurations recursively construct every component and
        standardize each with its fixed ``mean`` and ``scale``.

    Raises:
        TypeError: If ``config`` is neither a dataclass nor a mapping.
        ValueError: If a provider lacks a required local checkpoint.
    """
    if is_dataclass(config) and not isinstance(config, type):
        values = asdict(config)
    elif isinstance(config, dict):
        values = dict(config)
    else:
        raise TypeError("reward config must be a dataclass or mapping")

    provider = str(values.get("provider", "")).strip().lower()
    model_path = values.get("model_path")
    processor_path = values.get("processor_path")
    checkpoint_path = values.get("checkpoint_path")
    base_model_path = values.get("base_model_path")
    local_files_only = bool(values.get("local_files_only", True))

    if provider == "pickscore":
        return create_reward(
            provider,
            model_name_or_path=model_path or "yuvalkirstain/PickScore_v1",
            processor_name_or_path=processor_path
            or "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    if provider == "clip":
        return create_reward(
            provider,
            model_name_or_path=model_path or "openai/clip-vit-large-patch14",
            processor_name_or_path=processor_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    if provider == "imagereward":
        return create_reward(
            provider,
            model_name_or_path=model_path or "ImageReward-v1.0",
            checkpoint_root=checkpoint_path,
            tokenizer_name_or_path=processor_path or "bert-base-uncased",
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    if provider == "hpsv2":
        if checkpoint_path is None:
            raise ValueError("HPS v2.1 requires reward.checkpoint_path")
        return create_reward(
            provider,
            hps_checkpoint_path=checkpoint_path,
            model_checkpoint_path=base_model_path,
            device=device,
            dtype=dtype,
        )
    if provider == "laion_aesthetic":
        if checkpoint_path is None:
            raise ValueError("LAION aesthetics requires reward.checkpoint_path")
        return create_reward(
            provider,
            aesthetic_checkpoint_path=checkpoint_path,
            clip_model_name_or_path=model_path or "openai/clip-vit-large-patch14",
            processor_name_or_path=processor_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    if provider == "geneval":
        return create_reward(
            provider,
            socket_path=socket_path or values.get("socket_path"),
            timeout=float(values.get("timeout", 300.0)),
            startup_timeout=float(values.get("startup_timeout", 900.0)),
            reward_mode=str(values.get("reward_mode", "hybrid")),
            binary_bonus=float(values.get("binary_bonus", 0.25)),
            warmup=bool(values.get("warmup", False)),
        )
    if provider == "composite":
        evaluators = {}
        weights = {}
        centers = {}
        scales = {}
        for index, component in enumerate(values.get("components", ())):
            component_values = dict(component)
            component_provider = str(component_values["provider"])
            name = str(component_values.get("name") or f"{component_provider}_{index}")
            evaluators[name] = create_reward_from_config(
                component_values,
                device=device,
                dtype=dtype,
            )
            weights[name] = float(component_values["weight"])
            centers[name] = float(component_values["mean"])
            scales[name] = float(component_values["scale"])
        return create_reward(
            provider,
            evaluators=evaluators,
            weights=weights,
            centers=centers,
            scales=scales,
        )
    return create_reward(provider)
