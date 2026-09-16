"""Unified, lazy-loaded image reward evaluators."""

from __future__ import annotations

import importlib
from typing import Any

from .factory import (
    available_rewards,
    create_reward,
    create_reward_from_config,
    register_reward,
)


_LAZY_EXPORTS = {
    "AestheticMLP": ("src.rewards.aesthetics", "AestheticMLP"),
    "CLIPReward": ("src.rewards.clip", "CLIPReward"),
    "GenEvalReward": ("src.rewards.geneval", "GenEvalReward"),
    "HPSv21FeatureEncoder": ("src.rewards.hps", "HPSv21FeatureEncoder"),
    "HPSv21Reward": ("src.rewards.hps", "HPSv21Reward"),
    "ImageRewardEvaluator": ("src.rewards.imagereward", "ImageRewardEvaluator"),
    "LAIONAestheticReward": ("src.rewards.aesthetics", "LAIONAestheticReward"),
    "PickScoreReward": ("src.rewards.pickscore", "PickScoreReward"),
    "StandardizedWeightedReward": (
        "src.rewards.composite",
        "StandardizedWeightedReward",
    ),
}

__all__ = [
    *_LAZY_EXPORTS,
    "available_rewards",
    "create_reward",
    "create_reward_from_config",
    "register_reward",
]


def __getattr__(name: str) -> Any:
    """Resolve public evaluator classes only when first accessed.

    Args:
        name: Requested package attribute.

    Returns:
        Lazily imported public class cached in this module.

    Raises:
        AttributeError: If ``name`` is not a public lazy export.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
