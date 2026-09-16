"""Registry for algorithm plug-ins used by the generic trainer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..interfaces import OnlineObjective

if TYPE_CHECKING:
    from ..config import AlignmentConfig


ObjectiveFactory = Callable[["AlignmentConfig"], OnlineObjective]
_OBJECTIVES: dict[str, ObjectiveFactory] = {}


def register_objective(name: str) -> Callable[[ObjectiveFactory], ObjectiveFactory]:
    """Register a named objective factory.

    Args:
        name: Unique configuration name used to select the objective.

    Returns:
        Decorator that records and returns the supplied factory unchanged.

    Raises:
        ValueError: If ``name`` is empty or already registered.
    """
    if not name:
        raise ValueError("Objective name must not be empty")

    def decorator(factory: ObjectiveFactory) -> ObjectiveFactory:
        """Store one objective factory under the enclosing registry name."""
        if name in _OBJECTIVES:
            raise ValueError(f"Objective {name!r} is already registered")
        _OBJECTIVES[name] = factory
        return factory

    return decorator


def build_objective(config: "AlignmentConfig") -> OnlineObjective:
    """Construct the objective selected by a resolved alignment config.

    Args:
        config: Complete alignment configuration containing ``objective.name``.

    Returns:
        Objective plug-in compatible with the generic trainer.

    Raises:
        ValueError: If the configured objective has not been registered.
    """
    name = config.objective.name
    try:
        factory = _OBJECTIVES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown objective {name!r}; available objectives: {sorted(_OBJECTIVES)}"
        ) from exc
    return factory(config)
