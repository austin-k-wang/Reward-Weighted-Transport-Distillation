"""Built-in online-alignment objective plug-ins."""

from .base import build_objective, register_objective
from .rwtd import RWTDObjective

__all__ = [
    "RWTDObjective",
    "build_objective",
    "register_objective",
]
