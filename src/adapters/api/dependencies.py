"""FastAPI dependency providers for the crafting optimiser service.

Dependencies are wired here (composition root) so that route handlers
depend only on port abstractions, never on concrete implementations.
"""

from __future__ import annotations

from functools import lru_cache

from src.core.optimizer import CraftingOptimizer
from src.ports.optimizer_port import OptimizerPort


@lru_cache(maxsize=1)
def _build_optimizer() -> CraftingOptimizer:
    """Construct and cache the :class:`CraftingOptimizer` singleton.

    Using ``lru_cache`` ensures a single instance is shared across requests,
    which avoids repeated construction overhead while keeping the composition
    root explicit.

    Returns:
        Configured :class:`CraftingOptimizer`.
    """
    return CraftingOptimizer(iterations=500, max_simulation_depth=20)


def get_optimizer() -> OptimizerPort:
    """FastAPI dependency that provides the optimiser port implementation.

    Returns:
        Concrete :class:`~src.core.optimizer.CraftingOptimizer` cast to the
        :class:`~src.ports.optimizer_port.OptimizerPort` protocol.
    """
    return _build_optimizer()
