"""Port interface (abstraction) for the crafting optimiser use case.

Defines the :class:`OptimizerPort` protocol that all optimiser implementations
must satisfy.  Inbound adapters (HTTP, CLI) depend only on this protocol,
never on the concrete :class:`~src.core.optimizer.CraftingOptimizer`, ensuring
Dependency Inversion.
"""

from __future__ import annotations

from typing import Protocol

from src.core.mcts.search import SearchResult
from src.core.mdp.entities import (
    CraftingAction,
    CraftingGoal,
    CraftingStrategy,
    ItemState,
)


class OptimizerPort(Protocol):
    """Port interface for the MCTS-based crafting optimiser.

    Inbound adapters inject a concrete implementation at composition time.
    Domain use-case tests stub this protocol to remain infrastructure-free.
    """

    def optimize(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
        strategy: CraftingStrategy,
        iterations: int | None = None,
    ) -> SearchResult:
        """Optimise the crafting sequence for a single strategy.

        Args:
            initial_state: Item state at the start of crafting.
            goal: Target item configuration and constraints.
            available_actions: All crafting actions available to the player.
            strategy: Optimisation objective.
            iterations: Optional override for the number of MCTS iterations.

        Returns:
            :class:`~src.core.mcts.search.SearchResult` with the best path
            and run statistics.
        """
        ...

    def optimize_all_strategies(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
    ) -> dict[str, SearchResult]:
        """Optimise for all three strategies and return results keyed by name.

        Args:
            initial_state: Item state at the start of crafting.
            goal: Target item configuration and constraints.
            available_actions: All crafting actions available to the player.

        Returns:
            Mapping from strategy name (str) to
            :class:`~src.core.mcts.search.SearchResult`.
        """
        ...
