"""Crafting optimiser use case – the domain-level orchestration entry point.

The :class:`CraftingOptimizer` wires the MDP and MCTS components together and
exposes a single ``optimize`` method.  It depends only on domain entities and
port abstractions, satisfying the Dependency Inversion Principle.

Three optimisation strategies are supported:

* ``"deterministic"`` – Most reliable path (lowest variance).
* ``"cheapest"``      – Lowest expected currency cost.
* ``"balanced"``      – Trade-off between reliability and cost.
"""

from __future__ import annotations

import random

from src.core.mcts.policies import build_selection_policy, build_simulation_policy
from src.core.mcts.search import MCTSSearch, SearchResult
from src.core.mdp.entities import (
    CraftingAction,
    CraftingGoal,
    CraftingStrategy,
    ItemState,
)
from src.core.mdp.reward import build_reward_function
from src.core.mdp.transition import TransitionModel


class CraftingOptimizer:
    """Orchestrates the MCTS search to find the optimal crafting sequence.

    This is the primary **use case** of the domain core.  Infrastructure
    adapters (HTTP, persistence, caching) never appear here; all dependencies
    are injected through port abstractions.

    Args:
        transition_model: Domain transition function T(s, a).  Shared across
            all strategy runs for consistency.
        iterations: Number of MCTS iterations per strategy run.
        max_simulation_depth: Maximum rollout depth during simulation.
        seed: Optional seed for reproducible results.
    """

    def __init__(
        self,
        transition_model: TransitionModel | None = None,
        iterations: int = 500,
        max_simulation_depth: int = 20,
        seed: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._transition = transition_model or TransitionModel(
            rng=random.Random(self._rng.randint(0, 2**32))
        )
        self._iterations = iterations
        self._max_simulation_depth = max_simulation_depth

    # ------------------------------------------------------------------
    # Public use-case interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
        strategy: CraftingStrategy = CraftingStrategy.BALANCED,
        iterations: int | None = None,
    ) -> SearchResult:
        """Run MCTS with the given *strategy* and return the best path found.

        Args:
            initial_state: Item state at the start of crafting.
            goal: Target item configuration and constraints.
            available_actions: All crafting actions available to the player.
            strategy: Optimisation objective that shapes reward and simulation.
            iterations: Optional override for the number of MCTS iterations.
                When ``None`` (default) the instance-level default is used.

        Returns:
            :class:`~src.core.mcts.search.SearchResult` with the best path
            and run statistics.
        """
        strategy_name = strategy.value
        run_iterations = iterations if iterations is not None else self._iterations
        reward_fn = build_reward_function(strategy_name)
        selection_policy = build_selection_policy(strategy_name)
        simulation_policy = build_simulation_policy(
            strategy_name,
            rng=random.Random(self._rng.randint(0, 2**32)),
        )
        search = MCTSSearch(
            transition_model=self._transition,
            reward_function=reward_fn,
            selection_policy=selection_policy,
            simulation_policy=simulation_policy,
            iterations=run_iterations,
            max_simulation_depth=self._max_simulation_depth,
            rng=random.Random(self._rng.randint(0, 2**32)),
        )
        return search.search(
            initial_state=initial_state,
            goal=goal,
            available_actions=available_actions,
            strategy=strategy_name,
        )

    def optimize_all_strategies(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
    ) -> dict[str, SearchResult]:
        """Run MCTS for all three strategies and return results keyed by name.

        This convenience method runs ``deterministic``, ``cheapest``, and
        ``balanced`` strategies in sequence and collects the results into a
        single dictionary.

        Args:
            initial_state: Item state at the start of crafting.
            goal: Target item configuration and constraints.
            available_actions: All crafting actions available to the player.

        Returns:
            Mapping from strategy name to :class:`~src.core.mcts.search.SearchResult`.
        """
        results: dict[str, SearchResult] = {}
        for strategy in CraftingStrategy:
            results[strategy.value] = self.optimize(
                initial_state=initial_state,
                goal=goal,
                available_actions=available_actions,
                strategy=strategy,
            )
        return results
