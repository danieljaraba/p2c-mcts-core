"""MCTS selection and simulation policies.

Policies are implemented using the **Strategy** pattern so that the
:class:`~src.core.mcts.search.MCTSSearch` algorithm can swap behaviours
without modification, satisfying OCP.

Two pairs of policies are provided:

Selection policies (tree phase):
    :class:`UCB1SelectionPolicy` – standard UCB1 selection (all strategies).

Simulation policies (rollout phase):
    :class:`RandomSimulationPolicy`         – uniform random action selection.
    :class:`DeterministicPreferencePolicy`  – prefers actions with
    ``deterministic=True``.
    :class:`CheapestActionPolicy`           – prefers the cheapest available action.
"""

from __future__ import annotations

import random
from typing import Protocol

from src.core.mcts.node import MCTSNode
from src.core.mdp.entities import CraftingAction, CraftingGoal, ItemState

# ---------------------------------------------------------------------------
# Selection policy protocol and implementation
# ---------------------------------------------------------------------------


class SelectionPolicy(Protocol):
    """Protocol for the MCTS tree-selection phase.

    Implementors choose a child node to descend into during the selection
    phase of the MCTS loop.
    """

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Return the most promising child of *node*.

        Args:
            node: A fully expanded non-terminal node.

        Returns:
            The selected child node.
        """
        ...


class UCB1SelectionPolicy:
    """UCB1-based tree selection policy.

    Selects the child with the highest UCB1 score.  The exploration constant
    *C* controls the exploration–exploitation trade-off.

    Args:
        exploration_constant: UCB1 exploration parameter *C*.
            The commonly used default is ``sqrt(2) ≈ 1.414``.
    """

    def __init__(self, exploration_constant: float = 1.414) -> None:
        self._exploration_constant = exploration_constant

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Return the child with the highest UCB1 score.

        Args:
            node: Parent node with at least one child.

        Returns:
            Best child according to UCB1.

        Raises:
            ValueError: If *node* has no children.
        """
        if not node.children:
            raise ValueError("Cannot select child from a node with no children.")
        return max(
            node.children,
            key=lambda child: child.ucb1_score(self._exploration_constant),
        )


# ---------------------------------------------------------------------------
# Simulation policy protocol and implementations
# ---------------------------------------------------------------------------


class SimulationPolicy(Protocol):
    """Protocol for the MCTS simulation (rollout) phase.

    Implementors choose which action to apply at each step of the random
    rollout until a terminal state or depth limit is reached.
    """

    def select_action(
        self,
        state: ItemState,
        available_actions: list[CraftingAction],
        goal: CraftingGoal,
    ) -> CraftingAction:
        """Return the action to apply in the current rollout step.

        Args:
            state: Current item state during the rollout.
            available_actions: Actions that can be applied in *state*.
            goal: Crafting goal used for goal-aware heuristics.

        Returns:
            The chosen :class:`~src.core.mdp.entities.CraftingAction`.
        """
        ...


class RandomSimulationPolicy:
    """Simulation policy that selects actions uniformly at random.

    Used as the default rollout policy for the BALANCED strategy.

    Args:
        rng: Random number generator.  Defaults to a new instance.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def select_action(
        self,
        state: ItemState,
        available_actions: list[CraftingAction],
        goal: CraftingGoal,
    ) -> CraftingAction:
        """Return a uniformly random action.

        Args:
            state: Current rollout state (unused by this policy).
            available_actions: Non-empty list of applicable actions.
            goal: Crafting goal (unused by this policy).

        Returns:
            Randomly chosen action.
        """
        return self._rng.choice(available_actions)


class DeterministicPreferencePolicy:
    """Simulation policy that prefers deterministic crafting actions.

    If any deterministic actions are available they are preferred over
    stochastic ones.  Ties are broken randomly.  This drives the MCTS toward
    reliable, low-variance crafting paths.

    Args:
        rng: Random number generator.  Defaults to a new instance.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def select_action(
        self,
        state: ItemState,
        available_actions: list[CraftingAction],
        goal: CraftingGoal,
    ) -> CraftingAction:
        """Return a deterministic action when available, otherwise random.

        Args:
            state: Current rollout state (unused by this policy).
            available_actions: Non-empty list of applicable actions.
            goal: Crafting goal (unused by this policy).

        Returns:
            Preferred action.
        """
        deterministic = [a for a in available_actions if a.is_deterministic]
        candidates = deterministic if deterministic else available_actions
        return self._rng.choice(candidates)


class CheapestActionPolicy:
    """Simulation policy that selects the cheapest available action.

    When multiple actions share the minimum cost one is chosen at random.
    This drives the rollout toward low-cost crafting sequences.

    Args:
        rng: Random number generator.  Defaults to a new instance.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def select_action(
        self,
        state: ItemState,
        available_actions: list[CraftingAction],
        goal: CraftingGoal,
    ) -> CraftingAction:
        """Return the cheapest action, with random tie-breaking.

        Args:
            state: Current rollout state (unused by this policy).
            available_actions: Non-empty list of applicable actions.
            goal: Crafting goal (unused by this policy).

        Returns:
            Cheapest action.
        """
        min_cost = min(a.cost for a in available_actions)
        cheapest = [a for a in available_actions if a.cost == min_cost]
        return self._rng.choice(cheapest)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_simulation_policy(
    strategy: str, rng: random.Random | None = None
) -> SimulationPolicy:
    """Return the :class:`SimulationPolicy` for the given *strategy*.

    This is a **Factory function** that maps strategy names to concrete policy
    instances, keeping calling code decoupled from implementations.

    Args:
        strategy: One of ``"deterministic"``, ``"cheapest"``, ``"balanced"``.
        rng: Optional random number generator shared across policies.

    Returns:
        Concrete :class:`SimulationPolicy`.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    _rng = rng if rng is not None else random.Random()
    mapping: dict[str, SimulationPolicy] = {
        "deterministic": DeterministicPreferencePolicy(rng=_rng),
        "cheapest": CheapestActionPolicy(rng=_rng),
        "balanced": RandomSimulationPolicy(rng=_rng),
    }
    if strategy not in mapping:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {list(mapping.keys())}."
        )
    return mapping[strategy]


def build_selection_policy(strategy: str) -> SelectionPolicy:
    """Return the :class:`SelectionPolicy` for the given *strategy*.

    The exploration constant is tuned per strategy:
    - ``"deterministic"`` uses *C = 1.0* (more exploitation).
    - ``"cheapest"`` and ``"balanced"`` use *C = 1.414* (standard).

    Args:
        strategy: One of ``"deterministic"``, ``"cheapest"``, ``"balanced"``.

    Returns:
        Concrete :class:`UCB1SelectionPolicy`.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    exploration_constants: dict[str, float] = {
        "deterministic": 1.0,
        "cheapest": 1.414,
        "balanced": 1.414,
    }
    if strategy not in exploration_constants:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of "
            f"{list(exploration_constants.keys())}."
        )
    return UCB1SelectionPolicy(
        exploration_constant=exploration_constants[strategy]
    )
