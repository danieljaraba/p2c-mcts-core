"""Reward functions for the Path of Exile 2 crafting MDP.

Each reward function implements the :class:`RewardFunction` protocol and
represents a different optimization objective.  The Strategy pattern allows
the MCTS search engine to swap objectives without changing its core logic.

Three concrete strategies are provided:

* :class:`DeterministicReward`  – maximises goal progress, ignores cost.
* :class:`CheapestReward`       – penalises currency cost heavily.
* :class:`BalancedReward`       – balances progress, cost, and reliability.
"""

from __future__ import annotations

from typing import Protocol

from src.core.mdp.entities import CraftingGoal, GoalType, ItemState


class RewardFunction(Protocol):
    """Strategy interface for evaluating an item state against a crafting goal.

    Concrete implementations express different optimisation objectives
    (determinism, cost-minimisation, balanced).
    """

    def evaluate(self, state: ItemState, goal: CraftingGoal) -> float:
        """Return a scalar reward in [0, 1] for the given *state* w.r.t. *goal*.

        Args:
            state: Current item state to evaluate.
            goal: The crafting goal that defines what success looks like.

        Returns:
            Reward value between 0.0 (no progress) and 1.0 (goal fully met).
        """
        ...


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _compute_goal_progress(state: ItemState, goal: CraftingGoal) -> float:
    """Compute normalised progress toward the crafting goal.

    For PARTIAL and EXACT goals the score is the weighted fraction of target
    modifiers that are present with an acceptable tier.  For SCORE_BASED goals
    the same weighted-fraction formula is used.

    Args:
        state: Current item state.
        goal: The crafting goal.

    Returns:
        Float in [0, 1] representing how close the item is to the goal.
    """
    if not goal.target_modifiers:
        return 0.0

    total_weight = sum(m.weight for m in goal.target_modifiers)
    if total_weight == 0.0:
        return 0.0

    achieved_weight = 0.0
    for target in goal.target_modifiers:
        item_mod = state.get_modifier(target.modifier_id)
        if item_mod is not None and item_mod.tier <= target.min_tier:
            achieved_weight += target.weight

    progress = achieved_weight / total_weight

    if goal.goal_type == GoalType.EXACT:
        # For EXACT goals the item must have *only* the target modifiers.
        extra_mods = state.modifier_count - len(
            [
                t
                for t in goal.target_modifiers
                if state.has_modifier(t.modifier_id)
            ]
        )
        penalty = min(extra_mods * 0.1, 0.3)
        progress = max(0.0, progress - penalty)

    return min(progress, 1.0)


# ---------------------------------------------------------------------------
# Concrete reward functions  (Strategy pattern)
# ---------------------------------------------------------------------------


class DeterministicReward:
    """Reward function optimised for the most deterministic crafting path.

    This strategy ignores currency cost and focuses entirely on goal
    achievement.  Combined with a simulation policy that prefers deterministic
    actions it drives the MCTS toward low-variance crafting sequences.
    """

    def evaluate(self, state: ItemState, goal: CraftingGoal) -> float:
        """Return goal-progress reward with no cost penalty.

        Args:
            state: Current item state.
            goal: Crafting goal.

        Returns:
            Float in [0, 1].
        """
        return _compute_goal_progress(state, goal)


class CheapestReward:
    """Reward function optimised for the cheapest crafting path.

    Applies a cost penalty proportional to the fraction of the budget consumed.
    When no budget_limit is set, a default reference of 500 chaos equivalents
    is used to normalise the cost.

    Args:
        cost_weight: Coefficient controlling the strength of the cost penalty.
            Default is 0.4 (40 % of the reward can be erased by cost).
        default_budget: Reference budget used when no explicit limit is given.
    """

    def __init__(
        self,
        cost_weight: float = 0.4,
        default_budget: float = 500.0,
    ) -> None:
        self._cost_weight = cost_weight
        self._default_budget = default_budget

    def evaluate(self, state: ItemState, goal: CraftingGoal) -> float:
        """Return goal-progress reward heavily penalised by accumulated cost.

        Args:
            state: Current item state.
            goal: Crafting goal.

        Returns:
            Float in [0, 1].
        """
        progress = _compute_goal_progress(state, goal)
        budget = (
            goal.budget_limit if goal.budget_limit is not None else self._default_budget
        )
        cost_fraction = min(state.accumulated_cost / budget, 1.0)
        return max(0.0, progress - self._cost_weight * cost_fraction)


class BalancedReward:
    """Reward function that balances goal progress and cost efficiency.

    Blends goal progress (60 %), cost savings (20 %), and a small bonus
    for the step count remaining within limits (20 %).

    Args:
        default_budget: Reference budget used when no explicit limit is given.
    """

    def __init__(self, default_budget: float = 500.0) -> None:
        self._default_budget = default_budget

    def evaluate(self, state: ItemState, goal: CraftingGoal) -> float:
        """Return blended reward combining progress, economy, and efficiency.

        Args:
            state: Current item state.
            goal: Crafting goal.

        Returns:
            Float in [0, 1].
        """
        progress = _compute_goal_progress(state, goal)
        budget = (
            goal.budget_limit if goal.budget_limit is not None else self._default_budget
        )
        cost_fraction = min(state.accumulated_cost / budget, 1.0)
        step_fraction = min(state.step_count / goal.max_steps, 1.0)

        cost_savings = 1.0 - cost_fraction
        step_efficiency = 1.0 - step_fraction

        return max(
            0.0,
            0.6 * progress + 0.2 * cost_savings + 0.2 * step_efficiency,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_reward_function(strategy: str) -> RewardFunction:
    """Return the :class:`RewardFunction` implementation for *strategy*.

    This is a **Factory function** that decouples callers from concrete
    reward implementations.

    Args:
        strategy: One of ``"deterministic"``, ``"cheapest"``, ``"balanced"``.

    Returns:
        Matching concrete :class:`RewardFunction`.

    Raises:
        ValueError: If *strategy* is not a recognised value.
    """
    mapping: dict[str, RewardFunction] = {
        "deterministic": DeterministicReward(),
        "cheapest": CheapestReward(),
        "balanced": BalancedReward(),
    }
    if strategy not in mapping:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {list(mapping.keys())}."
        )
    return mapping[strategy]
