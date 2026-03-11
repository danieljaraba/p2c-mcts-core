"""Unit tests for the reward functions."""

from __future__ import annotations

import pytest

from src.core.mdp.entities import (
    BaseItem,
    CraftingGoal,
    GoalType,
    ItemModifier,
    ItemState,
    ModifierTarget,
)
from src.core.mdp.reward import (
    BalancedReward,
    CheapestReward,
    DeterministicReward,
    _compute_goal_progress,
    build_reward_function,
)


@pytest.fixture
def base_item() -> BaseItem:
    return BaseItem(item_type="helmet", item_level=86)


@pytest.fixture
def goal(base_item: BaseItem) -> CraftingGoal:
    return CraftingGoal(
        goal_type=GoalType.PARTIAL,
        target_modifiers=(
            ModifierTarget(modifier_id="life_regen", min_tier=1, weight=1.0),
            ModifierTarget(modifier_id="crit_chance", min_tier=2, weight=1.0),
        ),
        max_steps=30,
        budget_limit=100.0,
    )


@pytest.fixture
def full_goal_state(base_item: BaseItem) -> ItemState:
    """Item state that satisfies both target modifiers."""
    return ItemState(
        base_item=base_item,
        modifiers=frozenset(
            {
                ItemModifier("life_regen", tier=1, value=50.0),
                ItemModifier("crit_chance", tier=1, value=20.0),
            }
        ),
    )


@pytest.fixture
def half_goal_state(base_item: BaseItem) -> ItemState:
    """Item state that satisfies only one of two target modifiers."""
    return ItemState(
        base_item=base_item,
        modifiers=frozenset({ItemModifier("life_regen", tier=1, value=50.0)}),
    )


@pytest.fixture
def empty_state(base_item: BaseItem) -> ItemState:
    return ItemState(base_item=base_item, modifiers=frozenset())


class TestComputeGoalProgress:
    """Tests for the shared _compute_goal_progress helper."""

    def test_all_modifiers_met_returns_one(
        self,
        full_goal_state: ItemState,
        goal: CraftingGoal,
    ) -> None:
        assert _compute_goal_progress(full_goal_state, goal) == pytest.approx(1.0)

    def test_no_modifiers_met_returns_zero(
        self,
        empty_state: ItemState,
        goal: CraftingGoal,
    ) -> None:
        assert _compute_goal_progress(empty_state, goal) == pytest.approx(0.0)

    def test_half_modifiers_met_returns_half(
        self,
        half_goal_state: ItemState,
        goal: CraftingGoal,
    ) -> None:
        assert _compute_goal_progress(half_goal_state, goal) == pytest.approx(0.5)

    def test_empty_target_modifiers_returns_zero(
        self, empty_state: ItemState, base_item: BaseItem
    ) -> None:
        empty_goal = CraftingGoal(
            goal_type=GoalType.PARTIAL,
            target_modifiers=(),
        )
        assert _compute_goal_progress(empty_state, empty_goal) == pytest.approx(0.0)

    def test_tier_above_min_tier_not_counted(
        self, goal: CraftingGoal, base_item: BaseItem
    ) -> None:
        # tier=3 is worse than min_tier=1 (lower is better), so not counted.
        state = ItemState(
            base_item=base_item,
            modifiers=frozenset(
                {ItemModifier("life_regen", tier=3, value=5.0)}
            ),
        )
        assert _compute_goal_progress(state, goal) == pytest.approx(0.0)


class TestDeterministicReward:
    """Tests for DeterministicReward."""

    def test_full_goal_returns_one(
        self, full_goal_state: ItemState, goal: CraftingGoal
    ) -> None:
        reward = DeterministicReward()
        assert reward.evaluate(full_goal_state, goal) == pytest.approx(1.0)

    def test_no_progress_returns_zero(
        self, empty_state: ItemState, goal: CraftingGoal
    ) -> None:
        reward = DeterministicReward()
        assert reward.evaluate(empty_state, goal) == pytest.approx(0.0)

    def test_reward_ignores_accumulated_cost(
        self, goal: CraftingGoal, base_item: BaseItem
    ) -> None:
        expensive_state = ItemState(
            base_item=base_item,
            modifiers=frozenset(
                {
                    ItemModifier("life_regen", tier=1),
                    ItemModifier("crit_chance", tier=1),
                }
            ),
            accumulated_cost=999.0,
        )
        reward = DeterministicReward()
        assert reward.evaluate(expensive_state, goal) == pytest.approx(1.0)


class TestCheapestReward:
    """Tests for CheapestReward."""

    def test_full_goal_with_no_cost_returns_one(
        self, full_goal_state: ItemState, goal: CraftingGoal
    ) -> None:
        reward = CheapestReward()
        assert reward.evaluate(full_goal_state, goal) == pytest.approx(1.0)

    def test_high_cost_reduces_reward(
        self, full_goal_state: ItemState, goal: CraftingGoal, base_item: BaseItem
    ) -> None:
        expensive_state = ItemState(
            base_item=base_item,
            modifiers=full_goal_state.modifiers,
            accumulated_cost=goal.budget_limit or 100.0,  # 100% of budget
        )
        reward = CheapestReward(cost_weight=0.4)
        # progress=1.0, cost_fraction=1.0 → 1.0 - 0.4 * 1.0 = 0.6
        assert reward.evaluate(expensive_state, goal) == pytest.approx(0.6)

    def test_reward_never_below_zero(
        self, goal: CraftingGoal, base_item: BaseItem
    ) -> None:
        very_expensive = ItemState(
            base_item=base_item,
            modifiers=frozenset(),
            accumulated_cost=10000.0,
        )
        reward = CheapestReward()
        assert reward.evaluate(very_expensive, goal) >= 0.0


class TestBalancedReward:
    """Tests for BalancedReward."""

    def test_full_goal_no_cost_returns_high_reward(
        self, full_goal_state: ItemState, goal: CraftingGoal
    ) -> None:
        reward = BalancedReward()
        # progress=1.0, cost_savings=1.0, step_efficiency=1.0
        # 0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 1.0 = 1.0
        assert reward.evaluate(full_goal_state, goal) == pytest.approx(1.0)

    def test_empty_state_returns_positive_due_to_savings(
        self, empty_state: ItemState, goal: CraftingGoal
    ) -> None:
        reward = BalancedReward()
        r = reward.evaluate(empty_state, goal)
        # progress=0.0, cost_savings=1.0 (no cost), step_efficiency=1.0
        # 0.6*0 + 0.2*1 + 0.2*1 = 0.4
        assert r == pytest.approx(0.4)

    def test_reward_never_below_zero(
        self, goal: CraftingGoal, base_item: BaseItem
    ) -> None:
        state = ItemState(
            base_item=base_item,
            modifiers=frozenset(),
            accumulated_cost=10000.0,
            step_count=100,
        )
        assert BalancedReward().evaluate(state, goal) >= 0.0


class TestBuildRewardFunction:
    """Tests for the reward function factory."""

    def test_deterministic_strategy(self) -> None:
        rf = build_reward_function("deterministic")
        assert isinstance(rf, DeterministicReward)

    def test_cheapest_strategy(self) -> None:
        rf = build_reward_function("cheapest")
        assert isinstance(rf, CheapestReward)

    def test_balanced_strategy(self) -> None:
        rf = build_reward_function("balanced")
        assert isinstance(rf, BalancedReward)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_reward_function("god_mode")
