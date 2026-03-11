"""Unit tests for MDP domain entities."""

from __future__ import annotations

import pytest

from src.core.mdp.entities import (
    ActionEffects,
    ActionType,
    BaseItem,
    CraftingAction,
    CraftingGoal,
    CraftingStrategy,
    GoalType,
    ItemModifier,
    ItemState,
    ModifierTarget,
    Outcome,
)


class TestItemModifier:
    """Tests for ItemModifier validation and equality."""

    def test_valid_modifier(self) -> None:
        mod = ItemModifier(modifier_id="life_regen", tier=1, value=50.0)
        assert mod.modifier_id == "life_regen"
        assert mod.tier == 1
        assert mod.value == 50.0

    def test_tier_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="tier must be >= 1"):
            ItemModifier(modifier_id="x", tier=0)

    def test_modifiers_are_hashable(self) -> None:
        mod = ItemModifier(modifier_id="life_regen", tier=1, value=50.0)
        assert hash(mod) is not None
        assert {mod}  # Can be placed in a set.


class TestBaseItem:
    """Tests for BaseItem validation."""

    def test_valid_base_item(self) -> None:
        item = BaseItem(item_type="helmet", item_level=86)
        assert item.item_type == "helmet"
        assert item.item_level == 86
        assert item.influence is None

    def test_item_level_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="Item level must be between 1 and 100"):
            BaseItem(item_type="ring", item_level=0)

    def test_item_level_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="Item level must be between 1 and 100"):
            BaseItem(item_type="ring", item_level=101)


class TestItemState:
    """Tests for ItemState properties and helpers."""

    def test_modifier_count(self) -> None:
        mods = frozenset(
            {
                ItemModifier("life_regen", 1),
                ItemModifier("crit_chance", 2),
            }
        )
        state = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=mods,
        )
        assert state.modifier_count == 2

    def test_has_modifier_true(self) -> None:
        mod = ItemModifier("life_regen", 1)
        state = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset({mod}),
        )
        assert state.has_modifier("life_regen") is True

    def test_has_modifier_false(self) -> None:
        state = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset(),
        )
        assert state.has_modifier("life_regen") is False

    def test_get_modifier_found(self) -> None:
        mod = ItemModifier("life_regen", 1, 50.0)
        state = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset({mod}),
        )
        found = state.get_modifier("life_regen")
        assert found is not None
        assert found.modifier_id == "life_regen"

    def test_get_modifier_not_found(self) -> None:
        state = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset(),
        )
        assert state.get_modifier("life_regen") is None

    def test_state_is_immutable(self) -> None:
        state = ItemState(base_item=BaseItem("helmet", 86), modifiers=frozenset())
        with pytest.raises(AttributeError):
            state.step_count = 5  # type: ignore[misc]


class TestOutcome:
    """Tests for Outcome validation."""

    def test_probability_zero_is_valid(self) -> None:
        o = Outcome(probability=0.0, transformation_type="reroll_all")
        assert o.probability == 0.0

    def test_probability_one_is_valid(self) -> None:
        o = Outcome(probability=1.0, transformation_type="add_random_mod")
        assert o.probability == 1.0

    def test_probability_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability must be in"):
            Outcome(probability=-0.1, transformation_type="reroll_all")

    def test_probability_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability must be in"):
            Outcome(probability=1.1, transformation_type="reroll_all")


class TestActionEffects:
    """Tests for ActionEffects probability sum validation."""

    def test_valid_effects(self) -> None:
        effects = ActionEffects(
            deterministic=True,
            outcomes=(Outcome(probability=1.0, transformation_type="reroll_all"),),
        )
        assert effects.deterministic is True

    def test_probabilities_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="probabilities must sum to 1.0"):
            ActionEffects(
                deterministic=False,
                outcomes=(
                    Outcome(probability=0.5, transformation_type="reroll_all"),
                    Outcome(probability=0.3, transformation_type="reroll_all"),
                ),
            )

    def test_multiple_outcomes_summing_to_one(self) -> None:
        effects = ActionEffects(
            deterministic=False,
            outcomes=(
                Outcome(probability=0.7, transformation_type="reroll_all"),
                Outcome(probability=0.3, transformation_type="add_random_mod"),
            ),
        )
        assert len(effects.outcomes) == 2


class TestCraftingAction:
    """Tests for CraftingAction validation."""

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost must be >= 0"):
            CraftingAction(
                action_id="test",
                action_type=ActionType.CURRENCY,
                name="Test",
                cost=-1.0,
                effects=ActionEffects(
                    deterministic=True,
                    outcomes=(
                        Outcome(probability=1.0, transformation_type="reroll_all"),
                    ),
                ),
            )

    def test_is_deterministic_property(self) -> None:
        action = CraftingAction(
            action_id="bench",
            action_type=ActionType.BENCH,
            name="Bench Craft",
            cost=1.0,
            effects=ActionEffects(
                deterministic=True,
                outcomes=(
                    Outcome(probability=1.0, transformation_type="add_specific_mod"),
                ),
            ),
        )
        assert action.is_deterministic is True


class TestCraftingGoal:
    """Tests for CraftingGoal validation."""

    def test_max_steps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Max steps must be >= 1"):
            CraftingGoal(
                goal_type=GoalType.PARTIAL,
                target_modifiers=(ModifierTarget(modifier_id="x"),),
                max_steps=0,
            )

    def test_zero_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="Budget limit must be > 0"):
            CraftingGoal(
                goal_type=GoalType.PARTIAL,
                target_modifiers=(ModifierTarget(modifier_id="x"),),
                budget_limit=0.0,
            )

    def test_valid_goal(self) -> None:
        goal = CraftingGoal(
            goal_type=GoalType.PARTIAL,
            target_modifiers=(ModifierTarget(modifier_id="life_regen"),),
            max_steps=50,
            budget_limit=200.0,
        )
        assert goal.max_steps == 50
        assert goal.budget_limit == 200.0


class TestCraftingStrategy:
    """Tests for the CraftingStrategy enum."""

    def test_all_strategies_accessible(self) -> None:
        assert CraftingStrategy.DETERMINISTIC.value == "deterministic"
        assert CraftingStrategy.CHEAPEST.value == "cheapest"
        assert CraftingStrategy.BALANCED.value == "balanced"
