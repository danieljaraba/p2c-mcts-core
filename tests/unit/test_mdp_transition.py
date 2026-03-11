"""Unit tests for the TransitionModel."""

from __future__ import annotations

import random

import pytest

from src.core.mdp.entities import (
    ActionEffects,
    ActionType,
    BaseItem,
    CraftingAction,
    ItemModifier,
    ItemState,
    Outcome,
)
from src.core.mdp.transition import TransitionModel


@pytest.fixture
def seeded_transition() -> TransitionModel:
    """Return a TransitionModel with a fixed seed for reproducibility."""
    return TransitionModel(rng=random.Random(42))


@pytest.fixture
def empty_state() -> ItemState:  # noqa: F811
    return ItemState(base_item=BaseItem("helmet", 86), modifiers=frozenset())


class TestCanApply:
    """Tests for TransitionModel.can_apply."""

    def test_can_apply_basic_action(
        self,
        seeded_transition: TransitionModel,
        chaos_orb_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        assert seeded_transition.can_apply(empty_state, chaos_orb_action) is True

    def test_cannot_apply_to_corrupted_item(
        self,
        seeded_transition: TransitionModel,
        chaos_orb_action: CraftingAction,
    ) -> None:
        corrupted = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset(),
            is_corrupted=True,
        )
        assert seeded_transition.can_apply(corrupted, chaos_orb_action) is False

    def test_bench_craft_requires_empty_slot(
        self,
        seeded_transition: TransitionModel,
        bench_craft_action: CraftingAction,
    ) -> None:
        # Create a full item (6 modifiers).
        full_mods = frozenset(
            ItemModifier(modifier_id=f"mod_{i}", tier=1) for i in range(6)
        )
        full_state = ItemState(
            base_item=BaseItem("helmet", 86), modifiers=full_mods
        )
        assert seeded_transition.can_apply(full_state, bench_craft_action) is False

    def test_bench_craft_allowed_with_empty_slot(
        self,
        seeded_transition: TransitionModel,
        bench_craft_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        assert seeded_transition.can_apply(empty_state, bench_craft_action) is True


class TestApply:
    """Tests for TransitionModel.apply."""

    def test_chaos_orb_changes_state(
        self,
        seeded_transition: TransitionModel,
        chaos_orb_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        next_state = seeded_transition.apply(empty_state, chaos_orb_action)
        assert next_state.modifier_count >= 3
        assert next_state.step_count == 1
        assert next_state.accumulated_cost == pytest.approx(1.5)

    def test_bench_craft_adds_specific_modifier(
        self,
        seeded_transition: TransitionModel,
        bench_craft_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        next_state = seeded_transition.apply(empty_state, bench_craft_action)
        assert next_state.has_modifier("life_regen") is True
        assert next_state.step_count == 1
        assert next_state.accumulated_cost == pytest.approx(2.0)

    def test_exalted_orb_adds_random_modifier(
        self,
        seeded_transition: TransitionModel,
        exalted_orb_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        next_state = seeded_transition.apply(empty_state, exalted_orb_action)
        assert next_state.modifier_count == 1

    def test_apply_raises_for_inapplicable_action(
        self,
        seeded_transition: TransitionModel,
        bench_craft_action: CraftingAction,
    ) -> None:
        corrupted = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset(),
            is_corrupted=True,
        )
        with pytest.raises(ValueError, match="cannot be applied"):
            seeded_transition.apply(corrupted, bench_craft_action)

    def test_cost_accumulates_across_steps(
        self,
        seeded_transition: TransitionModel,
        chaos_orb_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        state = seeded_transition.apply(empty_state, chaos_orb_action)
        state2 = seeded_transition.apply(state, chaos_orb_action)
        assert state2.accumulated_cost == pytest.approx(3.0)
        assert state2.step_count == 2

    def test_remove_random_mod(
        self,
        seeded_transition: TransitionModel,
    ) -> None:
        remove_action = CraftingAction(
            action_id="annulment",
            action_type=ActionType.CURRENCY,
            name="Annulment Orb",
            cost=5.0,
            effects=ActionEffects(
                deterministic=False,
                outcomes=(
                    Outcome(
                        probability=1.0,
                        transformation_type="remove_random_mod",
                    ),
                ),
            ),
        )
        state_with_mod = ItemState(
            base_item=BaseItem("helmet", 86),
            modifiers=frozenset({ItemModifier("life_regen", 1)}),
        )
        next_state = seeded_transition.apply(state_with_mod, remove_action)
        assert next_state.modifier_count == 0

    def test_reroll_does_not_exceed_max_count(
        self,
        seeded_transition: TransitionModel,
        chaos_orb_action: CraftingAction,
        empty_state: ItemState,
    ) -> None:
        for _ in range(20):  # Run multiple times to check consistency.
            next_state = seeded_transition.apply(empty_state, chaos_orb_action)
            assert next_state.modifier_count <= 6

    def test_unknown_transformation_returns_unchanged_state(
        self,
        seeded_transition: TransitionModel,
        empty_state: ItemState,
    ) -> None:
        unknown_action = CraftingAction(
            action_id="unknown",
            action_type=ActionType.OTHER,
            name="Unknown",
            cost=0.0,
            effects=ActionEffects(
                deterministic=True,
                outcomes=(
                    Outcome(
                        probability=1.0,
                        transformation_type="teleport",  # unsupported
                    ),
                ),
            ),
        )
        next_state = seeded_transition.apply(empty_state, unknown_action)
        assert next_state.modifier_count == 0
        assert next_state.step_count == 1
