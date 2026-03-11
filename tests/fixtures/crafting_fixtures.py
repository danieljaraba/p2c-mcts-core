"""Reusable pytest fixtures for crafting domain entities."""

from __future__ import annotations

import pytest

from src.core.mdp.entities import (
    ActionEffects,
    ActionPrerequisites,
    ActionType,
    BaseItem,
    CraftingAction,
    CraftingGoal,
    GoalType,
    ItemModifier,
    ItemState,
    ModifierTarget,
    Outcome,
)

# ---------------------------------------------------------------------------
# Base item fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_item() -> BaseItem:
    """Return a standard chest armour base item."""
    return BaseItem(item_type="chest_armour", item_level=86, influence="hunter")


@pytest.fixture
def empty_state(base_item: BaseItem) -> ItemState:
    """Return an item state with no modifiers."""
    return ItemState(base_item=base_item, modifiers=frozenset())


@pytest.fixture
def partial_state(base_item: BaseItem) -> ItemState:
    """Return an item state with one matching modifier."""
    return ItemState(
        base_item=base_item,
        modifiers=frozenset(
            {ItemModifier(modifier_id="life_regen", tier=1, value=50.0)}
        ),
    )


# ---------------------------------------------------------------------------
# Action fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chaos_orb_action() -> CraftingAction:
    """Return a Chaos Orb action that rerolls all modifiers."""
    pool = ("life_regen", "crit_chance", "fire_res", "cold_res", "flat_life", "mana")
    return CraftingAction(
        action_id="chaos_orb",
        action_type=ActionType.CURRENCY,
        name="Chaos Orb",
        cost=1.5,
        effects=ActionEffects(
            deterministic=False,
            outcomes=(
                Outcome(
                    probability=1.0,
                    transformation_type="reroll_all",
                    modifier_pool=pool,
                    min_mod_count=3,
                    max_mod_count=6,
                ),
            ),
        ),
    )


@pytest.fixture
def bench_craft_action() -> CraftingAction:
    """Return a bench craft action that adds a specific modifier."""
    return CraftingAction(
        action_id="bench_life_regen",
        action_type=ActionType.BENCH,
        name="Bench: Life Regeneration",
        cost=2.0,
        effects=ActionEffects(
            deterministic=True,
            outcomes=(
                Outcome(
                    probability=1.0,
                    transformation_type="add_specific_mod",
                    specific_modifier=ItemModifier(
                        modifier_id="life_regen", tier=1, value=50.0
                    ),
                ),
            ),
        ),
        prerequisites=ActionPrerequisites(required_state="has_empty_mod_slot"),
    )


@pytest.fixture
def exalted_orb_action() -> CraftingAction:
    """Return an Exalted Orb action that adds a random modifier."""
    pool = ("life_regen", "crit_chance", "fire_res", "cold_res", "flat_life", "mana")
    return CraftingAction(
        action_id="exalted_orb",
        action_type=ActionType.CURRENCY,
        name="Exalted Orb",
        cost=100.0,
        effects=ActionEffects(
            deterministic=False,
            outcomes=(
                Outcome(
                    probability=1.0,
                    transformation_type="add_random_mod",
                    modifier_pool=pool,
                ),
            ),
        ),
        prerequisites=ActionPrerequisites(required_state="has_empty_mod_slot"),
    )


# ---------------------------------------------------------------------------
# Goal fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def partial_goal() -> CraftingGoal:
    """Return a partial crafting goal requiring life_regen and crit_chance."""
    return CraftingGoal(
        goal_type=GoalType.PARTIAL,
        target_modifiers=(
            ModifierTarget(modifier_id="life_regen", min_tier=1, weight=1.0),
            ModifierTarget(modifier_id="crit_chance", min_tier=2, weight=0.8),
        ),
        max_steps=30,
        budget_limit=200.0,
    )


@pytest.fixture
def single_mod_goal() -> CraftingGoal:
    """Return a simple goal requiring only one modifier."""
    return CraftingGoal(
        goal_type=GoalType.PARTIAL,
        target_modifiers=(
            ModifierTarget(modifier_id="life_regen", min_tier=1, weight=1.0),
        ),
        max_steps=20,
        budget_limit=100.0,
    )
