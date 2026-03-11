"""State transition model for the Path of Exile 2 crafting MDP.

The TransitionModel encapsulates the stochastic transition function T(s, a) → s'.
It samples one next state from the probability distribution defined by the
action's outcomes, keeping the domain core free of randomness-related I/O.
"""

from __future__ import annotations

import random

from src.core.mdp.entities import (
    ActionPrerequisites,
    CraftingAction,
    ItemModifier,
    ItemState,
    Outcome,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_outcome(outcomes: tuple[Outcome, ...], rng: random.Random) -> Outcome:
    """Return one outcome sampled according to the probability distribution.

    Args:
        outcomes: Non-empty sequence of outcomes whose probabilities sum to 1.
        rng: Seeded :class:`random.Random` instance used for reproducibility.

    Returns:
        The sampled :class:`Outcome`.
    """
    roll = rng.random()
    cumulative = 0.0
    for outcome in outcomes:
        cumulative += outcome.probability
        if roll < cumulative:
            return outcome
    # Floating-point safety: return the last outcome if roll == 1.0 exactly.
    return outcomes[-1]


def _apply_reroll_all(
    state: ItemState,
    outcome: Outcome,
    rng: random.Random,
) -> ItemState:
    """Replace all modifiers by sampling from the outcome's modifier pool.

    The number of new modifiers is drawn uniformly from
    [min_mod_count, max_mod_count], capped to the pool size.

    Args:
        state: Current item state.
        outcome: The ``reroll_all`` outcome specification.
        rng: Random number generator.

    Returns:
        New :class:`ItemState` with replaced modifiers.
    """
    pool = list(outcome.modifier_pool)
    if not pool:
        return state  # Empty pool: no change (defensive)

    count = rng.randint(
        outcome.min_mod_count, min(outcome.max_mod_count, len(pool))
    )
    chosen_ids = rng.sample(pool, count)
    new_mods: frozenset[ItemModifier] = frozenset(
        ItemModifier(
            modifier_id=mid,
            tier=rng.randint(1, 3),
            value=round(rng.uniform(10.0, 100.0), 1),
        )
        for mid in chosen_ids
    )
    return ItemState(
        base_item=state.base_item,
        modifiers=new_mods,
        step_count=state.step_count,
        accumulated_cost=state.accumulated_cost,
        is_corrupted=state.is_corrupted,
    )


def _apply_add_random_mod(
    state: ItemState,
    outcome: Outcome,
    rng: random.Random,
) -> ItemState:
    """Add one modifier sampled from the pool (if an empty slot exists).

    Items may have at most 6 modifiers.  If the item is already full the state
    is returned unchanged.

    Args:
        state: Current item state.
        outcome: The ``add_random_mod`` outcome specification.
        rng: Random number generator.

    Returns:
        New :class:`ItemState` with the added modifier, or the unchanged state.
    """
    if state.modifier_count >= 6:
        return state

    existing_ids = {m.modifier_id for m in state.modifiers}
    available = [mid for mid in outcome.modifier_pool if mid not in existing_ids]
    if not available:
        return state

    chosen_id = rng.choice(available)
    new_mod = ItemModifier(
        modifier_id=chosen_id,
        tier=rng.randint(1, 3),
        value=round(rng.uniform(10.0, 100.0), 1),
    )
    return ItemState(
        base_item=state.base_item,
        modifiers=state.modifiers | frozenset({new_mod}),
        step_count=state.step_count,
        accumulated_cost=state.accumulated_cost,
        is_corrupted=state.is_corrupted,
    )


def _apply_remove_random_mod(
    state: ItemState,
    rng: random.Random,
) -> ItemState:
    """Remove one modifier chosen at random.

    If the item has no modifiers the state is returned unchanged.

    Args:
        state: Current item state.
        rng: Random number generator.

    Returns:
        New :class:`ItemState` without the removed modifier.
    """
    if not state.modifiers:
        return state

    mod_to_remove = rng.choice(list(state.modifiers))
    return ItemState(
        base_item=state.base_item,
        modifiers=state.modifiers - frozenset({mod_to_remove}),
        step_count=state.step_count,
        accumulated_cost=state.accumulated_cost,
        is_corrupted=state.is_corrupted,
    )


def _apply_add_specific_mod(
    state: ItemState,
    outcome: Outcome,
) -> ItemState:
    """Add an exact modifier (deterministic bench craft).

    If the modifier is already present or the item is full the state is
    returned unchanged.

    Args:
        state: Current item state.
        outcome: The ``add_specific_mod`` outcome specification.

    Returns:
        New :class:`ItemState` with the added modifier.
    """
    if outcome.specific_modifier is None:
        return state
    if state.modifier_count >= 6:
        return state
    if state.has_modifier(outcome.specific_modifier.modifier_id):
        return state

    return ItemState(
        base_item=state.base_item,
        modifiers=state.modifiers | frozenset({outcome.specific_modifier}),
        step_count=state.step_count,
        accumulated_cost=state.accumulated_cost,
        is_corrupted=state.is_corrupted,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SUPPORTED_STATES = frozenset(
    {"is_rare", "is_magic", "is_normal", "not_corrupted", "has_empty_mod_slot"}
)


def _satisfies_required_state(
    state: ItemState, required: str | None
) -> bool:
    """Return True if the item state satisfies the named prerequisite.

    Args:
        state: Current item state.
        required: Named condition string, or None (no requirement).

    Returns:
        Boolean indicating whether the prerequisite is met.
    """
    if required is None:
        return True
    if required == "is_rare":
        return state.modifier_count >= 3
    if required == "is_magic":
        return 1 <= state.modifier_count <= 2
    if required == "is_normal":
        return state.modifier_count == 0
    if required == "not_corrupted":
        return not state.is_corrupted
    if required == "has_empty_mod_slot":
        return state.modifier_count < 6
    return True


class TransitionModel:
    """Stochastic transition function T(s, a) for the crafting MDP.

    The model is free of side effects; all randomness is channelled through an
    injected :class:`random.Random` instance so tests can reproduce results by
    seeding it.

    This is an **outbound port implementation** that belongs in the domain core
    because it depends only on domain entities.

    Args:
        rng: Random number generator used for all sampling operations.
            Defaults to a freshly seeded global-state instance if not provided.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng: random.Random = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def can_apply(self, state: ItemState, action: CraftingAction) -> bool:
        """Return True when *action* is applicable in *state*.

        Checks the action prerequisites (required_state, blocked_by) and
        whether the item is corrupted.

        Args:
            state: Current item state.
            action: The action to evaluate.

        Returns:
            True if the action may be applied.
        """
        if state.is_corrupted:
            return False
        prereqs: ActionPrerequisites = action.prerequisites
        if not _satisfies_required_state(state, prereqs.required_state):
            return False
        return True

    def apply(self, state: ItemState, action: CraftingAction) -> ItemState:
        """Sample one next state by applying *action* to *state*.

        This implements the transition function T(s, a) → s' by sampling an
        outcome from the action's probability distribution and transforming the
        item accordingly.  The returned state has an updated ``step_count`` and
        ``accumulated_cost``.

        Args:
            state: Current item state.
            action: The crafting action to apply.

        Returns:
            New :class:`ItemState` sampled from the transition distribution.

        Raises:
            ValueError: If the action cannot be applied in the current state.
        """
        if not self.can_apply(state, action):
            raise ValueError(
                f"Action '{action.action_id}' cannot be applied in the current state."
            )

        outcome = _sample_outcome(action.effects.outcomes, self._rng)
        transformed = self._apply_outcome(state, outcome)

        # Advance step counter and cost on the transformed state.
        return ItemState(
            base_item=transformed.base_item,
            modifiers=transformed.modifiers,
            step_count=state.step_count + 1,
            accumulated_cost=state.accumulated_cost + action.cost,
            is_corrupted=transformed.is_corrupted,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_outcome(self, state: ItemState, outcome: Outcome) -> ItemState:
        """Dispatch to the correct transformation based on the outcome type.

        Args:
            state: Current item state before transformation.
            outcome: The sampled outcome to apply.

        Returns:
            Transformed :class:`ItemState` (step count and cost are not updated
            here; the caller is responsible for that).
        """
        t = outcome.transformation_type
        if t == "reroll_all":
            return _apply_reroll_all(state, outcome, self._rng)
        if t == "add_random_mod":
            return _apply_add_random_mod(state, outcome, self._rng)
        if t == "remove_random_mod":
            return _apply_remove_random_mod(state, self._rng)
        if t == "add_specific_mod":
            return _apply_add_specific_mod(state, outcome)
        # Unknown transformation type: return state unchanged.
        return state
