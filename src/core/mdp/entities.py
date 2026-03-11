"""Domain entities for the Path of Exile 2 crafting MDP model.

All entities are immutable value objects (frozen dataclasses) to guarantee
correctness in the MCTS tree where the same state may be visited from multiple
paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class GoalType(StrEnum):
    """Classification of the crafting goal evaluation mode."""

    EXACT = "exact"
    PARTIAL = "partial"
    SCORE_BASED = "score_based"


class ActionType(StrEnum):
    """Classification of the crafting action source."""

    CURRENCY = "currency"
    BENCH = "bench"
    OTHER = "other"


class CraftingStrategy(StrEnum):
    """Optimization strategy that drives the MCTS reward and simulation policy.

    DETERMINISTIC  – minimises variance; prefers reliable, deterministic paths.
    CHEAPEST       – minimises expected currency cost to reach the goal.
    BALANCED       – balances goal-progress, cost, and path reliability.
    """

    DETERMINISTIC = "deterministic"
    CHEAPEST = "cheapest"
    BALANCED = "balanced"


@dataclass(frozen=True)
class ItemModifier:
    """A single modifier present on a crafted item.

    Args:
        modifier_id: Unique identifier of the modifier (e.g. ``"life_regen"``).
        tier: Tier of the modifier (1 = best).  Must be ≥ 1.
        value: Numeric roll value within the tier range.
    """

    modifier_id: str
    tier: int
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.tier < 1:
            raise ValueError(f"Modifier tier must be >= 1, got {self.tier}")


@dataclass(frozen=True)
class BaseItem:
    """Static description of the item being crafted.

    Args:
        item_type: Category of the item (e.g. ``"chest_armour"``).
        item_level: Item level (1–100).  Determines which modifiers can roll.
        influence: Optional influence tag (e.g. ``"hunter"``, ``"elder"``).
    """

    item_type: str
    item_level: int
    influence: str | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.item_level <= 100):
            raise ValueError(
                f"Item level must be between 1 and 100, got {self.item_level}"
            )


@dataclass(frozen=True)
class ItemState:
    """Snapshot of an item at a specific point in the crafting sequence.

    Represents the MDP **state** S.  Immutable so that nodes in the MCTS tree
    can safely share references to the same state object.

    Args:
        base_item: Static base-item information (does not change during crafting).
        modifiers: Current set of modifiers on the item.
        step_count: Number of crafting actions applied so far.
        accumulated_cost: Total currency cost spent reaching this state.
        is_corrupted: Whether the item has been corrupted (blocks most actions).
    """

    base_item: BaseItem
    modifiers: frozenset[ItemModifier]
    step_count: int = 0
    accumulated_cost: float = 0.0
    is_corrupted: bool = False

    @property
    def modifier_count(self) -> int:
        """Return the number of modifiers currently on the item."""
        return len(self.modifiers)

    def has_modifier(self, modifier_id: str) -> bool:
        """Return True if the item has a modifier with the given ID."""
        return any(m.modifier_id == modifier_id for m in self.modifiers)

    def get_modifier(self, modifier_id: str) -> ItemModifier | None:
        """Return the modifier with the given ID, or None if not present."""
        for mod in self.modifiers:
            if mod.modifier_id == modifier_id:
                return mod
        return None


@dataclass(frozen=True)
class Outcome:
    """One possible result of a crafting action.

    Args:
        probability: Probability in [0, 1] that this outcome occurs.
        transformation_type: How the item state changes.  Supported values:
            ``"reroll_all"`` – replace all modifiers from a pool,
            ``"add_random_mod"`` – add one random modifier from a pool,
            ``"remove_random_mod"`` – remove one random modifier,
            ``"add_specific_mod"`` – add an exact modifier (bench craft).
        modifier_pool: Pool of ``modifier_id`` strings to sample from
            (used by ``reroll_all`` and ``add_random_mod``).
        specific_modifier: Exact modifier to add (used by ``add_specific_mod``).
        min_mod_count: Minimum number of modifiers after reroll (default 3).
        max_mod_count: Maximum number of modifiers after reroll (default 6).
    """

    probability: float
    transformation_type: str
    modifier_pool: tuple[str, ...] = field(default_factory=tuple)
    specific_modifier: ItemModifier | None = None
    min_mod_count: int = 3
    max_mod_count: int = 6

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"Probability must be in [0, 1], got {self.probability}"
            )


@dataclass(frozen=True)
class ActionEffects:
    """Describes all possible outcomes of a crafting action.

    Args:
        deterministic: True when the action always produces an identical result.
        outcomes: Probability-weighted list of possible item transformations.
            Probabilities must sum to 1.0 (within floating-point tolerance).
    """

    deterministic: bool
    outcomes: tuple[Outcome, ...]

    def __post_init__(self) -> None:
        total = sum(o.probability for o in self.outcomes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Outcome probabilities must sum to 1.0, got {total}"
            )


@dataclass(frozen=True)
class ActionPrerequisites:
    """Conditions that must be satisfied before an action may be applied.

    Args:
        required_state: Named condition the item must satisfy.  Supported
            values: ``"is_rare"``, ``"is_magic"``, ``"is_normal"``,
            ``"not_corrupted"``, ``"has_empty_mod_slot"``.
        blocked_by: Set of ``action_id`` values that block this action when
            the same action has already been applied in the current path.
    """

    required_state: str | None = None
    blocked_by: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CraftingAction:
    """A crafting operation that may be applied to an item.

    Represents the MDP **action** A.

    Args:
        action_id: Unique identifier for this action (e.g. ``"chaos_orb"``).
        action_type: Source/category of the action.
        name: Human-readable display name.
        cost: Expected currency cost in Chaos Orb equivalents.  Must be ≥ 0.
        effects: Probabilistic effects applied when the action is used.
        prerequisites: Optional conditions required to apply the action.
    """

    action_id: str
    action_type: ActionType
    name: str
    cost: float
    effects: ActionEffects
    prerequisites: ActionPrerequisites = field(
        default_factory=ActionPrerequisites
    )

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"Action cost must be >= 0, got {self.cost}")

    @property
    def is_deterministic(self) -> bool:
        """Return True when the action always produces the same outcome."""
        return self.effects.deterministic


@dataclass(frozen=True)
class ModifierTarget:
    """Specification for a desired modifier in the crafting goal.

    Args:
        modifier_id: The modifier that must be present.
        min_tier: Minimum acceptable tier (lower = better).  Defaults to 1.
        weight: Contribution weight when computing goal progress.
            Must be ≥ 0.
    """

    modifier_id: str
    min_tier: int = 1
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"Modifier weight must be >= 0, got {self.weight}"
            )
        if self.min_tier < 1:
            raise ValueError(
                f"Min tier must be >= 1, got {self.min_tier}"
            )


@dataclass(frozen=True)
class CraftingGoal:
    """Defines the desired outcome of the crafting process.

    Args:
        goal_type: How the goal is evaluated.
        target_modifiers: Set of modifiers (with weights) that define success.
        max_steps: Hard limit on the number of crafting actions.  Must be ≥ 1.
        budget_limit: Optional upper bound on total currency cost.
    """

    goal_type: GoalType
    target_modifiers: tuple[ModifierTarget, ...]
    max_steps: int = 50
    budget_limit: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(
                f"Max steps must be >= 1, got {self.max_steps}"
            )
        if self.budget_limit is not None and self.budget_limit <= 0:
            raise ValueError(
                f"Budget limit must be > 0, got {self.budget_limit}"
            )
