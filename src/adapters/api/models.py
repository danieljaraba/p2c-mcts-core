"""Pydantic v2 request and response models for the crafting API.

These models belong exclusively to the inbound HTTP adapter layer.  The domain
core is never imported into request/response bodies; domain entities are
constructed from these models in the route handlers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class BaseItemModel(BaseModel):
    """Base item description."""

    item_type: str = Field(..., description="Item category, e.g. 'chest_armour'.")
    item_level: int = Field(..., ge=1, le=100, description="Item level (1–100).")
    influence: str | None = Field(None, description="Optional influence tag.")


class TargetModifierModel(BaseModel):
    """Desired modifier in the crafting goal."""

    modifier_id: str = Field(..., description="Modifier identifier.")
    min_tier: int = Field(
        1, ge=1, description="Maximum acceptable tier (lower = better)."
    )
    weight: float = Field(1.0, ge=0.0, description="Relative importance weight.")


class OutcomeModel(BaseModel):
    """One possible result of a crafting action."""

    probability: float = Field(..., ge=0.0, le=1.0)
    transformation_type: str = Field(
        ...,
        description=(
            "One of: 'reroll_all', 'add_random_mod', "
            "'remove_random_mod', 'add_specific_mod'."
        ),
    )
    modifier_pool: list[str] = Field(
        default_factory=list,
        description="Pool of modifier IDs to sample from.",
    )
    specific_modifier_id: str | None = Field(
        None, description="Exact modifier ID for deterministic bench crafts."
    )
    specific_modifier_tier: int | None = Field(
        None, ge=1, description="Tier of the specific modifier."
    )
    specific_modifier_value: float | None = Field(
        None, description="Numeric value of the specific modifier."
    )
    min_mod_count: int = Field(3, ge=1, description="Minimum modifiers after reroll.")
    max_mod_count: int = Field(6, ge=1, description="Maximum modifiers after reroll.")


class ActionEffectsModel(BaseModel):
    """Effects specification for a crafting action."""

    deterministic: bool = Field(
        ..., description="True when all outcomes are identical."
    )
    outcomes: list[OutcomeModel] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> ActionEffectsModel:
        """Validate that outcome probabilities sum to 1.0."""
        total = sum(o.probability for o in self.outcomes)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Outcome probabilities must sum to 1.0, got {total:.6f}."
            )
        return self


class ActionPrerequisitesModel(BaseModel):
    """Prerequisites for applying a crafting action."""

    required_state: str | None = Field(
        None,
        description=(
            "Named condition: 'is_rare', 'is_magic', 'is_normal', "
            "'not_corrupted', 'has_empty_mod_slot'."
        ),
    )
    blocked_by: list[str] = Field(
        default_factory=list,
        description="Action IDs that block this action.",
    )


class CraftingActionModel(BaseModel):
    """A crafting action available to the player."""

    action_id: str = Field(..., description="Unique action identifier.")
    action_type: str = Field(
        ..., description="Action category: 'currency', 'bench', or 'other'."
    )
    name: str = Field(..., description="Human-readable display name.")
    cost: float = Field(..., ge=0.0, description="Estimated Chaos Orb equivalent cost.")
    effects: ActionEffectsModel
    prerequisites: ActionPrerequisitesModel = Field(
        default_factory=ActionPrerequisitesModel
    )


class ScoringFunctionModel(BaseModel):
    """Parameters for the heuristic scoring function (optional, informational)."""

    modifier_weights: dict[str, float] = Field(default_factory=dict)
    penalty_for_unwanted_mods: float = Field(0.0)
    progress_bonus: float = Field(0.0)


class RewardSystemModel(BaseModel):
    """Reward system specification (informational; reward is computed internally)."""

    reward_type: str = Field(
        "heuristic",
        description="One of: 'heuristic', 'learned', 'hybrid'.",
    )
    scoring_function: ScoringFunctionModel = Field(
        default_factory=ScoringFunctionModel
    )


class GoalConstraintsModel(BaseModel):
    """Hard constraints on the crafting search."""

    max_steps: int = Field(50, ge=1, description="Maximum crafting steps.")
    budget_limit: float | None = Field(
        None, gt=0.0, description="Maximum currency budget (Chaos Orb equivalents)."
    )


class GoalModel(BaseModel):
    """Crafting goal definition."""

    goal_type: str = Field(
        "partial",
        description="One of: 'exact', 'partial', 'score_based'.",
    )
    target_modifiers: list[TargetModifierModel] = Field(..., min_length=1)
    base_item: BaseItemModel
    constraints: GoalConstraintsModel = Field(default_factory=GoalConstraintsModel)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """Request body for the /optimize endpoint (all strategies)."""

    goal: GoalModel
    reward_system: RewardSystemModel = Field(default_factory=RewardSystemModel)
    actions: list[CraftingActionModel] = Field(..., min_length=1)


class OptimizeStrategyRequest(BaseModel):
    """Request body for a single-strategy /optimize/{strategy} endpoint."""

    goal: GoalModel
    reward_system: RewardSystemModel = Field(default_factory=RewardSystemModel)
    actions: list[CraftingActionModel] = Field(..., min_length=1)
    iterations: int = Field(500, ge=10, le=10000, description="MCTS iteration count.")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PathStepResponse(BaseModel):
    """One step in the recommended crafting path."""

    step: int
    action_id: str
    action_name: str
    estimated_cost: float
    expected_reward: float
    confidence: float


class SearchSummaryResponse(BaseModel):
    """Aggregated statistics for a crafting path."""

    total_steps: int
    total_cost: float
    expected_final_reward: float
    success_probability: float
    alternative_paths: int


class SearchMetadataResponse(BaseModel):
    """MCTS run metadata."""

    iterations: int
    computation_time: float
    tree_depth: int
    nodes_explored: int


class StrategyResultResponse(BaseModel):
    """Result for a single optimisation strategy."""

    success: bool
    strategy: str
    best_path: list[PathStepResponse]
    summary: SearchSummaryResponse
    metadata: SearchMetadataResponse


class OptimizeAllResponse(BaseModel):
    """Response containing results for all three strategies."""

    deterministic: StrategyResultResponse
    cheapest: StrategyResultResponse
    balanced: StrategyResultResponse
