"""Pydantic models for Path of Exile 2 crafting API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModifierTarget(BaseModel):
    """Target modifier in crafting goal."""

    modifierId: str = Field(..., description="Modifier identifier")
    tier: int = Field(1, description="Target tier")
    weight: float = Field(1.0, description="Weight in scoring function")


class BaseItemSpec(BaseModel):
    """Base item specification."""

    itemType: str = Field(..., description="Item type identifier")
    itemLevel: int = Field(..., description="Item level", ge=1, le=100)
    influence: Optional[str] = Field(None, description="Influence type")


class GoalConstraints(BaseModel):
    """Crafting goal constraints."""

    maxSteps: int = Field(50, description="Maximum crafting steps", ge=1, le=100)
    budgetLimit: float = Field(1000.0, description="Maximum budget in chaos equivalents")


class CraftingGoalModel(BaseModel):
    """Crafting goal definition."""

    goalType: str = Field(..., description="Goal type: exact, partial, or score-based")
    targetModifiers: List[ModifierTarget] = Field(..., description="Target modifiers")
    baseItem: BaseItemSpec = Field(..., description="Base item specification")
    constraints: GoalConstraints = Field(
        default_factory=GoalConstraints, description="Crafting constraints"
    )


class ScoringFunction(BaseModel):
    """Reward scoring function configuration."""

    modifierWeights: Dict[str, float] = Field(
        default_factory=dict, description="Weights for each modifier"
    )
    penaltyForUnwantedMods: float = Field(
        -0.5, description="Penalty for unwanted modifiers"
    )
    progressBonus: float = Field(0.0, description="Bonus for making progress")


class TerminalReward(BaseModel):
    """Terminal state reward values."""

    successValue: float = Field(100.0, description="Reward for reaching goal")
    failureValue: float = Field(-10.0, description="Penalty for failure")


class RewardSystemModel(BaseModel):
    """Reward system configuration."""

    rewardType: str = Field(
        "heuristic", description="Reward type: heuristic, learned, or hybrid"
    )
    scoringFunction: ScoringFunction = Field(
        default_factory=ScoringFunction, description="Scoring function configuration"
    )
    terminalReward: TerminalReward = Field(
        default_factory=TerminalReward, description="Terminal reward values"
    )


class ActionOutcome(BaseModel):
    """Possible outcome of a crafting action."""

    probability: float = Field(..., description="Probability of this outcome")
    stateTransformation: Dict[str, Any] = Field(
        ..., description="State transformation details"
    )


class ActionEffects(BaseModel):
    """Effects of a crafting action."""

    deterministic: bool = Field(..., description="Whether outcome is deterministic")
    outcomes: List[ActionOutcome] = Field(..., description="Possible outcomes")


class ActionPrerequisites(BaseModel):
    """Prerequisites for applying an action."""

    requiredState: Dict[str, Any] = Field(
        default_factory=dict, description="Required state conditions"
    )
    blockedBy: List[str] = Field(default_factory=list, description="Blocking conditions")


class CraftingActionModel(BaseModel):
    """Crafting action definition."""

    actionId: str = Field(..., description="Unique action identifier")
    actionType: str = Field(..., description="Action type: currency, bench, or other")
    name: str = Field(..., description="Human-readable action name")
    cost: float = Field(..., description="Cost in chaos equivalents")
    effects: ActionEffects = Field(..., description="Action effects")
    prerequisites: ActionPrerequisites = Field(
        default_factory=ActionPrerequisites, description="Action prerequisites"
    )


class AvailableActions(BaseModel):
    """Available crafting actions."""

    actions: List[CraftingActionModel] = Field(..., description="List of actions")


class OptimizeRequest(BaseModel):
    """Request for crafting optimization."""

    goal: CraftingGoalModel = Field(..., description="Crafting goal")
    rewardSystem: RewardSystemModel = Field(..., description="Reward system")
    actions: List[CraftingActionModel] = Field(
        default_factory=list, description="Available actions"
    )
    mctsConfig: Dict[str, Any] = Field(
        default_factory=lambda: {
            "numSimulations": 1000,
            "explorationWeight": 1.41,
            "maxSimulationDepth": 50,
        },
        description="MCTS configuration",
    )


class CraftingStep(BaseModel):
    """A step in the crafting path."""

    step: int = Field(..., description="Step number")
    actionId: str = Field(..., description="Action identifier")
    actionName: str = Field(..., description="Action name")
    estimatedCost: float = Field(..., description="Estimated cost")
    expectedReward: float = Field(..., description="Expected reward")
    confidence: float = Field(..., description="Confidence level")


class PathSummary(BaseModel):
    """Summary of crafting path."""

    totalSteps: int = Field(..., description="Total number of steps")
    totalCost: float = Field(..., description="Total estimated cost")
    expectedFinalReward: float = Field(..., description="Expected final reward")
    successProbability: float = Field(..., description="Estimated success probability")
    alternativePaths: int = Field(..., description="Number of alternative paths explored")


class SearchMetadata(BaseModel):
    """Metadata about the search process."""

    iterations: int = Field(..., description="Number of MCTS iterations")
    computationTime: float = Field(..., description="Computation time in seconds")
    treeDepth: int = Field(..., description="Maximum tree depth reached")
    nodesExplored: int = Field(..., description="Total nodes explored")


class OptimizeResponse(BaseModel):
    """Response from crafting optimization."""

    success: bool = Field(..., description="Whether optimization succeeded")
    bestPath: List[CraftingStep] = Field(..., description="Best crafting path found")
    summary: PathSummary = Field(..., description="Path summary")
    metadata: SearchMetadata = Field(..., description="Search metadata")
    searchTree: Dict[str, Any] = Field(
        default_factory=dict, description="Search tree structure"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
