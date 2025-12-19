"""Pydantic models for API requests and responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StateModel(BaseModel):
    """API model for State."""

    id: str = Field(..., description="Unique identifier for the state")
    data: Dict[str, Any] = Field(default_factory=dict, description="State data")
    is_terminal: bool = Field(False, description="Whether this is a terminal state")


class ActionModel(BaseModel):
    """API model for Action."""

    id: str = Field(..., description="Unique identifier for the action")
    name: str = Field(..., description="Human-readable action name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )


class TransitionModel(BaseModel):
    """API model for Transition."""

    from_state_id: str = Field(..., description="Source state ID")
    action_id: str = Field(..., description="Action ID")
    to_state_id: str = Field(..., description="Destination state ID")
    reward: float = Field(..., description="Reward for this transition")
    probability: float = Field(1.0, description="Transition probability")


class MDPModel(BaseModel):
    """API model for MDP."""

    states: List[StateModel] = Field(..., description="List of states")
    actions: List[ActionModel] = Field(..., description="List of actions")
    transitions: List[TransitionModel] = Field(..., description="List of transitions")
    initial_state_id: str = Field(..., description="Initial state ID")
    gamma: float = Field(0.95, description="Discount factor", ge=0.0, le=1.0)


class MCTSSearchRequest(BaseModel):
    """Request model for MCTS search."""

    mdp: MDPModel = Field(..., description="MDP model to search")
    num_simulations: int = Field(
        100, description="Number of MCTS simulations", ge=1, le=10000
    )
    exploration_weight: float = Field(
        1.41, description="UCB1 exploration weight", ge=0.0
    )


class MCTSSearchResponse(BaseModel):
    """Response model for MCTS search."""

    best_action: Optional[ActionModel] = Field(
        None, description="Best action found by MCTS"
    )
    search_tree: Dict[str, Any] = Field(..., description="MCTS search tree structure")
    simulations_run: int = Field(..., description="Number of simulations performed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
