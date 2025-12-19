"""FastAPI routes for MCTS operations."""
from typing import Dict

from fastapi import APIRouter, HTTPException, status

from src.application.use_cases.mcts_service import MCTSService
from src.domain.entities.mdp import Action, MDP, State, Transition
from src.infrastructure.adapters.api.models import (
    ActionModel,
    ErrorResponse,
    HealthResponse,
    MCTSSearchRequest,
    MCTSSearchResponse,
    StateModel,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@router.post(
    "/mcts/search",
    response_model=MCTSSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def mcts_search(request: MCTSSearchRequest) -> MCTSSearchResponse:
    """
    Perform MCTS search on the given MDP.

    Args:
        request: MCTS search request with MDP and parameters

    Returns:
        MCTS search response with best action and search tree

    Raises:
        HTTPException: If search fails or invalid parameters
    """
    try:
        # Convert API models to domain entities
        states_map = {
            state_model.id: State(
                id=state_model.id,
                data=state_model.data,
                is_terminal=state_model.is_terminal,
            )
            for state_model in request.mdp.states
        }

        actions_map = {
            action_model.id: Action(
                id=action_model.id,
                name=action_model.name,
                parameters=action_model.parameters,
            )
            for action_model in request.mdp.actions
        }

        # Validate initial state exists
        if request.mdp.initial_state_id not in states_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Initial state '{request.mdp.initial_state_id}' not found in states",
            )

        initial_state = states_map[request.mdp.initial_state_id]

        # Create MDP
        mdp = MDP(
            states=list(states_map.values()),
            actions=list(actions_map.values()),
            initial_state=initial_state,
            gamma=request.mdp.gamma,
        )

        # Add transitions
        for trans_model in request.mdp.transitions:
            if trans_model.from_state_id not in states_map:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Transition from_state '{trans_model.from_state_id}' not found",
                )
            if trans_model.to_state_id not in states_map:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Transition to_state '{trans_model.to_state_id}' not found",
                )
            if trans_model.action_id not in actions_map:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Transition action '{trans_model.action_id}' not found",
                )

            transition = Transition(
                from_state=states_map[trans_model.from_state_id],
                action=actions_map[trans_model.action_id],
                to_state=states_map[trans_model.to_state_id],
                reward=trans_model.reward,
                probability=trans_model.probability,
            )
            mdp.add_transition(transition)

        # Perform MCTS search
        mcts_service = MCTSService(exploration_weight=request.exploration_weight)
        best_action = mcts_service.search(
            mdp=mdp, initial_state=initial_state, num_simulations=request.num_simulations
        )

        # Convert result back to API model
        best_action_model = None
        if best_action:
            best_action_model = ActionModel(
                id=best_action.id,
                name=best_action.name,
                parameters=best_action.parameters,
            )

        # Get search tree
        search_tree = {}
        if mcts_service.root:
            search_tree = mcts_service.get_search_tree(mcts_service.root)

        return MCTSSearchResponse(
            best_action=best_action_model,
            search_tree=search_tree,
            simulations_run=request.num_simulations,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log the error internally (in production, use proper logging)
        print(f"MCTS search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MCTS search failed. Please check your request and try again.",
        )
