"""FastAPI routes for Path of Exile 2 crafting optimization."""
import time
from typing import Dict

from fastapi import APIRouter, HTTPException, status

from src.application.use_cases.crafting_mcts_service import CraftingMCTSService
from src.domain.entities.crafting import (
    CraftingAction,
    CraftingGoal,
    ItemState,
    Modifier,
    ModifierType,
    Rarity,
    RewardSystem,
)
from src.infrastructure.adapters.api.crafting_models import (
    CraftingStep,
    ErrorResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    PathSummary,
    SearchMetadata,
)
from src.infrastructure.config_loader import config_loader

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@router.post(
    "/api/v1/optimize",
    response_model=OptimizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def optimize_crafting(request: OptimizeRequest) -> OptimizeResponse:
    """
    Optimize crafting path for Path of Exile 2 items.

    Args:
        request: Crafting optimization request

    Returns:
        Optimal crafting path with metadata

    Raises:
        HTTPException: If optimization fails
    """
    try:
        start_time = time.time()

        # Create initial item state
        base_item_data = config_loader.get_base_item(request.goal.baseItem.itemType)

        initial_state = ItemState(
            itemType=request.goal.baseItem.itemType,
            itemLevel=request.goal.baseItem.itemLevel,
            rarity=Rarity.NORMAL,
            modifiers=[],
            influence=request.goal.baseItem.influence,
            baseStats=base_item_data.get("baseStats", {}),
            is_terminal=False,
        )

        # Create crafting goal
        goal = CraftingGoal(
            goalType=request.goal.goalType,
            targetModifiers=[
                {
                    "modifierId": mod.modifierId,
                    "tier": mod.tier,
                    "weight": mod.weight,
                }
                for mod in request.goal.targetModifiers
            ],
            baseItem={
                "itemType": request.goal.baseItem.itemType,
                "itemLevel": request.goal.baseItem.itemLevel,
                "influence": request.goal.baseItem.influence,
            },
            constraints={
                "maxSteps": request.goal.constraints.maxSteps,
                "budgetLimit": request.goal.constraints.budgetLimit,
            },
        )

        # Create reward system
        reward_system = RewardSystem(
            rewardType=request.rewardSystem.rewardType,
            scoringFunction={
                "modifierWeights": request.rewardSystem.scoringFunction.modifierWeights,
                "penaltyForUnwantedMods": request.rewardSystem.scoringFunction.penaltyForUnwantedMods,
                "progressBonus": request.rewardSystem.scoringFunction.progressBonus,
            },
            terminalReward={
                "successValue": request.rewardSystem.terminalReward.successValue,
                "failureValue": request.rewardSystem.terminalReward.failureValue,
            },
        )

        # Create available actions
        if request.actions:
            # Use provided actions
            available_actions = [
                CraftingAction(
                    actionId=action.actionId,
                    actionType=action.actionType,
                    name=action.name,
                    cost=action.cost,
                    deterministic=action.effects.deterministic,
                    prerequisites=action.prerequisites.model_dump() if action.prerequisites else {},
                )
                for action in request.actions
            ]
        else:
            # Use all configured currencies
            available_actions = [
                config_loader.create_crafting_action(currency)
                for currency in config_loader.get_all_currencies()
            ]

        # Run MCTS
        mcts_config = request.mctsConfig
        num_simulations = mcts_config.get("numSimulations", 1000)
        exploration_weight = mcts_config.get("explorationWeight", 1.41)
        max_depth = mcts_config.get("maxSimulationDepth", 50)

        service = CraftingMCTSService(
            exploration_weight=exploration_weight, max_simulation_depth=max_depth
        )

        best_action = service.search(
            initial_state=initial_state,
            goal=goal,
            reward_system=reward_system,
            available_actions=available_actions,
            num_simulations=num_simulations,
        )

        end_time = time.time()
        computation_time = end_time - start_time

        # Build response
        best_path = []
        if best_action and service.root:
            # Get the best child to extract path info
            best_child = service.root.get_most_visited_child()
            if best_child:
                avg_value = best_child.value / best_child.visits if best_child.visits > 0 else 0
                confidence = min(1.0, best_child.visits / num_simulations)

                best_path.append(
                    CraftingStep(
                        step=1,
                        actionId=best_action.actionId,
                        actionName=best_action.name,
                        estimatedCost=best_action.cost,
                        expectedReward=avg_value,
                        confidence=confidence,
                    )
                )

        # Calculate tree statistics
        nodes_explored = 0
        max_depth_reached = 0

        def count_nodes(node, depth=0):
            nonlocal nodes_explored, max_depth_reached
            nodes_explored += 1
            max_depth_reached = max(max_depth_reached, depth)
            for child in node.children:
                count_nodes(child, depth + 1)

        if service.root:
            count_nodes(service.root)

        total_cost = sum(step.estimatedCost for step in best_path)
        success_probability = 0.0
        if service.root and service.root.visits > 0:
            success_probability = service.root.value / (service.root.visits * 100)
            success_probability = max(0.0, min(1.0, success_probability))

        summary = PathSummary(
            totalSteps=len(best_path),
            totalCost=total_cost,
            expectedFinalReward=best_path[0].expectedReward if best_path else 0.0,
            successProbability=success_probability,
            alternativePaths=len(service.root.children) if service.root else 0,
        )

        metadata = SearchMetadata(
            iterations=num_simulations,
            computationTime=computation_time,
            treeDepth=max_depth_reached,
            nodesExplored=nodes_explored,
        )

        search_tree = {}
        if service.root:
            search_tree = service.get_search_tree(service.root)

        return OptimizeResponse(
            success=best_action is not None,
            bestPath=best_path,
            summary=summary,
            metadata=metadata,
            searchTree=search_tree,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        # Log error internally
        print(f"Optimization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed. Please check your request and try again.",
        )
