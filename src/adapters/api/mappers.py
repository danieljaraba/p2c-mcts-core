"""Mapper functions: translate between API models and domain entities.

Mappers live in the adapter layer and are the only code allowed to import both
Pydantic API models and domain entities.  This keeps the domain core free of
serialisation concerns.
"""

from __future__ import annotations

from src.adapters.api.models import (
    CraftingActionModel,
    OptimizeRequest,
    OptimizeStrategyRequest,
    PathStepResponse,
    SearchMetadataResponse,
    SearchSummaryResponse,
    StrategyResultResponse,
)
from src.core.mcts.search import SearchResult
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


def _map_action(model: CraftingActionModel) -> CraftingAction:
    """Convert a :class:`CraftingActionModel` to a domain :class:`CraftingAction`.

    Args:
        model: Validated Pydantic action model.

    Returns:
        Immutable domain entity.

    Raises:
        ValueError: If the action_type is not a recognised :class:`ActionType`.
    """
    try:
        action_type = ActionType(model.action_type)
    except ValueError as exc:
        raise ValueError(
            f"Unknown action_type '{model.action_type}'. "
            f"Expected one of {[e.value for e in ActionType]}."
        ) from exc

    outcomes = tuple(
        Outcome(
            probability=o.probability,
            transformation_type=o.transformation_type,
            modifier_pool=tuple(o.modifier_pool),
            specific_modifier=(
                ItemModifier(
                    modifier_id=o.specific_modifier_id,
                    tier=o.specific_modifier_tier or 1,
                    value=o.specific_modifier_value or 0.0,
                )
                if o.specific_modifier_id is not None
                else None
            ),
            min_mod_count=o.min_mod_count,
            max_mod_count=o.max_mod_count,
        )
        for o in model.effects.outcomes
    )
    effects = ActionEffects(
        deterministic=model.effects.deterministic,
        outcomes=outcomes,
    )
    prerequisites = ActionPrerequisites(
        required_state=model.prerequisites.required_state,
        blocked_by=frozenset(model.prerequisites.blocked_by),
    )
    return CraftingAction(
        action_id=model.action_id,
        action_type=action_type,
        name=model.name,
        cost=model.cost,
        effects=effects,
        prerequisites=prerequisites,
    )


def map_request_to_domain(
    request: OptimizeRequest | OptimizeStrategyRequest,
) -> tuple[ItemState, CraftingGoal, list[CraftingAction]]:
    """Translate an API request into the three domain objects needed by the use case.

    Args:
        request: Validated Pydantic request model.

    Returns:
        Tuple of (initial_state, crafting_goal, available_actions).

    Raises:
        ValueError: If any field contains an unrecognised enumeration value.
    """
    goal_model = request.goal

    try:
        goal_type = GoalType(goal_model.goal_type)
    except ValueError as exc:
        raise ValueError(
            f"Unknown goal_type '{goal_model.goal_type}'. "
            f"Expected one of {[e.value for e in GoalType]}."
        ) from exc

    base_item = BaseItem(
        item_type=goal_model.base_item.item_type,
        item_level=goal_model.base_item.item_level,
        influence=goal_model.base_item.influence,
    )
    initial_state = ItemState(
        base_item=base_item,
        modifiers=frozenset(),
    )
    target_modifiers = tuple(
        ModifierTarget(
            modifier_id=t.modifier_id,
            min_tier=t.min_tier,
            weight=t.weight,
        )
        for t in goal_model.target_modifiers
    )
    crafting_goal = CraftingGoal(
        goal_type=goal_type,
        target_modifiers=target_modifiers,
        max_steps=goal_model.constraints.max_steps,
        budget_limit=goal_model.constraints.budget_limit,
    )
    actions = [_map_action(a) for a in request.actions]
    return initial_state, crafting_goal, actions


def map_search_result_to_response(result: SearchResult) -> StrategyResultResponse:
    """Translate a domain :class:`SearchResult` to an API response model.

    Args:
        result: Domain search result from the MCTS optimiser.

    Returns:
        :class:`StrategyResultResponse` ready for JSON serialisation.
    """
    path = [
        PathStepResponse(
            step=step.step,
            action_id=step.action_id,
            action_name=step.action_name,
            estimated_cost=step.estimated_cost,
            expected_reward=step.expected_reward,
            confidence=step.confidence,
        )
        for step in result.best_path
    ]
    summary = SearchSummaryResponse(
        total_steps=result.summary.total_steps,
        total_cost=result.summary.total_cost,
        expected_final_reward=result.summary.expected_final_reward,
        success_probability=result.summary.success_probability,
        alternative_paths=result.summary.alternative_paths,
    )
    metadata = SearchMetadataResponse(
        iterations=result.metadata.iterations,
        computation_time=result.metadata.computation_time,
        tree_depth=result.metadata.tree_depth,
        nodes_explored=result.metadata.nodes_explored,
    )
    return StrategyResultResponse(
        success=result.success,
        strategy=result.strategy,
        best_path=path,
        summary=summary,
        metadata=metadata,
    )
