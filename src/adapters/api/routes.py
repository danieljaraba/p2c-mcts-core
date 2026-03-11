"""API route handlers for the crafting optimiser service.

All business logic is delegated to the :class:`~src.ports.optimizer_port.OptimizerPort`
injected via FastAPI dependency injection.  Route handlers are responsible only
for translating between Pydantic models and domain entities (Adapter pattern).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.adapters.api.dependencies import get_optimizer
from src.adapters.api.mappers import (
    map_request_to_domain,
    map_search_result_to_response,
)
from src.adapters.api.models import (
    OptimizeAllResponse,
    OptimizeRequest,
    OptimizeStrategyRequest,
    StrategyResultResponse,
)
from src.core.mdp.entities import CraftingStrategy
from src.ports.optimizer_port import OptimizerPort

router = APIRouter(prefix="/api/v1", tags=["crafting"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", summary="Health check")
def health_check() -> dict[str, str]:
    """Return a simple liveness indicator.

    Returns:
        JSON body ``{"status": "ok"}``.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# All strategies at once
# ---------------------------------------------------------------------------


@router.post(
    "/optimize",
    response_model=OptimizeAllResponse,
    summary="Optimise crafting for all three strategies",
    description=(
        "Runs MCTS for the **deterministic**, **cheapest**, and **balanced** "
        "strategies in a single request and returns all three crafting paths."
    ),
)
def optimize_all(
    request: OptimizeRequest,
    optimizer: OptimizerPort = Depends(get_optimizer),  # noqa: B008
) -> OptimizeAllResponse:
    """Return recommended crafting paths for all three optimisation strategies.

    Args:
        request: Crafting goal, reward system, and available actions.
        optimizer: Injected optimiser port implementation.

    Returns:
        :class:`~src.adapters.api.models.OptimizeAllResponse` containing three
        strategy results.

    Raises:
        HTTPException 422: If the request body is invalid.
        HTTPException 500: If the optimiser encounters an unexpected error.
    """
    try:
        initial_state, goal, actions = map_request_to_domain(request)
        results = optimizer.optimize_all_strategies(
            initial_state=initial_state,
            goal=goal,
            available_actions=actions,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimisation failed: {exc}",
        ) from exc

    return OptimizeAllResponse(
        deterministic=map_search_result_to_response(
            results[CraftingStrategy.DETERMINISTIC.value]
        ),
        cheapest=map_search_result_to_response(
            results[CraftingStrategy.CHEAPEST.value]
        ),
        balanced=map_search_result_to_response(
            results[CraftingStrategy.BALANCED.value]
        ),
    )


# ---------------------------------------------------------------------------
# Single-strategy endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/optimize/deterministic",
    response_model=StrategyResultResponse,
    summary="Most deterministic crafting path",
    description=(
        "Runs MCTS with the **deterministic** strategy that minimises variance "
        "and prefers reliable, predictable crafting steps."
    ),
)
def optimize_deterministic(
    request: OptimizeStrategyRequest,
    optimizer: OptimizerPort = Depends(get_optimizer),  # noqa: B008
) -> StrategyResultResponse:
    """Return the most deterministic crafting path.

    Args:
        request: Crafting request with optional iteration override.
        optimizer: Injected optimiser port implementation.

    Returns:
        Single :class:`~src.adapters.api.models.StrategyResultResponse`.
    """
    return _run_single_strategy(
        request, optimizer, CraftingStrategy.DETERMINISTIC
    )


@router.post(
    "/optimize/cheapest",
    response_model=StrategyResultResponse,
    summary="Cheapest crafting path",
    description=(
        "Runs MCTS with the **cheapest** strategy that minimises expected "
        "currency cost to reach the desired item configuration."
    ),
)
def optimize_cheapest(
    request: OptimizeStrategyRequest,
    optimizer: OptimizerPort = Depends(get_optimizer),  # noqa: B008
) -> StrategyResultResponse:
    """Return the cheapest crafting path.

    Args:
        request: Crafting request with optional iteration override.
        optimizer: Injected optimiser port implementation.

    Returns:
        Single :class:`~src.adapters.api.models.StrategyResultResponse`.
    """
    return _run_single_strategy(request, optimizer, CraftingStrategy.CHEAPEST)


@router.post(
    "/optimize/balanced",
    response_model=StrategyResultResponse,
    summary="Balanced crafting path",
    description=(
        "Runs MCTS with the **balanced** strategy that trades off goal "
        "progress, cost, and path reliability."
    ),
)
def optimize_balanced(
    request: OptimizeStrategyRequest,
    optimizer: OptimizerPort = Depends(get_optimizer),  # noqa: B008
) -> StrategyResultResponse:
    """Return the balanced crafting path.

    Args:
        request: Crafting request with optional iteration override.
        optimizer: Injected optimiser port implementation.

    Returns:
        Single :class:`~src.adapters.api.models.StrategyResultResponse`.
    """
    return _run_single_strategy(request, optimizer, CraftingStrategy.BALANCED)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_single_strategy(
    request: OptimizeStrategyRequest,
    optimizer: OptimizerPort,
    strategy: CraftingStrategy,
) -> StrategyResultResponse:
    """Execute the optimiser for one strategy and map the result.

    Args:
        request: Validated request model.
        optimizer: Optimiser port implementation.
        strategy: Target crafting strategy.

    Returns:
        Mapped response model.

    Raises:
        HTTPException 422: On domain validation error.
        HTTPException 500: On unexpected optimiser failure.
    """
    try:
        initial_state, goal, actions = map_request_to_domain(request)
        result = optimizer.optimize(
            initial_state=initial_state,
            goal=goal,
            available_actions=actions,
            strategy=strategy,
            iterations=request.iterations,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimisation failed: {exc}",
        ) from exc

    return map_search_result_to_response(result)
