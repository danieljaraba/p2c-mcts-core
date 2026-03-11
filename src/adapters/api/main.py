"""FastAPI application factory for the Path of Exile 2 MCTS crafting service.

The application is assembled here (composition root).  Infrastructure
components are wired into the domain via dependency injection so the domain
core remains free of framework imports.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Any startup/shutdown logic (e.g. warming up caches, opening DB pools)
    should be placed here.  Currently a no-op placeholder.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup; cleanup happens after the yield.
    """
    # Startup: nothing to initialise yet.
    yield
    # Shutdown: nothing to clean up yet.


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured :class:`fastapi.FastAPI` application ready to serve.
    """
    application = FastAPI(
        title="p2c-mcts-core",
        description=(
            "Path of Exile 2 MCTS Crafting Core Service — "
            "optimises item crafting using Monte Carlo Tree Search over a "
            "Markov Decision Process."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    return application


app = create_app()
