"""Main FastAPI application."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.infrastructure.adapters.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    print("Starting MCTS Core microservice...")
    yield
    # Shutdown
    print("Shutting down MCTS Core microservice...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="P2C MCTS Core",
        description="Monte Carlo Tree Search algorithm with MDP using Hexagonal Architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware - configure allowed origins via environment variables
    # In production, set specific allowed origins instead of using wildcard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Include routers
    app.include_router(router, prefix="/api/v1", tags=["MCTS"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        # Log the full error internally (in production, use proper logging)
        print(f"Internal error: {exc}")
        # Return generic error message to client
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred"},
        )

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
