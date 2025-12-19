"""Application configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # API Settings
    api_title: str = "P2C MCTS Core"
    api_version: str = "0.1.0"
    api_description: str = "Monte Carlo Tree Search with MDP"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # CORS Settings
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # MCTS Settings
    default_simulations: int = 100
    default_exploration_weight: float = 1.41
    max_simulations: int = 10000


settings = Settings()
