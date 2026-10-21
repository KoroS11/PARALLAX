"""Application settings for PARALLAX API serving plane."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from parallax.contracts.api import ScanMode


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables with PARALLAX_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="PARALLAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="API host address")
    port: int = Field(default=8000, description="API port")
    log_level: str = Field(default="info", description="Logging level")
    environment: str = Field(default="production", description="Runtime environment")
    api_version: str = Field(default="0.1.0", description="Service API version")

    # Protection & Limits
    max_input_size: int = Field(
        default=32 * 1024,
        description="Maximum request payload size in bytes (default: 32 KB)",
    )
    max_concurrency: int = Field(
        default=50,
        description="Bulkhead limiter maximum concurrent in-flight requests",
    )

    # Orchestrator & Detector settings
    default_timeout: float = Field(
        default=2.0,
        description="Default per-detector timeout in seconds",
    )
    default_scan_mode: ScanMode = Field(
        default=ScanMode.FAST,
        description="Default scan mode (fast, deep, bench-seq)",
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable LRU prompt cache",
    )
    cache_capacity: int = Field(
        default=1000,
        description="Maximum capacity of LRU cache",
    )

    # Victim LLM forwarding (Ollama)
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL for victim LLM",
    )
    default_victim_model: str = Field(
        default="gemma3:4b",
        description="Default victim model identifier",
    )
    victim_timeout: float = Field(
        default=15.0,
        description="Timeout for victim model generation in seconds",
    )
    allow_flagged: bool = Field(
        default=False,
        description="Whether to forward prompts with FLAG decision to victim LLM",
    )

    # Storage settings
    database_url: str = Field(
        default="sqlite:///./parallax.db",
        description="Database connection URL",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
