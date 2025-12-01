"""
Settings module for environment-aware configuration.

Manages all configuration including API keys, file paths, and runtime settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # LLM API Keys
    google_api_key: str = ""
    google_api_key_1: str = ""
    google_api_key_2: str = ""
    google_api_paid_key: str = ""  # Paid key for image generation
    openai_api_key: str = ""
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_api_version: str = "2025-01-01-preview"
    azure_deployment_name: str = "gpt-5-chat"
    
    # Image Generation (Banana/Nano)
    banana_api_key: str = ""
    banana_model_key: str = ""
    
    # Environment Configuration
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    # Rate Limiting & Retries
    max_retries: int = 3
    retry_delay: int = 2
    scraping_rate_limit: int = 10
    
    # Content Generation Settings
    max_refinement_iterations: int = 3
    quality_threshold: int = 8
    max_outline_iterations: int = 3
    outline_quality_threshold: float = 9.5
    outline_reviewer_model: str = "gemini-2.5-flash"
    
    # Section Content Review Settings
    max_section_review_iterations: int = 3
    section_quality_threshold: float = 9.0
    section_reviewer_model: str = "gemini-2.5-flash"
    
    # Diagram Generation Settings
    diagram_identifier_model: str = "gemini-2.5-flash"
    diagram_generator_model: str = "gemini-2.5-flash"
    diagram_reviewer_model: str = "gemini-2.5-flash"
    max_diagram_generation_attempts: int = 3
    diagram_quality_threshold: float = 9.0
    
    # Blog Image Generation Settings
    blog_image_description_model: str = "gemini-2.5-flash"
    
    # Image Embedder Settings
    image_embedder_model: str = "gemini-2.5-flash"
    
    # Blog Reviewer Settings
    blog_reviewer_max_iterations: int = 5
    blog_reviewer_score_threshold: float = 9.0
    
    # File Paths
    input_dir: Path = Path("inputs")
    output_dir: Path = Path("outputs")
    
    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self.input_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "drafts").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "final").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "images").mkdir(exist_ok=True, parents=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()
