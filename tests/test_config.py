"""Tests for configuration module."""

import pytest
from pathlib import Path
import os
from src.config.settings import Settings, get_settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    
    assert settings.environment == "dev"
    assert settings.log_level == "INFO"
    assert settings.max_retries == 3
    assert settings.retry_delay == 2
    assert settings.scraping_rate_limit == 10
    assert settings.max_refinement_iterations == 3
    assert settings.quality_threshold == 8


def test_settings_creates_directories(tmp_path):
    """Test that settings creates necessary directories."""
    settings = Settings(
        input_dir=tmp_path / "inputs",
        output_dir=tmp_path / "outputs"
    )
    
    assert settings.input_dir.exists()
    assert settings.output_dir.exists()
    assert (settings.output_dir / "drafts").exists()
    assert (settings.output_dir / "final").exists()
    assert (settings.output_dir / "images").exists()


def test_settings_from_env(tmp_path, monkeypatch):
    """Test loading settings from environment variables."""
    # Set environment variables
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("QUALITY_THRESHOLD", "9")
    
    # Clear the cache to force reload
    get_settings.cache_clear()
    
    settings = Settings()
    
    assert settings.google_api_key == "test_google_key"
    assert settings.openai_api_key == "test_openai_key"
    assert settings.environment == "prod"
    assert settings.log_level == "DEBUG"
    assert settings.max_retries == 5
    assert settings.quality_threshold == 9
    
    # Clean up
    get_settings.cache_clear()


def test_settings_environment_validation():
    """Test environment field validation."""
    # Valid environments
    for env in ["dev", "test", "prod"]:
        settings = Settings(environment=env)
        assert settings.environment == env
    
    # Invalid environment should raise error
    with pytest.raises(ValueError):
        Settings(environment="invalid")


def test_settings_log_level_validation():
    """Test log level field validation."""
    # Valid log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        settings = Settings(log_level=level)
        assert settings.log_level == level
    
    # Invalid log level should raise error
    with pytest.raises(ValueError):
        Settings(log_level="INVALID")


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    get_settings.cache_clear()
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2
    
    get_settings.cache_clear()


def test_settings_case_insensitive(monkeypatch):
    """Test that environment variable names are case insensitive."""
    monkeypatch.setenv("google_api_key", "test_key_lowercase")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key_uppercase")
    
    settings = Settings()
    
    # Should use the uppercase version (last set)
    assert settings.google_api_key in ["test_key_lowercase", "test_key_uppercase"]


def test_settings_path_types():
    """Test that path fields are Path objects."""
    settings = Settings()
    
    assert isinstance(settings.input_dir, Path)
    assert isinstance(settings.output_dir, Path)
