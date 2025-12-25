"""Unit tests for configuration helpers."""

import os
from unittest.mock import patch

import pytest

from src.agent.config import (
    LLM_MODEL_FULL,
    LLM_MODEL_LITE,
    LLM_TEMPERATURE_LOW,
    LLM_TEMPERATURE_MEDIUM,
    get_llm_config,
)


class TestGetLlmConfig:
    """Tests for get_llm_config function."""

    def test_returns_defaults_when_no_env_vars(self):
        """Returns default model and temperature when env vars not set."""
        model, temp = get_llm_config(
            "NONEXISTENT_MODEL", "NONEXISTENT_TEMP", LLM_MODEL_LITE, LLM_TEMPERATURE_LOW
        )
        assert model == LLM_MODEL_LITE
        assert temp == LLM_TEMPERATURE_LOW

    def test_uses_env_model_with_default_temp(self):
        """Uses env model when set, default temperature when not set."""
        with patch.dict(
            os.environ, {"TEST_MODEL": "gemini-2.0-flash-exp"}, clear=False
        ):
            model, temp = get_llm_config(
                "TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, LLM_TEMPERATURE_MEDIUM
            )
            assert model == "gemini-2.0-flash-exp"
            assert temp == LLM_TEMPERATURE_MEDIUM

    def test_uses_default_model_with_env_temp(self):
        """Uses default model when not set, env temperature when set."""
        with patch.dict(os.environ, {"TEST_TEMP": "0.5"}, clear=False):
            model, temp = get_llm_config(
                "TEST_MODEL", "TEST_TEMP", LLM_MODEL_FULL, LLM_TEMPERATURE_LOW
            )
            assert model == LLM_MODEL_FULL
            assert temp == 0.5

    def test_uses_both_env_vars(self):
        """Uses both env vars when both are set."""
        with patch.dict(
            os.environ,
            {"TEST_MODEL": "gemini-2.0-flash-exp", "TEST_TEMP": "0.9"},
            clear=False,
        ):
            model, temp = get_llm_config(
                "TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, LLM_TEMPERATURE_LOW
            )
            assert model == "gemini-2.0-flash-exp"
            assert temp == 0.9

    def test_temperature_conversion_to_float(self):
        """Temperature env var is correctly converted to float."""
        with patch.dict(os.environ, {"TEST_TEMP": "0.3333"}, clear=False):
            _, temp = get_llm_config("TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, 0.5)
            assert temp == pytest.approx(0.3333)

    def test_temperature_zero_is_valid(self):
        """Temperature of 0.0 is valid and not treated as falsy."""
        with patch.dict(os.environ, {"TEST_TEMP": "0.0"}, clear=False):
            _, temp = get_llm_config("TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, 0.7)
            assert temp == 0.0

    def test_temperature_one_is_valid(self):
        """Temperature of 1.0 is valid."""
        with patch.dict(os.environ, {"TEST_TEMP": "1.0"}, clear=False):
            _, temp = get_llm_config("TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, 0.5)
            assert temp == 1.0

    def test_env_var_names_are_case_sensitive(self):
        """Environment variable names are case-sensitive."""
        with patch.dict(
            os.environ, {"test_model": "gemini-2.0-flash-exp"}, clear=False
        ):
            # Should not find lowercase version
            model, temp = get_llm_config(
                "TEST_MODEL", "TEST_TEMP", LLM_MODEL_LITE, LLM_TEMPERATURE_LOW
            )
            assert model == LLM_MODEL_LITE  # Uses default
            assert temp == LLM_TEMPERATURE_LOW

    def test_actual_config_constants_exist(self):
        """Verify that the actual config constants are properly defined."""
        assert LLM_MODEL_LITE == "gemini-2.5-flash-lite"
        assert LLM_MODEL_FULL == "gemini-2.5-flash"
        assert LLM_TEMPERATURE_LOW == 0.3
        assert LLM_TEMPERATURE_MEDIUM == 0.7

    def test_realistic_topic_discovery_config(self):
        """Test realistic configuration for topic discovery node."""
        with patch.dict(
            os.environ,
            {
                "TOPIC_DISCOVERY_MODEL": "gemini-2.0-flash-exp",
                "TOPIC_DISCOVERY_TEMPERATURE": "0.8",
            },
            clear=False,
        ):
            model, temp = get_llm_config(
                "TOPIC_DISCOVERY_MODEL",
                "TOPIC_DISCOVERY_TEMPERATURE",
                LLM_MODEL_LITE,
                LLM_TEMPERATURE_MEDIUM,
            )
            assert model == "gemini-2.0-flash-exp"
            assert temp == 0.8

    def test_realistic_write_section_config(self):
        """Test realistic configuration for write section node."""
        with patch.dict(
            os.environ,
            {
                "WRITE_SECTION_MODEL": "gemini-2.5-flash",
                "WRITE_SECTION_TEMPERATURE": "0.7",
            },
            clear=False,
        ):
            model, temp = get_llm_config(
                "WRITE_SECTION_MODEL",
                "WRITE_SECTION_TEMPERATURE",
                LLM_MODEL_FULL,
                LLM_TEMPERATURE_MEDIUM,
            )
            assert model == "gemini-2.5-flash"
            assert temp == 0.7
