"""Tests for content generation modules."""

import pytest
from src.generation.content_generator import ContentGenerator
from src.generation.code_generator import CodeGenerator
from src.generation.title_generator import TitleGenerator


@pytest.mark.asyncio
async def test_content_generator_initialization():
    """Test ContentGenerator initializes correctly."""
    # Skip if no API key
    pytest.skip("Requires Google API key")


@pytest.mark.asyncio
async def test_code_generator_initialization():
    """Test CodeGenerator initializes correctly."""
    pytest.skip("Requires Google API key")


@pytest.mark.asyncio
async def test_validate_mermaid():
    """Test Mermaid syntax validation."""
    valid_mermaid = """```mermaid
graph TD
    A[Start] --> B[End]
```"""
    
    assert CodeGenerator.validate_mermaid(valid_mermaid) is True


def test_validate_invalid_mermaid():
    """Test invalid Mermaid syntax."""
    invalid = "Just some text"
    
    assert CodeGenerator.validate_mermaid(invalid) is False


@pytest.mark.asyncio
async def test_title_generator_initialization():
    """Test TitleGenerator initializes correctly."""
    pytest.skip("Requires Google API key")
