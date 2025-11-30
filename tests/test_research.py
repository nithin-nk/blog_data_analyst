"""Tests for research modules."""

import pytest
from src.research.search_agent import SearchAgent
from src.research.content_extractor import ContentExtractor


@pytest.mark.asyncio
async def test_search_agent_initialization():
    """Test SearchAgent initializes correctly."""
    agent = SearchAgent()
    assert agent is not None
    assert agent.max_results == 10


@pytest.mark.asyncio
async def test_generate_search_queries():
    """Test search query generation."""
    agent = SearchAgent()
    queries = agent.generate_search_queries(
        "What is Python?",
        "Programming Languages"
    )
    
    assert len(queries) > 0
    assert any("Python" in q for q in queries)


@pytest.mark.asyncio
async def test_content_extractor_initialization():
    """Test ContentExtractor initializes correctly."""
    extractor = ContentExtractor()
    assert extractor is not None


@pytest.mark.asyncio
async def test_clean_text():
    """Test text cleaning."""
    dirty_text = "  This   is   messy    text  "
    clean = ContentExtractor.clean_text(dirty_text)
    
    assert clean == "This is messy text"
