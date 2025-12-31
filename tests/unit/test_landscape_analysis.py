"""Unit tests for content landscape analysis node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.nodes import content_landscape_analysis_node
from src.agent.state import Phase


@pytest.mark.asyncio
async def test_landscape_analysis_creates_default_strategy_when_no_topic_context():
    """Test that landscape analysis returns a default strategy when topic_context is None."""
    state = {
        "job_id": "test-job",
        "title": "Test Topic",
        "context": "Test context",
        "topic_context": None,  # ← Missing context (triggers skip)
        "metrics": {},  # metrics is a dict, not list
    }

    # Mock JobManager to avoid file I/O
    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = MagicMock()
        mock_job_manager.return_value = mock_instance

        result = await content_landscape_analysis_node(state)

    # Should return default strategy, not None
    assert result["content_strategy"] is not None
    assert isinstance(result["content_strategy"], dict)
    assert result["content_strategy"]["unique_angle"] != ""
    assert result["content_strategy"]["target_persona"] == "senior_engineer"
    assert result["content_strategy"]["reader_problem"] == "Understanding and implementing Test Topic"
    assert len(result["content_strategy"]["gaps_to_fill"]) == 1
    assert result["content_strategy"]["analyzed_articles"] == []
    assert result["current_phase"] == Phase.PLANNING.value


@pytest.mark.asyncio
async def test_landscape_analysis_creates_default_strategy_when_empty_topic_context():
    """Test that landscape analysis returns a default strategy when topic_context is empty list."""
    state = {
        "job_id": "test-job",
        "title": "Semantic Caching",
        "context": "Redis implementation",
        "topic_context": [],  # ← Empty list (triggers skip)
        "metrics": {},
    }

    # Mock JobManager to avoid file I/O
    with patch("src.agent.nodes.JobManager") as mock_job_manager:
        mock_instance = MagicMock()
        mock_instance.get_job_dir.return_value = MagicMock()
        mock_job_manager.return_value = mock_instance

        result = await content_landscape_analysis_node(state)

    # Should return default strategy
    assert result["content_strategy"] is not None
    assert result["content_strategy"]["reader_problem"] == "Understanding and implementing Semantic Caching"
    assert result["current_phase"] == Phase.PLANNING.value


@pytest.mark.asyncio
async def test_landscape_analysis_no_job_id():
    """Test that landscape analysis works without job_id (no checkpoint saving)."""
    state = {
        "job_id": "",  # Empty job_id
        "title": "Test Topic",
        "context": "Test context",
        "topic_context": None,
        "metrics": {},
    }

    result = await content_landscape_analysis_node(state)

    # Should still return default strategy
    assert result["content_strategy"] is not None
    assert result["current_phase"] == Phase.PLANNING.value


@pytest.mark.asyncio
async def test_landscape_analysis_handles_runtime_error():
    """Test that RuntimeError returns default strategy instead of failing."""
    state = {
        "job_id": "test-job",
        "title": "Test Topic",
        "context": "Test context",
        "topic_context": [{"title": "Article", "url": "http://example.com", "snippet": "test"}],
        "metrics": {},
    }

    # Mock JobManager and file operations
    with patch("src.agent.nodes.JobManager") as mock_jm:
        with patch("builtins.open", MagicMock()):
            mock_instance = MagicMock()
            mock_instance.get_job_dir.return_value = MagicMock()
            mock_jm.return_value = mock_instance

            # Mock KeyManager.from_env to raise RuntimeError
            with patch("src.agent.nodes.KeyManager.from_env", side_effect=RuntimeError("API quota exhausted")):
                result = await content_landscape_analysis_node(state)

    # Should return default strategy, NOT Phase.FAILED
    assert result["current_phase"] == Phase.PLANNING.value
    assert result["content_strategy"] is not None
    assert "Content analysis failed" in result["content_strategy"]["existing_content_summary"]


@pytest.mark.asyncio
async def test_landscape_analysis_handles_generic_exception():
    """Test that generic Exception returns default strategy instead of failing."""
    state = {
        "job_id": "test-job",
        "title": "Test Topic",
        "context": "Test context",
        "topic_context": [{"title": "Article", "url": "http://example.com", "snippet": "test"}],
        "metrics": {},
    }

    # Mock JobManager and file operations
    with patch("src.agent.nodes.JobManager") as mock_jm:
        with patch("builtins.open", MagicMock()):
            mock_instance = MagicMock()
            mock_instance.get_job_dir.return_value = MagicMock()
            mock_jm.return_value = mock_instance

            # Mock KeyManager.from_env to raise generic Exception
            with patch("src.agent.nodes.KeyManager.from_env", side_effect=Exception("Network error")):
                result = await content_landscape_analysis_node(state)

    # Should return default strategy, NOT Phase.FAILED
    assert result["current_phase"] == Phase.PLANNING.value
    assert result["content_strategy"] is not None
    assert "Content analysis failed with error" in result["content_strategy"]["existing_content_summary"]
