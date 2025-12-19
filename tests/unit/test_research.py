"""Unit tests for research_node."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    research_node,
    _research_section,
    _hash_url,
)
from src.agent.state import Phase, BlogAgentState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_plan():
    """Sample plan with sections and search queries."""
    return {
        "blog_title": "Semantic Caching for LLM Applications",
        "target_words": 1500,
        "sections": [
            {
                "id": "problem",
                "title": "The LLM Cost Problem",
                "role": "problem",
                "search_queries": ["LLM API costs", "repeated queries"],
                "optional": False,
            },
            {
                "id": "why",
                "title": "Why Semantic Caching Works",
                "role": "why",
                "search_queries": ["semantic similarity caching"],
                "optional": False,
            },
            {
                "id": "conclusion",
                "title": "Key Takeaways",
                "role": "conclusion",
                "search_queries": [],  # No queries for conclusion
                "optional": False,
            },
        ],
    }


@pytest.fixture
def sample_state(sample_plan) -> BlogAgentState:
    """Sample state for testing."""
    return {
        "job_id": "test-job",
        "title": "Semantic Caching for LLM Applications",
        "plan": sample_plan,
    }


@pytest.fixture
def mock_search_results():
    """Mock DuckDuckGo search results."""
    return [
        {"title": "Article 1", "url": "https://example1.com/article", "snippet": "Snippet 1"},
        {"title": "Article 2", "url": "https://example2.com/article", "snippet": "Snippet 2"},
        {"title": "Article 3", "url": "https://example3.com/article", "snippet": "Snippet 3"},
    ]


@pytest.fixture
def mock_fetch_result():
    """Mock successful fetch result."""
    return {
        "url": "https://example.com",
        "title": "Example Article",
        "content": "This is the article content about LLM caching...",
        "success": True,
        "error": None,
        "tokens_estimate": 100,
    }


# =============================================================================
# Hash URL Tests
# =============================================================================


class TestHashUrl:
    """Tests for _hash_url helper."""

    def test_returns_string(self):
        """Returns a string hash."""
        result = _hash_url("https://example.com")
        assert isinstance(result, str)

    def test_consistent_hash(self):
        """Same URL produces same hash."""
        url = "https://example.com/article"
        hash1 = _hash_url(url)
        hash2 = _hash_url(url)
        assert hash1 == hash2

    def test_different_urls_different_hashes(self):
        """Different URLs produce different hashes."""
        hash1 = _hash_url("https://example1.com")
        hash2 = _hash_url("https://example2.com")
        assert hash1 != hash2

    def test_hash_length(self):
        """Hash is truncated to 12 characters."""
        result = _hash_url("https://example.com")
        assert len(result) == 12


# =============================================================================
# Research Section Tests
# =============================================================================


class TestResearchSection:
    """Tests for _research_section helper."""

    @pytest.mark.asyncio
    async def test_returns_sources_and_cache(self, mock_search_results, mock_fetch_result):
        """Returns tuple of sources and cache updates."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                mock_search.return_value = mock_search_results
                mock_fetch.return_value = mock_fetch_result

                sources, cache = await _research_section(
                    section_id="test",
                    search_queries=["test query"],
                    existing_cache={},
                )

                assert isinstance(sources, list)
                assert isinstance(cache, dict)
                assert len(sources) > 0

    @pytest.mark.asyncio
    async def test_deduplicates_urls(self, mock_fetch_result):
        """Duplicate URLs are not fetched twice."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                # Return same URL in multiple queries
                mock_search.return_value = [
                    {"title": "A", "url": "https://same.com", "snippet": "S"},
                ]
                mock_fetch.return_value = mock_fetch_result

                sources, cache = await _research_section(
                    section_id="test",
                    search_queries=["query1", "query2"],  # Two queries
                    existing_cache={},
                )

                # Should only fetch once despite two queries returning same URL
                assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_existing_cache(self, mock_fetch_result):
        """Uses cached content instead of re-fetching."""
        url = "https://cached.com"
        url_hash = _hash_url(url)
        existing_cache = {
            url_hash: {
                "url": url,
                "title": "Cached Title",
                "content": "Cached content",
                "tokens_estimate": 50,
            }
        }

        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                mock_search.return_value = [
                    {"title": "A", "url": url, "snippet": "S"},
                ]
                mock_fetch.return_value = mock_fetch_result

                sources, cache = await _research_section(
                    section_id="test",
                    search_queries=["query"],
                    existing_cache=existing_cache,
                )

                # Should not fetch - already in cache
                mock_fetch.assert_not_called()
                # Should still include the cached source
                assert len(sources) == 1
                assert sources[0]["content"] == "Cached content"

    @pytest.mark.asyncio
    async def test_handles_fetch_failures(self, mock_search_results):
        """Continues when individual fetches fail."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                mock_search.return_value = mock_search_results
                # First fetch fails, rest succeed
                mock_fetch.side_effect = [
                    {"success": False, "error": "Timeout"},
                    {"success": True, "content": "Content 2", "title": "T2", "tokens_estimate": 50},
                    {"success": True, "content": "Content 3", "title": "T3", "tokens_estimate": 50},
                ]

                sources, cache = await _research_section(
                    section_id="test",
                    search_queries=["query"],
                    existing_cache={},
                )

                # Should have 2 successful sources
                assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_limits_sources_per_section(self, mock_fetch_result):
        """Stops after collecting enough sources."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                # Return many results
                mock_search.return_value = [
                    {"title": f"A{i}", "url": f"https://site{i}.com", "snippet": f"S{i}"}
                    for i in range(10)
                ]
                mock_fetch.return_value = mock_fetch_result

                sources, cache = await _research_section(
                    section_id="test",
                    search_queries=["q1", "q2", "q3"],
                    existing_cache={},
                    max_urls_per_query=3,
                )

                # Should stop at 6 sources max
                assert len(sources) <= 6


# =============================================================================
# Research Node Tests
# =============================================================================


class TestResearchNode:
    """Tests for research_node function."""

    @pytest.mark.asyncio
    async def test_returns_cache_and_advances_phase(self, sample_state):
        """Node returns research_cache and advances to VALIDATING_SOURCES."""
        with patch("src.agent.nodes._research_section") as mock_research:
            with patch("src.agent.nodes.JobManager") as mock_jm:
                mock_research.return_value = (
                    [{"url": "https://example.com", "content": "Test"}],
                    {"hash123": {"url": "https://example.com", "content": "Test"}},
                )
                mock_jm.return_value = MagicMock()

                result = await research_node(sample_state)

                assert "research_cache" in result
                assert result["current_phase"] == Phase.VALIDATING_SOURCES.value

    @pytest.mark.asyncio
    async def test_fails_without_sections(self):
        """Node fails if no sections in plan."""
        state: BlogAgentState = {
            "plan": {"sections": []},
        }

        result = await research_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "No sections" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_fails_without_plan(self):
        """Node fails if no plan in state."""
        state: BlogAgentState = {}

        result = await research_node(state)

        assert result["current_phase"] == Phase.FAILED.value

    @pytest.mark.asyncio
    async def test_skips_sections_without_queries(self, sample_state):
        """Skips sections that have no search queries."""
        with patch("src.agent.nodes._research_section") as mock_research:
            with patch("src.agent.nodes.JobManager") as mock_jm:
                mock_research.return_value = ([], {})
                mock_jm.return_value = MagicMock()

                result = await research_node(sample_state)

                # Should call research for 2 sections (problem, why), not conclusion
                assert mock_research.call_count == 2

    @pytest.mark.asyncio
    async def test_processes_all_sections_including_optional(self):
        """Researches both required and optional sections."""
        state: BlogAgentState = {
            "plan": {
                "sections": [
                    {"id": "required", "search_queries": ["q1"], "optional": False},
                    {"id": "optional", "search_queries": ["q2"], "optional": True},
                ]
            }
        }

        with patch("src.agent.nodes._research_section") as mock_research:
            mock_research.return_value = ([], {})

            await research_node(state)

            # Should research both sections
            assert mock_research.call_count == 2

    @pytest.mark.asyncio
    async def test_saves_checkpoint_on_success(self, sample_state):
        """Checkpoint is saved after successful research."""
        with patch("src.agent.nodes._research_section") as mock_research:
            with patch("src.agent.nodes.JobManager") as mock_jm_class:
                mock_research.return_value = ([], {})
                mock_jm = MagicMock()
                mock_jm_class.return_value = mock_jm

                await research_node(sample_state)

                mock_jm.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_without_job_id(self, sample_plan):
        """No checkpoint if job_id not in state."""
        state: BlogAgentState = {
            "plan": sample_plan,
            # No job_id
        }

        with patch("src.agent.nodes._research_section") as mock_research:
            with patch("src.agent.nodes.JobManager") as mock_jm_class:
                mock_research.return_value = ([], {})

                await research_node(state)

                mock_jm_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_accumulates_cache_across_sections(self, sample_state):
        """Research cache accumulates results from all sections."""
        call_count = 0

        def mock_research_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (
                [],
                {f"hash{call_count}": {"url": f"https://site{call_count}.com", "content": "C"}},
            )

        with patch("src.agent.nodes._research_section") as mock_research:
            with patch("src.agent.nodes.JobManager") as mock_jm:
                mock_research.side_effect = mock_research_side_effect
                mock_jm.return_value = MagicMock()

                result = await research_node(sample_state)

                # Should have entries from both researched sections
                assert len(result["research_cache"]) == 2
