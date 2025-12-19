"""Unit tests for topic_discovery_node."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    topic_discovery_node,
    _generate_discovery_queries,
    _execute_searches,
)
from src.agent.state import Phase, BlogAgentState, DiscoveryQueries


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_state() -> BlogAgentState:
    """Sample state for testing."""
    return {
        "job_id": "test-job",
        "title": "Semantic Caching for LLM Applications",
        "context": "Saw GPTCache on Twitter. Redis vector search for caching.",
    }


@pytest.fixture
def mock_discovery_queries():
    """Mock DiscoveryQueries response."""
    return DiscoveryQueries(
        queries=[
            "semantic caching LLM 2024",
            "GPTCache how it works",
            "vector similarity caching AI",
        ]
    )


@pytest.fixture
def mock_search_results():
    """Mock search results."""
    return [
        {"title": "Article 1", "url": "https://example1.com", "snippet": "Snippet 1"},
        {"title": "Article 2", "url": "https://example2.com", "snippet": "Snippet 2"},
    ]


# =============================================================================
# Topic Discovery Node Tests
# =============================================================================


class TestTopicDiscoveryNode:
    """Tests for topic_discovery_node function."""

    @pytest.mark.asyncio
    async def test_returns_queries_and_context(
        self,
        sample_state,
        mock_discovery_queries,
        mock_search_results,
    ):
        """Node returns discovery_queries and topic_context."""
        with patch("src.agent.nodes._generate_discovery_queries") as mock_gen:
            with patch("src.agent.nodes._execute_searches") as mock_search:
                with patch("src.agent.nodes.KeyManager") as mock_km:
                    with patch("src.agent.nodes.JobManager") as mock_jm:
                        mock_gen.return_value = mock_discovery_queries
                        mock_search.return_value = mock_search_results
                        mock_km.from_env.return_value = MagicMock()
                        mock_jm.return_value = MagicMock()

                        result = await topic_discovery_node(sample_state)

                        assert "discovery_queries" in result
                        assert "topic_context" in result
                        assert len(result["discovery_queries"]) == 3
                        assert len(result["topic_context"]) == 2

    @pytest.mark.asyncio
    async def test_advances_to_planning_phase(
        self,
        sample_state,
        mock_discovery_queries,
        mock_search_results,
    ):
        """Phase advances to PLANNING on success."""
        with patch("src.agent.nodes._generate_discovery_queries") as mock_gen:
            with patch("src.agent.nodes._execute_searches") as mock_search:
                with patch("src.agent.nodes.KeyManager") as mock_km:
                    with patch("src.agent.nodes.JobManager") as mock_jm:
                        mock_gen.return_value = mock_discovery_queries
                        mock_search.return_value = mock_search_results
                        mock_km.from_env.return_value = MagicMock()
                        mock_jm.return_value = MagicMock()

                        result = await topic_discovery_node(sample_state)

                        assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_fails_without_title(self):
        """Node fails if title is missing."""
        state: BlogAgentState = {"context": "Some context"}

        result = await topic_discovery_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_fails_with_empty_title(self):
        """Node fails if title is empty string."""
        state: BlogAgentState = {"title": "", "context": "Some context"}

        result = await topic_discovery_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_handles_query_generation_failure(self, sample_state):
        """Node handles LLM failure gracefully."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes._generate_discovery_queries") as mock_gen:
                mock_km.from_env.return_value = MagicMock()
                mock_gen.side_effect = RuntimeError("All API keys exhausted")

                result = await topic_discovery_node(sample_state)

                assert result["current_phase"] == Phase.FAILED.value
                assert "exhausted" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_handles_unexpected_error(self, sample_state):
        """Node handles unexpected errors gracefully."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            mock_km.from_env.side_effect = ValueError("No API keys found")

            result = await topic_discovery_node(sample_state)

            assert result["current_phase"] == Phase.FAILED.value
            assert "error" in result.get("error_message", "").lower()


# =============================================================================
# Generate Discovery Queries Tests
# =============================================================================


class TestGenerateDiscoveryQueries:
    """Tests for _generate_discovery_queries helper."""

    @pytest.mark.asyncio
    async def test_returns_discovery_queries_model(self):
        """Returns DiscoveryQueries Pydantic model."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = DiscoveryQueries(
                queries=["q1", "q2", "q3"]
            )
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_discovery_queries(
                title="Test",
                context="Context",
                key_manager=mock_key_manager,
            )

            assert isinstance(result, DiscoveryQueries)
            assert len(result.queries) == 3

    @pytest.mark.asyncio
    async def test_records_usage_on_success(self):
        """Records API usage after successful call."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = DiscoveryQueries(
                queries=["q1", "q2", "q3"]
            )
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            await _generate_discovery_queries(
                title="Test",
                context="Context",
                key_manager=mock_key_manager,
            )

            mock_key_manager.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotates_key_on_rate_limit(self):
        """Rotates to next key on 429 error."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = "key2"

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Resource Exhausted")
            return DiscoveryQueries(queries=["q1", "q2", "q3"])

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = invoke_side_effect
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_discovery_queries(
                title="Test",
                context="Context",
                key_manager=mock_key_manager,
            )

            mock_key_manager.mark_rate_limited.assert_called_once_with("key1")
            assert result.queries == ["q1", "q2", "q3"]

    @pytest.mark.asyncio
    async def test_raises_when_all_keys_exhausted(self):
        """Raises RuntimeError when all keys are exhausted."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = None  # No more keys

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = Exception("429 quota exceeded")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            with pytest.raises(RuntimeError, match="exhausted"):
                await _generate_discovery_queries(
                    title="Test",
                    context="Context",
                    key_manager=mock_key_manager,
                )


# =============================================================================
# Execute Searches Tests
# =============================================================================


class TestExecuteSearches:
    """Tests for _execute_searches helper."""

    @pytest.mark.asyncio
    async def test_returns_deduplicated_results(self):
        """Duplicate URLs are removed."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            # Return duplicate URLs across queries
            mock_search.side_effect = [
                [
                    {"title": "A", "url": "https://a.com", "snippet": "S1"},
                    {"title": "B", "url": "https://b.com", "snippet": "S2"},
                ],
                [
                    {"title": "A2", "url": "https://a.com", "snippet": "S1 dup"},  # Duplicate
                    {"title": "C", "url": "https://c.com", "snippet": "S3"},
                ],
            ]

            result = await _execute_searches(
                queries=["query1", "query2"],
                max_results_per_query=5,
            )

            urls = [r["url"] for r in result]
            assert len(urls) == len(set(urls))  # All unique
            assert len(result) == 3  # A, B, C (not A2)

    @pytest.mark.asyncio
    async def test_respects_max_total_results(self):
        """Limits total results to max_total_results."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            mock_search.return_value = [
                {"title": f"T{i}", "url": f"https://site{i}.com", "snippet": f"S{i}"}
                for i in range(10)
            ]

            result = await _execute_searches(
                queries=["query1", "query2", "query3"],
                max_results_per_query=10,
                max_total_results=5,
            )

            assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_handles_search_failures(self):
        """Continues if individual search fails."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            mock_search.side_effect = [
                Exception("Search failed"),  # First query fails
                [{"title": "B", "url": "https://b.com", "snippet": "S"}],  # Second succeeds
            ]

            result = await _execute_searches(
                queries=["query1", "query2"],
            )

            # Should still have results from successful query
            assert len(result) == 1
            assert result[0]["url"] == "https://b.com"

    @pytest.mark.asyncio
    async def test_skips_empty_urls(self):
        """Skips results with empty URLs."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            mock_search.return_value = [
                {"title": "A", "url": "", "snippet": "S1"},  # Empty URL
                {"title": "B", "url": "https://b.com", "snippet": "S2"},
            ]

            result = await _execute_searches(queries=["query1"])

            assert len(result) == 1
            assert result[0]["url"] == "https://b.com"

    @pytest.mark.asyncio
    async def test_returns_empty_on_all_failures(self):
        """Returns empty list if all searches fail."""
        with patch("src.agent.nodes.search_duckduckgo") as mock_search:
            mock_search.side_effect = Exception("All searches failed")

            result = await _execute_searches(
                queries=["query1", "query2"],
            )

            assert result == []


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpointing:
    """Tests for checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_saves_checkpoint_on_success(
        self,
        sample_state,
        mock_discovery_queries,
        mock_search_results,
    ):
        """Checkpoint is saved after successful discovery."""
        with patch("src.agent.nodes._generate_discovery_queries") as mock_gen:
            with patch("src.agent.nodes._execute_searches") as mock_search:
                with patch("src.agent.nodes.KeyManager") as mock_km:
                    with patch("src.agent.nodes.JobManager") as mock_jm_class:
                        mock_gen.return_value = mock_discovery_queries
                        mock_search.return_value = mock_search_results
                        mock_km.from_env.return_value = MagicMock()

                        mock_jm = MagicMock()
                        mock_jm_class.return_value = mock_jm

                        await topic_discovery_node(sample_state)

                        mock_jm.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_without_job_id(
        self,
        mock_discovery_queries,
        mock_search_results,
    ):
        """No checkpoint if job_id not in state."""
        state: BlogAgentState = {
            "title": "Test",
            "context": "Context",
            # No job_id
        }

        with patch("src.agent.nodes._generate_discovery_queries") as mock_gen:
            with patch("src.agent.nodes._execute_searches") as mock_search:
                with patch("src.agent.nodes.KeyManager") as mock_km:
                    with patch("src.agent.nodes.JobManager") as mock_jm_class:
                        mock_gen.return_value = mock_discovery_queries
                        mock_search.return_value = mock_search_results
                        mock_km.from_env.return_value = MagicMock()

                        await topic_discovery_node(state)

                        # JobManager should not be instantiated
                        mock_jm_class.assert_not_called()
