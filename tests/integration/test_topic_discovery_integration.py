"""Integration tests for topic_discovery_node (requires API keys)."""

import os

import pytest

from src.agent.nodes import topic_discovery_node
from src.agent.state import Phase, BlogAgentState


@pytest.fixture
def real_state() -> BlogAgentState:
    """Real state for integration testing (no job_id to skip checkpointing)."""
    return {
        "title": "Semantic Caching for LLM Applications",
        "context": "Exploring GPTCache and Redis vector search for caching LLM responses.",
    }


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestTopicDiscoveryIntegration:
    """Integration tests that hit real APIs."""

    @pytest.mark.asyncio
    async def test_end_to_end_discovery(self, real_state):
        """Full discovery flow with real APIs."""
        result = await topic_discovery_node(real_state)

        # Should have generated queries
        assert "discovery_queries" in result
        assert len(result["discovery_queries"]) >= 3
        assert len(result["discovery_queries"]) <= 5

        # Should have topic context
        assert "topic_context" in result
        assert len(result["topic_context"]) > 0

        # Each result should have required fields
        for item in result["topic_context"]:
            assert "title" in item
            assert "url" in item
            assert "snippet" in item

        # Should advance to planning
        assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_queries_are_relevant(self, real_state):
        """Generated queries are relevant to the topic."""
        result = await topic_discovery_node(real_state)

        queries = result.get("discovery_queries", [])

        # At least one query should mention caching or LLM
        relevant_terms = ["caching", "cache", "llm", "semantic", "gptcache", "vector"]
        has_relevant = any(
            any(term in q.lower() for term in relevant_terms) for q in queries
        )
        assert has_relevant, f"Queries not relevant: {queries}"

    @pytest.mark.asyncio
    async def test_different_topic(self):
        """Test with a different topic."""
        state: BlogAgentState = {
            "title": "Kubernetes Pod Autoscaling",
            "context": "HPA vs VPA, custom metrics, best practices",
        }

        result = await topic_discovery_node(state)

        # Should succeed
        assert result["current_phase"] == Phase.PLANNING.value
        assert len(result.get("discovery_queries", [])) >= 3

        # Queries should be relevant to Kubernetes
        queries = result.get("discovery_queries", [])
        k8s_terms = ["kubernetes", "k8s", "pod", "autoscal", "hpa", "vpa"]
        has_relevant = any(
            any(term in q.lower() for term in k8s_terms) for q in queries
        )
        assert has_relevant, f"Queries not relevant to Kubernetes: {queries}"
