"""Integration tests for content_landscape_analysis_node (requires API keys)."""

import os

import pytest

from src.agent.nodes import content_landscape_analysis_node
from src.agent.state import Phase, BlogAgentState


@pytest.fixture
def real_state_with_topic_context() -> BlogAgentState:
    """Real state with topic context for integration testing."""
    return {
        "title": "Semantic Caching for LLM Applications",
        "context": "Exploring GPTCache and Redis vector search for caching LLM responses.",
        "topic_context": [
            {
                "title": "GPTCache: A Library for Creating Semantic Cache for LLM Queries",
                "url": "https://gptcache.io/",
                "snippet": "GPTCache is an open source semantic cache for LLM APIs.",
            },
            {
                "title": "Redis Vector Similarity Search",
                "url": "https://redis.io/docs/stack/search/reference/vectors/",
                "snippet": "Redis Stack provides vector similarity search capabilities.",
            },
            {
                "title": "Reducing LLM API Costs with Semantic Caching",
                "url": "https://towardsdatascience.com/semantic-caching-llm",
                "snippet": "How to use semantic caching to reduce your LLM API costs.",
            },
            {
                "title": "OpenAI API Optimization Guide",
                "url": "https://platform.openai.com/docs/guides/optimization",
                "snippet": "Best practices for optimizing OpenAI API usage.",
            },
            {
                "title": "LangChain Caching Documentation",
                "url": "https://python.langchain.com/docs/modules/memory/caching",
                "snippet": "LangChain provides built-in caching mechanisms.",
            },
        ],
    }


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestContentLandscapeAnalysisIntegration:
    """Integration tests that hit real APIs."""

    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self, real_state_with_topic_context):
        """Full content landscape analysis with real APIs."""
        result = await content_landscape_analysis_node(real_state_with_topic_context)

        # Should have content strategy
        assert "content_strategy" in result
        strategy = result["content_strategy"]
        assert strategy is not None

        # Strategy should have required fields
        assert "unique_angle" in strategy
        assert "target_persona" in strategy
        assert "reader_problem" in strategy
        assert "gaps_to_fill" in strategy
        assert "differentiation_requirements" in strategy

        # Should advance to planning
        assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_unique_angle_is_meaningful(self, real_state_with_topic_context):
        """Generated unique angle is relevant and specific."""
        result = await content_landscape_analysis_node(real_state_with_topic_context)

        strategy = result.get("content_strategy")
        assert strategy is not None

        unique_angle = strategy.get("unique_angle", "")

        # Should be a non-empty meaningful string
        assert len(unique_angle) > 10
        # Should be specific (not generic)
        generic_phrases = ["general overview", "basic introduction"]
        for phrase in generic_phrases:
            assert phrase not in unique_angle.lower(), f"Unique angle too generic: {unique_angle}"

    @pytest.mark.asyncio
    async def test_gaps_are_identified(self, real_state_with_topic_context):
        """Content gaps are identified from analyzed articles."""
        result = await content_landscape_analysis_node(real_state_with_topic_context)

        strategy = result.get("content_strategy")
        assert strategy is not None

        gaps = strategy.get("gaps_to_fill", [])

        # Should identify at least some gaps (unless content is already comprehensive)
        # Note: Could be empty if analyzed articles are comprehensive
        if gaps:
            # Each gap should have required fields
            for gap in gaps:
                assert "gap_type" in gap
                assert "description" in gap
                assert "opportunity" in gap

    @pytest.mark.asyncio
    async def test_differentiation_requirements_generated(self, real_state_with_topic_context):
        """Differentiation requirements are generated."""
        result = await content_landscape_analysis_node(real_state_with_topic_context)

        strategy = result.get("content_strategy")
        assert strategy is not None

        requirements = strategy.get("differentiation_requirements", [])

        # Should have some requirements
        assert len(requirements) >= 1

        # Requirements should be actionable
        for req in requirements:
            assert isinstance(req, str)
            assert len(req) > 5  # Not just a single word

    @pytest.mark.asyncio
    async def test_handles_limited_topic_context(self):
        """Test with minimal topic context (edge case)."""
        state: BlogAgentState = {
            "title": "Advanced Kubernetes Operators",
            "context": "Building custom operators with Kubebuilder",
            "topic_context": [
                {
                    "title": "Kubernetes Operators Overview",
                    "url": "https://kubernetes.io/docs/concepts/extend-kubernetes/operator/",
                    "snippet": "Operators are software extensions to Kubernetes.",
                },
            ],
        }

        result = await content_landscape_analysis_node(state)

        # Should still produce a result (with minimal strategy if needed)
        assert result["current_phase"] == Phase.PLANNING.value
        assert "content_strategy" in result

    @pytest.mark.asyncio
    async def test_different_topic(self):
        """Test with a different topic to verify generalization."""
        state: BlogAgentState = {
            "title": "Building RAG Applications with LangChain",
            "context": "Vector databases, embeddings, retrieval strategies",
            "topic_context": [
                {
                    "title": "LangChain RAG Tutorial",
                    "url": "https://python.langchain.com/docs/use_cases/question_answering/",
                    "snippet": "How to build RAG applications with LangChain.",
                },
                {
                    "title": "Pinecone Vector Database",
                    "url": "https://www.pinecone.io/",
                    "snippet": "Pinecone is a managed vector database for ML applications.",
                },
                {
                    "title": "OpenAI Embeddings Guide",
                    "url": "https://platform.openai.com/docs/guides/embeddings",
                    "snippet": "How to use OpenAI embeddings for various tasks.",
                },
            ],
        }

        result = await content_landscape_analysis_node(state)

        # Should succeed
        assert result["current_phase"] == Phase.PLANNING.value
        assert result.get("content_strategy") is not None

        strategy = result["content_strategy"]

        # Target persona should be relevant
        assert strategy.get("target_persona") in [
            "junior_engineer",
            "senior_engineer",
            "senior_architect",
            "data_scientist",
            "devops_engineer",
            "ml_engineer",
        ] or "engineer" in strategy.get("target_persona", "").lower()


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestContentLandscapeArticleAnalysis:
    """Tests for article analysis quality."""

    @pytest.mark.asyncio
    async def test_analyzed_articles_have_structure(self, real_state_with_topic_context):
        """Analyzed articles have proper structure."""
        result = await content_landscape_analysis_node(real_state_with_topic_context)

        strategy = result.get("content_strategy")
        assert strategy is not None

        articles = strategy.get("analyzed_articles", [])

        # Should have analyzed some articles
        if articles:
            for article in articles:
                assert "url" in article
                assert "title" in article
                assert "main_angle" in article
                assert "strengths" in article
                assert "weaknesses" in article
                assert "key_points_covered" in article

                # Main angle should be meaningful
                assert len(article.get("main_angle", "")) > 5


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestContentLandscapeEdgeCases:
    """Edge case tests for content landscape analysis."""

    @pytest.mark.asyncio
    async def test_empty_topic_context_returns_none(self):
        """Empty topic context returns None strategy and continues."""
        state: BlogAgentState = {
            "title": "Test Topic",
            "context": "Test context",
            "topic_context": [],
        }

        result = await content_landscape_analysis_node(state)

        assert result["content_strategy"] is None
        assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_missing_title_fails(self):
        """Missing title causes failure."""
        state: BlogAgentState = {
            "context": "Test context",
            "topic_context": [{"title": "T", "url": "https://a.com", "snippet": "S"}],
        }

        result = await content_landscape_analysis_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")
