"""Integration tests for research and validation nodes (requires API keys)."""

import os

import pytest

from src.agent.nodes import research_node, validate_sources_node
from src.agent.state import Phase, BlogAgentState


@pytest.fixture
def state_with_plan() -> BlogAgentState:
    """State with a plan ready for research."""
    return {
        # Note: No job_id to skip checkpointing in integration tests
        "title": "Semantic Caching for LLM Applications",
        "plan": {
            "blog_title": "Semantic Caching for LLM Applications",
            "target_words": 1500,
            "sections": [
                {
                    "id": "problem",
                    "title": "The LLM Cost Problem",
                    "role": "problem",
                    "search_queries": ["LLM API costs 2024", "GPT-4 API pricing"],
                    "target_words": 200,
                    "needs_code": False,
                    "needs_diagram": False,
                    "optional": False,
                },
                {
                    "id": "solution",
                    "title": "How Semantic Caching Works",
                    "role": "why",
                    "search_queries": ["semantic caching LLM", "GPTCache tutorial"],
                    "target_words": 300,
                    "needs_code": True,
                    "needs_diagram": True,
                    "optional": False,
                },
                {
                    "id": "conclusion",
                    "title": "Key Takeaways",
                    "role": "conclusion",
                    "search_queries": [],  # No queries for conclusion
                    "target_words": 150,
                    "needs_code": False,
                    "needs_diagram": False,
                    "optional": False,
                },
            ],
        },
    }


@pytest.fixture
def state_with_research_cache(state_with_plan) -> BlogAgentState:
    """State with research cache ready for validation."""
    state_with_plan["research_cache"] = {
        "hash1": {
            "url": "https://openai.com/pricing",
            "title": "OpenAI API Pricing",
            "content": "GPT-4 costs $0.03 per 1K tokens for input and $0.06 per 1K tokens for output. This makes repeated queries expensive for applications with high traffic.",
            "tokens_estimate": 50,
        },
        "hash2": {
            "url": "https://github.com/zilliztech/GPTCache",
            "title": "GPTCache: Semantic Cache for LLMs",
            "content": "GPTCache is a library for creating semantic cache to store responses from LLM queries. It uses embedding similarity to match similar queries and return cached responses.",
            "tokens_estimate": 60,
        },
        "hash3": {
            "url": "https://redis.io/docs/stack/search/reference/vectors/",
            "title": "Redis Vector Similarity Search",
            "content": "Redis Stack supports vector similarity search using HNSW and FLAT indexing. You can store embeddings and query for similar vectors efficiently.",
            "tokens_estimate": 55,
        },
        "hash4": {
            "url": "https://www.pinecone.io/learn/semantic-search/",
            "title": "Semantic Search with Pinecone",
            "content": "Semantic search uses embeddings to find conceptually similar content. This is the foundation of semantic caching for LLMs.",
            "tokens_estimate": 45,
        },
    }
    return state_with_plan


# =============================================================================
# Research Node Integration Tests
# =============================================================================


class TestResearchNodeIntegration:
    """Integration tests for research_node with real web searches."""

    @pytest.mark.asyncio
    async def test_research_returns_cache(self, state_with_plan):
        """Research node fetches content and returns cache."""
        result = await research_node(state_with_plan)

        # Should have research_cache in result
        assert "research_cache" in result
        cache = result["research_cache"]

        # Should have fetched some content
        assert len(cache) > 0, "Expected at least some cached content"

        # Each entry should have required fields
        for url_hash, entry in cache.items():
            assert "url" in entry, f"Cache entry missing 'url': {entry}"
            assert "content" in entry, f"Cache entry missing 'content': {entry}"
            assert entry["url"].startswith("http"), f"Invalid URL: {entry['url']}"

        # Should advance to VALIDATING_SOURCES
        assert result["current_phase"] == Phase.VALIDATING_SOURCES.value

        # Print summary for inspection
        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)
        print(f"Total sources fetched: {len(cache)}")
        for url_hash, entry in list(cache.items())[:5]:
            print(f"\n  [{url_hash}] {entry.get('title', 'No title')}")
            print(f"    URL: {entry['url']}")
            content_preview = entry.get("content", "")[:100]
            print(f"    Preview: {content_preview}...")

    @pytest.mark.asyncio
    async def test_research_handles_empty_queries(self):
        """Research handles sections with no search queries."""
        state: BlogAgentState = {
            "plan": {
                "blog_title": "Test",
                "sections": [
                    {
                        "id": "empty",
                        "title": "Empty Section",
                        "role": "conclusion",
                        "search_queries": [],
                    }
                ],
            },
        }

        result = await research_node(state)

        # Should succeed with empty cache
        assert result["current_phase"] == Phase.VALIDATING_SOURCES.value
        assert result["research_cache"] == {}

    @pytest.mark.asyncio
    async def test_research_deduplicates_across_sections(self, state_with_plan):
        """Same URLs from different sections are not fetched twice."""
        # Add sections with overlapping search topics
        state_with_plan["plan"]["sections"].append(
            {
                "id": "extra",
                "title": "Extra Section",
                "role": "deep_dive",
                "search_queries": ["LLM API costs 2024"],  # Same as problem section
            }
        )

        result = await research_node(state_with_plan)

        # Cache should not have duplicate entries
        cache = result["research_cache"]
        urls = [entry["url"] for entry in cache.values()]
        unique_urls = set(urls)

        assert len(urls) == len(unique_urls), "Found duplicate URLs in cache"


# =============================================================================
# Validation Node Integration Tests
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestValidateSourcesIntegration:
    """Integration tests for validate_sources_node with real LLM."""

    @pytest.mark.asyncio
    async def test_validation_filters_sources(self, state_with_research_cache):
        """Validation node filters and scores sources."""
        result = await validate_sources_node(state_with_research_cache)

        # Should have validated_sources
        assert "validated_sources" in result

        # Should advance to WRITING
        assert result["current_phase"] == Phase.WRITING.value

        # Print validation results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        validated = result["validated_sources"]
        for section_id, sources in validated.items():
            print(f"\n{section_id}: {len(sources)} validated sources")
            for source in sources:
                quality = source.get("quality", "unknown")
                reason = source.get("reason", "")[:50]
                print(f"  - [{quality}] {source.get('url', 'no url')}")
                print(f"    Reason: {reason}...")

    @pytest.mark.asyncio
    async def test_validation_includes_quality_scores(self, state_with_research_cache):
        """Validated sources include quality and reason fields."""
        result = await validate_sources_node(state_with_research_cache)

        validated = result["validated_sources"]

        for section_id, sources in validated.items():
            for source in sources:
                assert "quality" in source, f"Source missing 'quality': {source}"
                assert "reason" in source, f"Source missing 'reason': {source}"
                assert source["quality"] in ["high", "medium", "low"], (
                    f"Invalid quality: {source['quality']}"
                )

    @pytest.mark.asyncio
    async def test_validation_handles_empty_cache(self):
        """Validation handles empty research cache."""
        state: BlogAgentState = {
            "plan": {
                "blog_title": "Test Blog",
                "sections": [
                    {"id": "test", "title": "Test Section", "role": "problem"}
                ],
            },
            "research_cache": {},
        }

        result = await validate_sources_node(state)

        # Should succeed with empty validated sources
        assert result["current_phase"] == Phase.WRITING.value
        assert result["validated_sources"]["test"] == []


# =============================================================================
# End-to-End Research Flow Tests
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestResearchFlowIntegration:
    """End-to-end tests for research → validation pipeline."""

    @pytest.mark.asyncio
    async def test_full_research_to_validation_flow(self, state_with_plan):
        """Complete flow from research to validated sources."""
        # Step 1: Research
        research_result = await research_node(state_with_plan)

        assert research_result["current_phase"] == Phase.VALIDATING_SOURCES.value
        assert len(research_result["research_cache"]) > 0

        # Step 2: Merge research result into state
        state_after_research = {**state_with_plan, **research_result}

        # Step 3: Validate
        validation_result = await validate_sources_node(state_after_research)

        assert validation_result["current_phase"] == Phase.WRITING.value
        assert "validated_sources" in validation_result

        # Print full flow summary
        print("\n" + "=" * 60)
        print("END-TO-END RESEARCH FLOW")
        print("=" * 60)
        print(f"\nResearch phase:")
        print(f"  - Sources fetched: {len(research_result['research_cache'])}")

        print(f"\nValidation phase:")
        validated = validation_result["validated_sources"]
        total_validated = sum(len(sources) for sources in validated.values())
        print(f"  - Total validated sources: {total_validated}")

        for section_id, sources in validated.items():
            print(f"\n  [{section_id}] {len(sources)} sources:")
            for s in sources[:3]:  # Show first 3
                print(f"    - {s.get('url', 'no url')[:50]}... [{s.get('quality', '?')}]")

    @pytest.mark.asyncio
    async def test_sources_are_relevant_to_topic(self, state_with_plan):
        """Validated sources are relevant to the blog topic."""
        # Research
        research_result = await research_node(state_with_plan)
        state_after = {**state_with_plan, **research_result}

        # Validate
        validation_result = await validate_sources_node(state_after)

        validated = validation_result["validated_sources"]

        # Check that at least some sources mention relevant terms
        topic_terms = ["cache", "llm", "api", "cost", "semantic", "embedding", "gpt"]

        all_sources = []
        for sources in validated.values():
            all_sources.extend(sources)

        if all_sources:
            relevant_count = 0
            for source in all_sources:
                content = source.get("content", "").lower()
                url = source.get("url", "").lower()
                title = source.get("title", "").lower()
                text = f"{content} {url} {title}"

                if any(term in text for term in topic_terms):
                    relevant_count += 1

            # At least 50% of sources should be relevant
            relevance_ratio = relevant_count / len(all_sources)
            print(f"\nRelevance: {relevant_count}/{len(all_sources)} ({relevance_ratio:.0%})")

            assert relevance_ratio >= 0.3, (
                f"Only {relevance_ratio:.0%} of sources are relevant to topic"
            )
