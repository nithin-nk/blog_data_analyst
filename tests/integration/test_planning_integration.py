"""Integration tests for planning_node (requires API keys)."""

import os

import pytest

from src.agent.nodes import planning_node
from src.agent.state import Phase, BlogAgentState


@pytest.fixture
def real_state_with_context() -> BlogAgentState:
    """Real state with topic_context for integration testing."""
    return {
        "title": "Semantic Caching for LLM Applications",
        "context": "Exploring GPTCache and Redis vector search for caching LLM responses.",
        "target_length": "medium",
        "topic_context": [
            {
                "title": "GPTCache: A Library for Creating Semantic Cache",
                "url": "https://github.com/zilliztech/GPTCache",
                "snippet": "GPTCache is a library for creating semantic cache to store responses from LLM queries.",
            },
            {
                "title": "Redis Vector Similarity Search",
                "url": "https://redis.io/docs/stack/search/reference/vectors/",
                "snippet": "Redis supports vector similarity search using HNSW and FLAT indexing algorithms.",
            },
            {
                "title": "Reducing LLM API Costs with Semantic Caching",
                "url": "https://www.pinecone.io/learn/semantic-search/",
                "snippet": "Semantic caching can reduce API costs by 70% by returning cached responses for similar queries.",
            },
        ],
    }


@pytest.fixture
def real_state_no_context() -> BlogAgentState:
    """Real state without topic_context for integration testing."""
    return {
        "title": "Introduction to GraphQL",
        "context": "REST vs GraphQL, when to use each, practical examples.",
        "target_length": "short",
    }


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY_1"),
    reason="Requires GOOGLE_API_KEY_1 environment variable",
)
class TestPlanningIntegration:
    """Integration tests that hit real APIs."""

    @pytest.mark.asyncio
    async def test_end_to_end_planning(self, real_state_with_context):
        """Full planning flow with real APIs."""
        result = await planning_node(real_state_with_context)

        # Should have generated plan
        assert "plan" in result
        plan = result["plan"]

        # Should have sections
        assert "sections" in plan
        sections = plan["sections"]
        assert len(sections) >= 6, f"Expected at least 6 sections, got {len(sections)}"

        # Each section should have required fields
        for section in sections:
            assert "id" in section, f"Section missing 'id': {section}"
            assert "role" in section, f"Section missing 'role': {section}"
            assert "search_queries" in section, f"Section missing 'search_queries': {section}"
            assert "target_words" in section, f"Section missing 'target_words': {section}"
            assert "needs_code" in section, f"Section missing 'needs_code': {section}"
            assert "needs_diagram" in section, f"Section missing 'needs_diagram': {section}"
            assert "optional" in section, f"Section missing 'optional': {section}"

        # Should advance to researching
        assert result["current_phase"] == Phase.RESEARCHING.value

        # Count required and optional sections
        required = [s for s in sections if not s["optional"]]
        optional = [s for s in sections if s["optional"]]

        # Print plan for inspection
        print("\n" + "=" * 60)
        print("GENERATED BLOG PLAN")
        print("=" * 60)
        print(f"\nRequired Sections: {len(required)}")
        print(f"Optional Sections: {len(optional)} (user can choose)")
        print("-" * 60)

        required_words = 0
        optional_words = 0
        for i, section in enumerate(sections, 1):
            opt_marker = " [OPTIONAL]" if section["optional"] else ""
            print(f"\n{i}. [{section['role'].upper()}]{opt_marker} {section.get('title') or '(no title)'}")
            print(f"   ID: {section['id']}")
            print(f"   Target Words: {section['target_words']}")
            print(f"   Search Queries: {section['search_queries']}")
            print(f"   Needs Code: {section['needs_code']}, Needs Diagram: {section['needs_diagram']}")
            if section["optional"]:
                optional_words += section["target_words"]
            else:
                required_words += section["target_words"]

        print(f"\nRequired Words: {required_words}")
        print(f"Optional Words: {optional_words}")
        print(f"Total (if all selected): {required_words + optional_words}")
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_plan_has_correct_roles(self, real_state_with_context):
        """Plan includes expected section roles."""
        result = await planning_node(real_state_with_context)

        sections = result["plan"]["sections"]
        roles = [s["role"] for s in sections]

        # Should have problem and conclusion at minimum
        assert "problem" in roles or "why" in roles, f"Missing problem/why section in roles: {roles}"
        assert "conclusion" in roles, f"Missing conclusion in roles: {roles}"

    @pytest.mark.asyncio
    async def test_word_distribution_matches_target(self, real_state_with_context):
        """Required sections target words approximately matches medium (1500)."""
        result = await planning_node(real_state_with_context)

        sections = result["plan"]["sections"]
        required_sections = [s for s in sections if not s["optional"]]
        required_words = sum(s["target_words"] for s in required_sections)

        # Required words should be within 30% of target (1500 for medium)
        target = 1500
        tolerance = target * 0.3  # 30% tolerance
        assert abs(required_words - target) < tolerance, (
            f"Required words {required_words} not within {tolerance} of target {target}"
        )

        print(f"\nRequired word distribution: {required_words} (target: {target})")

    @pytest.mark.asyncio
    async def test_generates_optional_sections(self, real_state_with_context):
        """Plan includes optional deep_dive sections for user choice."""
        result = await planning_node(real_state_with_context)

        sections = result["plan"]["sections"]
        optional = [s for s in sections if s["optional"]]
        required = [s for s in sections if not s["optional"]]

        # Should have at least 1 optional section (targeting 2)
        assert len(optional) >= 1, f"Expected at least 1 optional section, got {len(optional)}"

        # Optional sections should be deep_dive or implementation
        for s in optional:
            assert s["role"] in ["deep_dive", "implementation"], (
                f"Optional section has wrong role: {s['role']}"
            )

        # Core sections (hook, problem, why, conclusion) should NOT be optional
        core_roles = {"hook", "problem", "why", "conclusion"}
        for s in required:
            if s["role"] in core_roles:
                assert not s["optional"], f"{s['role']} section should not be optional"

        print(f"\nOptional sections: {len(optional)}")
        for s in optional:
            print(f"  - {s['title']}: {s['target_words']} words")

    @pytest.mark.asyncio
    async def test_planning_without_topic_context(self, real_state_no_context):
        """Planning works without topic_context (uses LLM knowledge)."""
        result = await planning_node(real_state_no_context)

        # Should succeed
        assert result["current_phase"] == Phase.RESEARCHING.value
        assert "plan" in result

        sections = result["plan"]["sections"]
        assert len(sections) >= 4

        # Print plan
        print("\n" + "=" * 60)
        print("PLAN WITHOUT TOPIC CONTEXT")
        print("=" * 60)
        for section in sections:
            print(f"  - [{section['role']}] {section.get('title') or '(no title)'}: {section['target_words']} words")

    @pytest.mark.asyncio
    async def test_short_blog_plan(self):
        """Short length produces fewer target words."""
        state: BlogAgentState = {
            "title": "Quick Guide to Python Type Hints",
            "context": "Basic type annotations, common patterns.",
            "target_length": "short",
        }

        result = await planning_node(state)

        sections = result["plan"]["sections"]
        total_words = sum(s["target_words"] for s in sections)

        # Should be around 800 for short
        assert total_words < 1200, f"Short blog too long: {total_words} words"
        print(f"\nShort blog: {total_words} words (target: 800)")

    @pytest.mark.asyncio
    async def test_long_blog_plan(self):
        """Long length produces more target words."""
        state: BlogAgentState = {
            "title": "Comprehensive Guide to Kubernetes Networking",
            "context": "CNI plugins, service mesh, network policies, ingress controllers.",
            "target_length": "long",
        }

        result = await planning_node(state)

        sections = result["plan"]["sections"]
        total_words = sum(s["target_words"] for s in sections)

        # Should be around 2500 for long
        assert total_words > 1800, f"Long blog too short: {total_words} words"
        print(f"\nLong blog: {total_words} words (target: 2500)")

    @pytest.mark.asyncio
    async def test_search_queries_are_relevant(self, real_state_with_context):
        """Generated search queries are relevant to sections."""
        result = await planning_node(real_state_with_context)

        sections = result["plan"]["sections"]

        # Collect all queries
        all_queries = []
        for section in sections:
            all_queries.extend(section.get("search_queries", []))

        # At least some queries should be related to the topic
        topic_terms = ["caching", "cache", "llm", "semantic", "gptcache", "vector", "redis"]
        relevant_queries = [
            q for q in all_queries
            if any(term in q.lower() for term in topic_terms)
        ]

        assert len(relevant_queries) > 0, (
            f"No relevant queries found. Queries: {all_queries}"
        )
        print(f"\nRelevant queries: {len(relevant_queries)} of {len(all_queries)}")
