"""Full flow integration test demo - Discovery through Validation."""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agent.nodes import (
    topic_discovery_node,
    planning_node,
    research_node,
    validate_sources_node,
)
from src.agent.state import BlogAgentState


async def run_full_flow():
    """Run complete flow from topic discovery to source validation."""

    print("\n" + "=" * 70)
    print("BLOG AGENT - FULL INTEGRATION TEST")
    print("Topic: Semantic Caching for LLM Applications")
    print("=" * 70)

    # Initial state
    state: BlogAgentState = {
        "title": "Semantic Caching for LLM Applications",
        "context": "Exploring GPTCache and Redis vector search for caching LLM responses. Focus on cost reduction and latency improvements.",
        "target_length": "medium",
    }

    # =========================================================================
    # Phase 0.5: Topic Discovery
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 0.5: TOPIC DISCOVERY")
    print("-" * 70)

    discovery_result = await topic_discovery_node(state)

    if discovery_result.get("current_phase") == "failed":
        print(f"❌ Discovery failed: {discovery_result.get('error_message')}")
        return

    print(f"✅ Phase: {discovery_result['current_phase']}")
    print(f"\nDiscovery Queries Generated:")
    for i, q in enumerate(discovery_result.get("discovery_queries", []), 1):
        print(f"  {i}. {q}")

    print(f"\nTopic Context ({len(discovery_result.get('topic_context', []))} results):")
    for item in discovery_result.get("topic_context", [])[:5]:
        print(f"  • {item.get('title', 'No title')[:60]}...")

    # Merge into state
    state.update(discovery_result)

    # =========================================================================
    # Phase 1: Planning
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 1: PLANNING")
    print("-" * 70)

    planning_result = await planning_node(state)

    if planning_result.get("current_phase") == "failed":
        print(f"❌ Planning failed: {planning_result.get('error_message')}")
        return

    plan = planning_result.get("plan", {})
    sections = plan.get("sections", [])

    required = [s for s in sections if not s.get("optional")]
    optional = [s for s in sections if s.get("optional")]

    print(f"✅ Phase: {planning_result['current_phase']}")
    print(f"\nBlog Plan: {plan.get('blog_title')}")
    print(f"Target Words: {plan.get('target_words')}")
    print(f"Total Sections: {len(sections)} ({len(required)} required, {len(optional)} optional)")

    print("\n📋 SECTIONS:")
    required_words = 0
    optional_words = 0

    for i, section in enumerate(sections, 1):
        opt_tag = " [OPTIONAL]" if section.get("optional") else ""
        code_tag = " 💻" if section.get("needs_code") else ""
        diagram_tag = " 📊" if section.get("needs_diagram") else ""

        print(f"\n  {i}. [{section['role'].upper()}]{opt_tag}{code_tag}{diagram_tag}")
        print(f"     Title: {section.get('title') or '(hook - no title)'}")
        print(f"     ID: {section['id']}")
        print(f"     Words: {section.get('target_words', 0)}")
        print(f"     Queries: {section.get('search_queries', [])}")

        if section.get("optional"):
            optional_words += section.get("target_words", 0)
        else:
            required_words += section.get("target_words", 0)

    print(f"\n📊 Word Distribution:")
    print(f"   Required sections: {required_words} words")
    print(f"   Optional sections: {optional_words} words")
    print(f"   Total (if all selected): {required_words + optional_words} words")

    # Merge into state
    state.update(planning_result)

    # =========================================================================
    # Phase 2: Research
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 2: RESEARCH")
    print("-" * 70)

    research_result = await research_node(state)

    if research_result.get("current_phase") == "failed":
        print(f"❌ Research failed: {research_result.get('error_message')}")
        return

    cache = research_result.get("research_cache", {})

    print(f"✅ Phase: {research_result['current_phase']}")
    print(f"\n📚 Research Cache: {len(cache)} unique sources fetched")

    print("\nTop Sources:")
    for i, (url_hash, data) in enumerate(list(cache.items())[:8], 1):
        title = data.get("title", "No title")[:50]
        url = data.get("url", "")[:60]
        tokens = data.get("tokens_estimate", 0)
        print(f"  {i}. [{url_hash}] {title}...")
        print(f"     URL: {url}...")
        print(f"     Tokens: ~{tokens}")

    # Merge into state
    state.update(research_result)

    # =========================================================================
    # Phase 2.5: Validate Sources
    # =========================================================================
    print("\n" + "-" * 70)
    print("PHASE 2.5: VALIDATE SOURCES")
    print("-" * 70)

    validation_result = await validate_sources_node(state)

    if validation_result.get("current_phase") == "failed":
        print(f"❌ Validation failed: {validation_result.get('error_message')}")
        return

    validated = validation_result.get("validated_sources", {})

    print(f"✅ Phase: {validation_result['current_phase']}")

    total_validated = 0
    print("\n📝 VALIDATED SOURCES BY SECTION:")

    for section_id, sources in validated.items():
        total_validated += len(sources)
        print(f"\n  [{section_id}] - {len(sources)} validated sources:")

        for source in sources[:3]:  # Show top 3 per section
            quality = source.get("quality", "?")
            quality_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(quality, "⚪")
            url = source.get("url", "")[:50]
            reason = source.get("reason", "")[:60]

            print(f"    {quality_emoji} [{quality}] {url}...")
            print(f"       Reason: {reason}...")

        if len(sources) > 3:
            print(f"    ... and {len(sources) - 3} more")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Topic Discovery: {len(state.get('topic_context', []))} context items")
    print(f"✅ Planning: {len(sections)} sections ({len(required)} required, {len(optional)} optional)")
    print(f"✅ Research: {len(cache)} sources fetched")
    print(f"✅ Validation: {total_validated} sources validated")
    print(f"\n🎯 Ready for Phase 3: WRITING")
    print("=" * 70)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY_1"):
        print("❌ Error: GOOGLE_API_KEY_1 not set. Please configure .env file.")
        sys.exit(1)

    # Run the flow
    asyncio.run(run_full_flow())
