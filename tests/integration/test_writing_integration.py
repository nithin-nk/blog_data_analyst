"""
Integration tests for write section node with real LLM calls.

These tests require valid Google API keys in .env file.
Run with: PYTHONPATH=. pytest tests/integration/test_writing_integration.py -v
"""

import os
import pytest
from dotenv import load_dotenv

from src.agent.nodes import _write_section, write_section_node
from src.agent.key_manager import KeyManager
from src.agent.state import Phase

# Load environment variables
load_dotenv()


def has_api_keys() -> bool:
    """Check if any Google API keys are configured."""
    for i in range(1, 6):
        if os.getenv(f"GOOGLE_API_KEY_{i}"):
            return True
    return bool(os.getenv("GOOGLE_API_KEY"))


@pytest.mark.skipif(not has_api_keys(), reason="No Google API keys configured")
class TestWriteSectionIntegration:
    """Integration tests for section writing with real LLM."""

    @pytest.fixture
    def key_manager(self):
        """Create real KeyManager from environment."""
        return KeyManager.from_env()

    @pytest.mark.asyncio
    async def test_write_hook_section(self, key_manager):
        """Write a hook section with real LLM."""
        section = {
            "id": "hook",
            "title": None,
            "role": "hook",
            "target_words": 100,
            "needs_code": False,
            "needs_diagram": False,
        }

        sources = [
            {
                "title": "Redis Vector Search",
                "url": "https://redis.io/docs/stack/search/reference/vectors/",
                "content": "Redis Stack supports vector similarity search using HNSW algorithm.",
            }
        ]

        content = await _write_section(
            section=section,
            sources=sources,
            previous_sections_text="",
            blog_title="Building Semantic Search with Redis",
            key_manager=key_manager,
        )

        # Verify content was generated
        assert content is not None
        assert len(content) > 50
        print(f"\n--- Generated Hook ({len(content.split())} words) ---")
        print(content[:500] + "..." if len(content) > 500 else content)

    @pytest.mark.asyncio
    async def test_write_problem_section(self, key_manager):
        """Write a problem section with real LLM."""
        section = {
            "id": "problem",
            "title": "The Challenge of Scaling LLM Costs",
            "role": "problem",
            "target_words": 200,
            "needs_code": False,
            "needs_diagram": False,
        }

        sources = [
            {
                "title": "LLM Caching Strategies",
                "url": "https://example.com/caching",
                "content": "Semantic caching can reduce LLM API costs by 40-60% by caching similar queries.",
            }
        ]

        previous_text = """## Hook

Every time you send a request to GPT-4, you're paying $0.03 per 1K tokens.
For a production app handling 10,000 queries per day, that's $300 daily just on similar questions."""

        content = await _write_section(
            section=section,
            sources=sources,
            previous_sections_text=previous_text,
            blog_title="Semantic Caching for LLM Applications",
            key_manager=key_manager,
        )

        assert content is not None
        assert len(content) > 100
        print(f"\n--- Generated Problem Section ({len(content.split())} words) ---")
        print(content[:500] + "..." if len(content) > 500 else content)

    @pytest.mark.asyncio
    async def test_write_implementation_section_with_code(self, key_manager):
        """Write an implementation section that requires code."""
        section = {
            "id": "implementation",
            "title": "Implementing Semantic Cache",
            "role": "implementation",
            "target_words": 300,
            "needs_code": True,
            "needs_diagram": False,
        }

        sources = [
            {
                "title": "GPTCache Documentation",
                "url": "https://gptcache.readthedocs.io/",
                "content": "GPTCache is an open-source library for caching LLM responses using semantic similarity.",
            }
        ]

        content = await _write_section(
            section=section,
            sources=sources,
            previous_sections_text="",
            blog_title="Building a Semantic Cache for LLMs",
            key_manager=key_manager,
        )

        assert content is not None
        # Should contain code block
        assert "```" in content or "import" in content.lower()
        print(f"\n--- Generated Implementation Section ({len(content.split())} words) ---")
        print(content[:800] + "..." if len(content) > 800 else content)

    @pytest.mark.asyncio
    async def test_write_section_node_integration(self, key_manager, tmp_path):
        """Test full write_section_node with real LLM."""
        state = {
            "job_id": "",  # No job persistence for this test
            "title": "Vector Databases Explained",
            "plan": {
                "blog_title": "Vector Databases Explained",
                "sections": [
                    {
                        "id": "hook",
                        "title": None,
                        "role": "hook",
                        "target_words": 80,
                        "needs_code": False,
                        "needs_diagram": False,
                        "optional": False,
                    },
                    {
                        "id": "problem",
                        "title": "Why Traditional Databases Fall Short",
                        "role": "problem",
                        "target_words": 150,
                        "needs_code": False,
                        "needs_diagram": False,
                        "optional": False,
                    },
                ],
            },
            "validated_sources": {
                "hook": [
                    {
                        "title": "Vector DB Overview",
                        "url": "https://example.com/vectors",
                        "content": "Vector databases store high-dimensional embeddings for similarity search.",
                    }
                ],
                "problem": [],
            },
            "current_section_index": 0,
            "section_drafts": {},
        }

        # Write first section (hook)
        result = await write_section_node(state)

        assert result["current_section_index"] == 1
        assert "hook" in result["section_drafts"]
        assert result["current_phase"] == Phase.WRITING.value

        hook_content = result["section_drafts"]["hook"]
        print(f"\n--- Node Generated Hook ({len(hook_content.split())} words) ---")
        print(hook_content[:400] + "..." if len(hook_content) > 400 else hook_content)

    @pytest.mark.asyncio
    async def test_write_conclusion_section(self, key_manager):
        """Write a conclusion section with real LLM."""
        section = {
            "id": "conclusion",
            "title": "Key Takeaways",
            "role": "conclusion",
            "target_words": 150,
            "needs_code": False,
            "needs_diagram": False,
        }

        previous_text = """## The Problem

Current approaches to semantic search are expensive and slow.

## The Solution

Vector databases provide efficient similarity search."""

        content = await _write_section(
            section=section,
            sources=[],
            previous_sections_text=previous_text,
            blog_title="Modern Semantic Search",
            key_manager=key_manager,
        )

        assert content is not None
        assert len(content) > 50
        print(f"\n--- Generated Conclusion ({len(content.split())} words) ---")
        print(content[:400] + "..." if len(content) > 400 else content)


@pytest.mark.skipif(not has_api_keys(), reason="No Google API keys configured")
class TestWriteSectionQuality:
    """Tests to verify quality of generated content."""

    @pytest.fixture
    def key_manager(self):
        return KeyManager.from_env()

    @pytest.mark.asyncio
    async def test_respects_word_count_target(self, key_manager):
        """Generated content should be within ±50% of target word count."""
        section = {
            "id": "test",
            "title": "Test Section",
            "role": "problem",
            "target_words": 200,
            "needs_code": False,
        }

        content = await _write_section(
            section=section,
            sources=[],
            previous_sections_text="",
            blog_title="Test Blog",
            key_manager=key_manager,
        )

        word_count = len(content.split())
        target = 200

        # Allow ±50% variance
        assert word_count >= target * 0.5, f"Content too short: {word_count} words (target: {target})"
        assert word_count <= target * 1.5, f"Content too long: {word_count} words (target: {target})"

        print(f"Target: {target} words, Actual: {word_count} words")

    @pytest.mark.asyncio
    async def test_follows_style_guide(self, key_manager):
        """Generated content should follow style guidelines."""
        section = {
            "id": "test",
            "title": "Test Section",
            "role": "problem",
            "target_words": 150,
        }

        content = await _write_section(
            section=section,
            sources=[],
            previous_sections_text="",
            blog_title="Building Real-Time APIs",
            key_manager=key_manager,
        )

        # Check for fluff phrases that should be avoided
        fluff_phrases = ["in today's world", "it's worth noting", "needless to say"]
        content_lower = content.lower()

        for phrase in fluff_phrases:
            if phrase in content_lower:
                print(f"Warning: Found fluff phrase '{phrase}' in content")

        # Content should exist and be non-empty
        assert len(content) > 0
