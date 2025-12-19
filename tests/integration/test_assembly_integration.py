"""
Integration tests for final assembly node.

These tests verify assembly with real file system operations.
Run with: PYTHONPATH=. pytest tests/integration/test_assembly_integration.py -v
"""

import json
import pytest

from src.agent.nodes import final_assembly_node
from src.agent.state import JobManager, Phase


class TestFinalAssemblyIntegration:
    """Integration tests for final assembly with real file operations."""

    @pytest.fixture
    def job_manager(self, tmp_path):
        """Create JobManager with temporary directory."""
        return JobManager(base_dir=tmp_path)

    @pytest.fixture
    def sample_sections(self):
        """Sample section drafts for testing."""
        return {
            "hook": """Every API call to GPT-4 costs money. At scale, those pennies become thousands of dollars.
What if you could cache 60% of your LLM requests with zero quality loss?""",
            "problem": """Traditional caching doesn't work for LLM applications. Here's why:

1. **Exact match fails**: Users ask the same question differently. "How do I deploy?" and "What's the deployment process?" are semantically identical but textually different.

2. **Cache invalidation is tricky**: When your knowledge base updates, which cached responses need invalidation?

3. **Context matters**: The same question might need different answers depending on conversation history.""",
            "implementation": """Here's how to implement semantic caching with Redis:

```python
import redis
from sentence_transformers import SentenceTransformer

# Initialize
r = redis.Redis()
model = SentenceTransformer('all-MiniLM-L6-v2')

def cache_response(query: str, response: str):
    embedding = model.encode(query)
    # Store with vector similarity index
    r.hset(f"cache:{hash(query)}", mapping={
        "query": query,
        "response": response,
        "embedding": embedding.tobytes()
    })
```

This approach gives you sub-millisecond lookups while handling semantic similarity.""",
            "conclusion": """Key takeaways:

1. Semantic caching can reduce LLM costs by 40-60%
2. Use vector similarity, not exact matching
3. Start with Redis + Sentence Transformers for quick wins

Next step: Try implementing this in your staging environment. Track your cache hit rate for a week.""",
        }

    @pytest.fixture
    def sample_plan(self):
        """Sample plan for testing."""
        return {
            "blog_title": "Semantic Caching for LLM Applications",
            "sections": [
                {"id": "hook", "title": None, "role": "hook", "target_words": 50, "optional": False},
                {"id": "problem", "title": "The Problem with Traditional Caching", "role": "problem", "target_words": 150, "optional": False},
                {"id": "implementation", "title": "Implementation with Redis", "role": "implementation", "target_words": 200, "optional": False},
                {"id": "conclusion", "title": "Key Takeaways", "role": "conclusion", "target_words": 100, "optional": False},
            ],
        }

    @pytest.mark.asyncio
    async def test_assembly_creates_final_md(self, job_manager, sample_sections, sample_plan):
        """Assembly creates final.md file."""
        from unittest.mock import patch

        job_id = job_manager.create_job("Test Blog", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        # Patch JobManager to use our test instance
        with patch("src.agent.nodes.JobManager", return_value=job_manager):
            result = await final_assembly_node(state)

        # Check result
        assert result["current_phase"] == Phase.REVIEWING.value

        # Check file was created
        job_dir = job_manager.get_job_dir(job_id)
        final_md = job_dir / "final.md"
        assert final_md.exists()

        content = final_md.read_text()
        assert "# Semantic Caching for LLM Applications" in content
        assert "Every API call to GPT-4" in content  # Hook content
        assert "## The Problem with Traditional Caching" in content
        assert "## Implementation with Redis" in content
        assert "```python" in content  # Code block preserved

    @pytest.mark.asyncio
    async def test_assembly_creates_metadata_json(self, job_manager, sample_sections, sample_plan):
        """Assembly creates metadata.json file."""
        from unittest.mock import patch

        job_id = job_manager.create_job("Test Blog", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        with patch("src.agent.nodes.JobManager", return_value=job_manager):
            result = await final_assembly_node(state)

        # Check metadata file
        job_dir = job_manager.get_job_dir(job_id)
        metadata_file = job_dir / "metadata.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text())
        assert metadata["blog_title"] == "Semantic Caching for LLM Applications"
        assert metadata["word_count"] > 100
        assert metadata["reading_time_minutes"] >= 1
        assert metadata["section_count"] == 4
        assert result["current_phase"] == Phase.REVIEWING.value

    @pytest.mark.asyncio
    async def test_assembly_creates_v1_draft(self, job_manager, sample_sections, sample_plan):
        """Assembly creates drafts/v1.md file."""
        from unittest.mock import patch

        job_id = job_manager.create_job("Test Blog", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        with patch("src.agent.nodes.JobManager", return_value=job_manager):
            result = await final_assembly_node(state)

        # Check v1 draft
        job_dir = job_manager.get_job_dir(job_id)
        v1_md = job_dir / "drafts" / "v1.md"
        assert v1_md.exists()

        # v1.md should match final.md
        final_content = (job_dir / "final.md").read_text()
        v1_content = v1_md.read_text()
        assert final_content == v1_content
        assert result["current_phase"] == Phase.REVIEWING.value

    @pytest.mark.asyncio
    async def test_assembly_word_count_accuracy(self, job_manager, sample_sections, sample_plan):
        """Word count in metadata matches actual content."""
        job_id = job_manager.create_job("Test Blog", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        result = await final_assembly_node(state)

        # Verify word count
        combined = result["combined_draft"]
        actual_words = len(combined.split())
        reported_words = result["metadata"]["word_count"]

        assert actual_words == reported_words

    @pytest.mark.asyncio
    async def test_assembly_preserves_section_order(self, job_manager, sample_sections, sample_plan):
        """Sections appear in the order specified in plan."""
        job_id = job_manager.create_job("Test Blog", "Test context")

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        result = await final_assembly_node(state)
        combined = result["combined_draft"]

        # Find positions
        hook_pos = combined.find("Every API call")
        problem_pos = combined.find("Traditional caching")
        impl_pos = combined.find("Here's how to implement")
        conclusion_pos = combined.find("Key takeaways")

        # Verify order
        assert hook_pos < problem_pos < impl_pos < conclusion_pos

    @pytest.mark.asyncio
    async def test_assembly_without_job_id(self, sample_sections, sample_plan):
        """Assembly works without job_id (no file persistence)."""
        state = {
            "job_id": "",  # No persistence
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sample_sections,
        }

        result = await final_assembly_node(state)

        assert result["current_phase"] == Phase.REVIEWING.value
        assert "combined_draft" in result
        assert len(result["combined_draft"]) > 0

    @pytest.mark.asyncio
    async def test_assembly_reading_time_calculation(self, job_manager, sample_plan):
        """Reading time calculated correctly for various lengths."""
        job_id = job_manager.create_job("Test Blog", "Test context")

        # Create sections with known word counts
        # ~400 words = 2 minutes at 200 wpm
        sections = {
            "hook": " ".join(["word"] * 100),
            "problem": " ".join(["word"] * 100),
            "implementation": " ".join(["word"] * 100),
            "conclusion": " ".join(["word"] * 100),
        }

        state = {
            "job_id": job_id,
            "title": "Test Blog",
            "plan": sample_plan,
            "section_drafts": sections,
        }

        result = await final_assembly_node(state)

        # 400 words in sections + some header text ≈ 2 min
        assert result["metadata"]["reading_time_minutes"] >= 2


class TestAssemblyEdgeCases:
    """Edge case tests for assembly."""

    @pytest.fixture
    def job_manager(self, tmp_path):
        return JobManager(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_assembly_with_missing_section(self, job_manager):
        """Handles missing section gracefully."""
        job_id = job_manager.create_job("Test", "context")

        plan = {
            "blog_title": "Test",
            "sections": [
                {"id": "hook", "role": "hook", "optional": False},
                {"id": "problem", "title": "Problem", "role": "problem", "optional": False},
            ],
        }

        # Only provide hook, not problem
        sections = {"hook": "Hook content only."}

        state = {
            "job_id": job_id,
            "plan": plan,
            "section_drafts": sections,
        }

        result = await final_assembly_node(state)

        # Should still succeed with available content
        assert result["current_phase"] == Phase.REVIEWING.value
        assert "Hook content only" in result["combined_draft"]

    @pytest.mark.asyncio
    async def test_assembly_with_markdown_special_chars(self, job_manager):
        """Handles markdown special characters correctly."""
        job_id = job_manager.create_job("Test", "context")

        plan = {
            "blog_title": "Test: Special *Characters* & More",
            "sections": [
                {"id": "content", "title": "The `Code` Section", "role": "problem", "optional": False},
            ],
        }

        sections = {
            "content": "This has **bold**, *italic*, and `code`.\n\n> A blockquote\n\n---"
        }

        state = {
            "job_id": job_id,
            "plan": plan,
            "section_drafts": sections,
        }

        result = await final_assembly_node(state)

        combined = result["combined_draft"]
        assert "**bold**" in combined
        assert "*italic*" in combined
        assert "`code`" in combined
        assert "> A blockquote" in combined

    @pytest.mark.asyncio
    async def test_assembly_empty_section_content(self):
        """Handles empty section content."""
        state = {
            "job_id": "",
            "plan": {
                "blog_title": "Test",
                "sections": [
                    {"id": "problem", "title": "Problem", "role": "problem", "optional": False},
                ],
            },
            "section_drafts": {"problem": ""},  # Empty content
        }

        result = await final_assembly_node(state)

        # Should skip empty section
        assert "## Problem" not in result["combined_draft"]
