"""
Unit tests for the final assembly node and helper functions.

Tests cover:
- _combine_sections()
- _calculate_reading_time()
- _count_words()
- final_assembly_node() with mocked JobManager
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.nodes import (
    _combine_sections,
    _calculate_reading_time,
    _count_words,
    final_assembly_node,
)
from src.agent.state import Phase


class TestCombineSections:
    """Tests for _combine_sections helper."""

    def test_empty_sections(self):
        """Returns just title when no sections."""
        result = _combine_sections(
            section_drafts={},
            plan={"sections": []},
            blog_title="Test Blog",
        )
        assert "# Test Blog" in result

    def test_single_section(self):
        """Combines a single section with header."""
        section_drafts = {"problem": "This is the problem content."}
        plan = {
            "sections": [
                {"id": "problem", "title": "The Problem", "role": "problem", "optional": False}
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        assert "# Test Blog" in result
        assert "## The Problem" in result
        assert "This is the problem content." in result

    def test_hook_section_no_header(self):
        """Hook section should NOT have H2 header."""
        section_drafts = {"hook": "Opening hook content here."}
        plan = {
            "sections": [
                {"id": "hook", "title": None, "role": "hook", "optional": False}
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        assert "# Test Blog" in result
        assert "Opening hook content here." in result
        # Hook should NOT have ## header
        assert "## hook" not in result.lower()

    def test_multiple_sections_in_order(self):
        """Sections are combined in plan order."""
        section_drafts = {
            "hook": "Hook content.",
            "problem": "Problem content.",
            "conclusion": "Conclusion content.",
        }
        plan = {
            "sections": [
                {"id": "hook", "role": "hook", "optional": False},
                {"id": "problem", "title": "The Problem", "role": "problem", "optional": False},
                {"id": "conclusion", "title": "Conclusion", "role": "conclusion", "optional": False},
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        # Check order
        hook_pos = result.find("Hook content.")
        problem_pos = result.find("Problem content.")
        conclusion_pos = result.find("Conclusion content.")

        assert hook_pos < problem_pos < conclusion_pos

    def test_skips_optional_sections(self):
        """Optional sections are not included."""
        section_drafts = {
            "problem": "Problem content.",
            "optional_deep_dive": "Optional content.",
        }
        plan = {
            "sections": [
                {"id": "problem", "title": "Problem", "role": "problem", "optional": False},
                {"id": "optional_deep_dive", "title": "Extra", "role": "deep_dive", "optional": True},
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        assert "Problem content." in result
        assert "Optional content." not in result

    def test_missing_section_content_skipped(self):
        """Sections with no content are skipped."""
        section_drafts = {"problem": "Problem content."}  # Missing "why"
        plan = {
            "sections": [
                {"id": "problem", "title": "Problem", "role": "problem", "optional": False},
                {"id": "why", "title": "Why", "role": "why", "optional": False},
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        assert "Problem content." in result
        assert "## Why" not in result  # Skipped because no content

    def test_uses_id_when_no_title(self):
        """Uses section ID (titleized) when title is None."""
        section_drafts = {"implementation_details": "Implementation content."}
        plan = {
            "sections": [
                {"id": "implementation_details", "title": None, "role": "implementation", "optional": False}
            ]
        }
        result = _combine_sections(section_drafts, plan, "Test Blog")

        assert "## Implementation Details" in result

    def test_h1_title_at_start(self):
        """Blog title is H1 at the start."""
        section_drafts = {"problem": "Content."}
        plan = {"sections": [{"id": "problem", "title": "P", "role": "problem", "optional": False}]}
        result = _combine_sections(section_drafts, plan, "My Great Blog")

        assert result.startswith("# My Great Blog")


class TestCalculateReadingTime:
    """Tests for _calculate_reading_time helper."""

    def test_minimum_one_minute(self):
        """Short text returns at least 1 minute."""
        assert _calculate_reading_time("Hello world") == 1

    def test_two_hundred_words(self):
        """200 words at 200 wpm = 1 minute."""
        text = " ".join(["word"] * 200)
        assert _calculate_reading_time(text) == 1

    def test_four_hundred_words(self):
        """400 words at 200 wpm = 2 minutes."""
        text = " ".join(["word"] * 400)
        assert _calculate_reading_time(text) == 2

    def test_custom_wpm(self):
        """Custom words per minute works."""
        text = " ".join(["word"] * 300)
        assert _calculate_reading_time(text, words_per_minute=100) == 3

    def test_rounding(self):
        """Reading time is rounded to nearest minute."""
        text = " ".join(["word"] * 250)  # 1.25 min at 200 wpm
        assert _calculate_reading_time(text) == 1


class TestCountWords:
    """Tests for _count_words helper."""

    def test_empty_string(self):
        """Empty string has 0 words."""
        assert _count_words("") == 0

    def test_single_word(self):
        """Single word counted correctly."""
        assert _count_words("hello") == 1

    def test_multiple_words(self):
        """Multiple words counted correctly."""
        assert _count_words("hello world foo bar") == 4

    def test_whitespace_variations(self):
        """Handles various whitespace."""
        assert _count_words("hello   world\nfoo\tbar") == 4


class TestFinalAssemblyNode:
    """Tests for final_assembly_node."""

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return {
            "job_id": "",  # No persistence for unit tests
            "title": "Test Blog Title",
            "plan": {
                "blog_title": "Test Blog Title",
                "sections": [
                    {"id": "hook", "title": None, "role": "hook", "optional": False},
                    {"id": "problem", "title": "The Problem", "role": "problem", "optional": False},
                    {"id": "conclusion", "title": "Conclusion", "role": "conclusion", "optional": False},
                ],
            },
            "section_drafts": {
                "hook": "This is an attention-grabbing hook.",
                "problem": "Here is the problem we're solving.",
                "conclusion": "In conclusion, here are your next steps.",
            },
        }

    @pytest.mark.asyncio
    async def test_combines_sections_successfully(self, sample_state):
        """Successfully combines all sections."""
        result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value
        assert "combined_draft" in result
        assert "final_markdown" in result
        assert "metadata" in result

        # Check combined content
        combined = result["combined_draft"]
        assert "# Test Blog Title" in combined
        assert "attention-grabbing hook" in combined
        assert "## The Problem" in combined
        assert "## Conclusion" in combined

    @pytest.mark.asyncio
    async def test_calculates_metadata(self, sample_state):
        """Calculates word count and reading time."""
        result = await final_assembly_node(sample_state)

        metadata = result["metadata"]
        assert "word_count" in metadata
        assert "reading_time_minutes" in metadata
        assert "section_count" in metadata
        assert metadata["word_count"] > 0
        assert metadata["reading_time_minutes"] >= 1
        assert metadata["section_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_drafts_fails(self):
        """Fails gracefully when no drafts exist."""
        state = {
            "plan": {"sections": []},
            "section_drafts": {},
        }
        result = await final_assembly_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_final_markdown_same_as_combined(self, sample_state):
        """In minimal version, final_markdown equals combined_draft."""
        result = await final_assembly_node(sample_state)

        assert result["final_markdown"] == result["combined_draft"]

    @pytest.mark.asyncio
    async def test_saves_to_job_directory(self, sample_state, tmp_path):
        """Saves files when job_id is provided."""
        from src.agent.state import JobManager

        # Create a real job
        job_manager = JobManager(base_dir=tmp_path)
        job_id = job_manager.create_job("Test Blog", "Test context")

        sample_state["job_id"] = job_id

        with patch("src.agent.nodes.JobManager") as MockJobManager:
            mock_instance = MagicMock()
            mock_instance.get_job_dir.return_value = tmp_path / "jobs" / job_id
            mock_instance.save_state = MagicMock()
            MockJobManager.return_value = mock_instance

            # Create directories
            (tmp_path / "jobs" / job_id / "drafts").mkdir(parents=True, exist_ok=True)

            result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value

    @pytest.mark.asyncio
    async def test_uses_title_from_plan(self, sample_state):
        """Uses blog_title from plan if available."""
        sample_state["plan"]["blog_title"] = "Plan Title"
        sample_state["title"] = "State Title"

        result = await final_assembly_node(sample_state)

        assert "# Plan Title" in result["combined_draft"]

    @pytest.mark.asyncio
    async def test_falls_back_to_state_title(self, sample_state):
        """Falls back to state title if plan has no blog_title."""
        del sample_state["plan"]["blog_title"]
        sample_state["title"] = "Fallback Title"

        result = await final_assembly_node(sample_state)

        assert "# Fallback Title" in result["combined_draft"]

    @pytest.mark.asyncio
    async def test_preserves_markdown_formatting(self, sample_state):
        """Preserves markdown formatting in sections."""
        sample_state["section_drafts"]["problem"] = """Here is **bold** text.

And a code block:

```python
print("hello")
```

And a list:
- Item 1
- Item 2
"""
        result = await final_assembly_node(sample_state)

        combined = result["combined_draft"]
        assert "**bold**" in combined
        assert "```python" in combined
        assert "- Item 1" in combined
