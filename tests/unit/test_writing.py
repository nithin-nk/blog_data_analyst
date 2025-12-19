"""
Unit tests for the write section node and helper functions.

Tests cover:
- _format_sources_for_prompt()
- _get_previous_sections_text()
- _build_writer_prompt()
- write_section_node() with mocked LLM
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    _format_sources_for_prompt,
    _get_previous_sections_text,
    _build_writer_prompt,
    write_section_node,
)
from src.agent.state import Phase


class TestFormatSourcesForPrompt:
    """Tests for _format_sources_for_prompt helper."""

    def test_empty_sources(self):
        """Returns fallback message when no sources provided."""
        result = _format_sources_for_prompt([])
        assert "No research sources available" in result
        assert "Write based on your knowledge" in result

    def test_single_source(self):
        """Formats a single source correctly."""
        sources = [
            {
                "title": "Redis Vector Search Guide",
                "url": "https://redis.io/docs/vector",
                "content": "Redis supports vector similarity search...",
            }
        ]
        result = _format_sources_for_prompt(sources)

        assert "Source 1:" in result
        assert "Redis Vector Search Guide" in result
        assert "https://redis.io/docs/vector" in result
        assert "Redis supports vector similarity search" in result

    def test_multiple_sources(self):
        """Formats multiple sources with separators."""
        sources = [
            {"title": "Source A", "url": "https://a.com", "content": "Content A"},
            {"title": "Source B", "url": "https://b.com", "content": "Content B"},
        ]
        result = _format_sources_for_prompt(sources)

        assert "Source 1:" in result
        assert "Source 2:" in result
        assert "---" in result  # Separator between sources

    def test_max_sources_limit(self):
        """Respects max_sources parameter."""
        sources = [
            {"title": f"Source {i}", "url": f"https://{i}.com", "content": f"Content {i}"}
            for i in range(10)
        ]
        result = _format_sources_for_prompt(sources, max_sources=3)

        assert "Source 1:" in result
        assert "Source 2:" in result
        assert "Source 3:" in result
        assert "Source 4:" not in result

    def test_content_truncation(self):
        """Truncates very long content."""
        long_content = "x" * 5000  # 5000 chars
        sources = [{"title": "Long", "url": "https://long.com", "content": long_content}]
        result = _format_sources_for_prompt(sources)

        # Content should be truncated to ~2000 chars
        assert len(result) < 3000


class TestGetPreviousSectionsText:
    """Tests for _get_previous_sections_text helper."""

    def test_empty_drafts(self):
        """Returns empty string when no drafts exist."""
        result = _get_previous_sections_text({}, [])
        assert result == ""

    def test_no_matching_sections(self):
        """Returns empty when drafts don't match sections."""
        drafts = {"hook": "Hook content"}
        sections = [{"id": "problem", "title": "The Problem"}]
        result = _get_previous_sections_text(drafts, sections)
        assert result == ""

    def test_single_previous_section(self):
        """Formats a single previous section correctly."""
        drafts = {"hook": "This is the hook content."}
        sections = [{"id": "hook", "title": "Hook"}]
        result = _get_previous_sections_text(drafts, sections)

        assert "## Hook" in result
        assert "This is the hook content." in result

    def test_multiple_previous_sections(self):
        """Formats multiple sections in order."""
        drafts = {
            "hook": "Hook content here.",
            "problem": "Problem content here.",
        }
        sections = [
            {"id": "hook", "title": "Hook"},
            {"id": "problem", "title": "The Problem"},
        ]
        result = _get_previous_sections_text(drafts, sections)

        assert "## Hook" in result
        assert "## The Problem" in result
        assert result.index("Hook") < result.index("Problem")

    def test_uses_role_when_no_title(self):
        """Falls back to role when title is None."""
        drafts = {"hook": "Hook content."}
        sections = [{"id": "hook", "role": "hook", "title": None}]
        result = _get_previous_sections_text(drafts, sections)

        assert "## hook" in result


class TestBuildWriterPrompt:
    """Tests for _build_writer_prompt helper."""

    def test_basic_prompt_structure(self):
        """Prompt contains all required sections."""
        section = {
            "id": "problem",
            "title": "The Problem",
            "role": "problem",
            "target_words": 300,
            "needs_code": False,
            "needs_diagram": False,
        }
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="Be direct.",
            blog_title="Test Blog",
        )

        assert "Test Blog" in prompt
        assert "The Problem" in prompt
        assert "300 words" in prompt
        assert "Be direct." in prompt
        assert "PROBLEM section" in prompt

    def test_hook_role_instructions(self):
        """Hook role gets specific instructions."""
        section = {"id": "hook", "role": "hook", "target_words": 100}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "HOOK section" in prompt
        assert "grab the reader's attention" in prompt

    def test_deep_dive_role_instructions(self):
        """Deep dive role gets implementation instructions."""
        section = {"id": "impl", "role": "deep_dive", "target_words": 500}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "DEEP DIVE" in prompt
        assert "technical depth" in prompt

    def test_conclusion_role_instructions(self):
        """Conclusion role gets takeaway instructions."""
        section = {"id": "conclusion", "role": "conclusion", "target_words": 200}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "CONCLUSION section" in prompt
        assert "actionable takeaways" in prompt

    def test_needs_code_requirement(self):
        """Code requirement is included when needs_code=True."""
        section = {"id": "impl", "role": "implementation", "needs_code": True}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "MUST include working code examples" in prompt

    def test_needs_diagram_requirement(self):
        """Diagram requirement is included when needs_diagram=True."""
        section = {"id": "arch", "role": "deep_dive", "needs_diagram": True}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "MUST include a mermaid diagram" in prompt

    def test_sources_included(self):
        """Sources are formatted and included in prompt."""
        section = {"id": "test", "role": "problem"}
        sources = [{"title": "Test Source", "url": "https://test.com", "content": "Source content"}]
        prompt = _build_writer_prompt(
            section=section,
            sources=sources,
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "Test Source" in prompt
        assert "Source content" in prompt

    def test_previous_sections_included(self):
        """Previous sections are included for context."""
        section = {"id": "problem", "role": "problem"}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="## Hook\n\nPrevious hook content here.",
            style_guide="",
            blog_title="Test",
        )

        assert "Previous hook content here" in prompt

    def test_first_section_message(self):
        """First section shows appropriate message."""
        section = {"id": "hook", "role": "hook"}
        prompt = _build_writer_prompt(
            section=section,
            sources=[],
            previous_sections_text="",
            style_guide="",
            blog_title="Test",
        )

        assert "This is the first section" in prompt


class TestWriteSectionNode:
    """Tests for write_section_node with mocked LLM."""

    @pytest.fixture
    def mock_key_manager(self):
        """Create a mock KeyManager."""
        manager = MagicMock()
        manager.get_best_key.return_value = "test-api-key"
        manager.record_usage = MagicMock()
        return manager

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return {
            "job_id": "test-job",
            "title": "Test Blog",
            "plan": {
                "blog_title": "Test Blog",
                "sections": [
                    {"id": "hook", "title": None, "role": "hook", "target_words": 100, "optional": False},
                    {"id": "problem", "title": "The Problem", "role": "problem", "target_words": 200, "optional": False},
                    {"id": "optional_section", "title": "Extra", "role": "deep_dive", "optional": True},
                ],
            },
            "validated_sources": {
                "hook": [{"title": "Source", "url": "https://test.com", "content": "Test content"}],
                "problem": [],
            },
            "current_section_index": 0,
            "section_drafts": {},
        }

    @pytest.mark.asyncio
    async def test_writes_first_section(self, sample_state, mock_key_manager):
        """Successfully writes the first section."""
        mock_response = MagicMock()
        mock_response.content = "Generated hook content here."

        with patch("src.agent.nodes.KeyManager") as MockKeyManager, \
             patch("src.agent.nodes.ChatGoogleGenerativeAI") as MockLLM, \
             patch("src.agent.nodes.JobManager"):

            MockKeyManager.from_env.return_value = mock_key_manager
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            result = await write_section_node(sample_state)

        assert result["current_section_index"] == 1
        assert "hook" in result["section_drafts"]
        assert result["section_drafts"]["hook"] == "Generated hook content here."
        assert result["current_phase"] == Phase.WRITING.value

    @pytest.mark.asyncio
    async def test_increments_section_index(self, sample_state, mock_key_manager):
        """Increments section index after writing."""
        sample_state["current_section_index"] = 0

        mock_response = MagicMock()
        mock_response.content = "Content"

        with patch("src.agent.nodes.KeyManager") as MockKeyManager, \
             patch("src.agent.nodes.ChatGoogleGenerativeAI") as MockLLM, \
             patch("src.agent.nodes.JobManager"):

            MockKeyManager.from_env.return_value = mock_key_manager
            MockLLM.return_value.invoke.return_value = mock_response

            result = await write_section_node(sample_state)

        assert result["current_section_index"] == 1

    @pytest.mark.asyncio
    async def test_skips_optional_sections(self, sample_state, mock_key_manager):
        """Only writes required (non-optional) sections."""
        # Set index to after all required sections
        sample_state["current_section_index"] = 2  # Past hook and problem

        result = await write_section_node(sample_state)

        # Should transition to assembly phase
        assert result["current_phase"] == Phase.ASSEMBLING.value

    @pytest.mark.asyncio
    async def test_preserves_previous_drafts(self, sample_state, mock_key_manager):
        """Preserves previously written section drafts."""
        sample_state["current_section_index"] = 1
        sample_state["section_drafts"] = {"hook": "Existing hook content"}

        mock_response = MagicMock()
        mock_response.content = "New problem content"

        with patch("src.agent.nodes.KeyManager") as MockKeyManager, \
             patch("src.agent.nodes.ChatGoogleGenerativeAI") as MockLLM, \
             patch("src.agent.nodes.JobManager"):

            MockKeyManager.from_env.return_value = mock_key_manager
            MockLLM.return_value.invoke.return_value = mock_response

            result = await write_section_node(sample_state)

        assert "hook" in result["section_drafts"]
        assert result["section_drafts"]["hook"] == "Existing hook content"
        assert result["section_drafts"]["problem"] == "New problem content"

    @pytest.mark.asyncio
    async def test_handles_llm_error(self, sample_state, mock_key_manager):
        """Handles LLM errors gracefully."""
        with patch("src.agent.nodes.KeyManager") as MockKeyManager, \
             patch("src.agent.nodes.ChatGoogleGenerativeAI") as MockLLM, \
             patch("src.agent.nodes.JobManager"):

            MockKeyManager.from_env.return_value = mock_key_manager
            MockLLM.return_value.invoke.side_effect = Exception("API Error")

            result = await write_section_node(sample_state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_all_sections_complete_transitions_to_assembly(self, sample_state, mock_key_manager):
        """Transitions to assembly when all sections are written."""
        # Simulate all required sections written
        sample_state["current_section_index"] = 2  # 2 required sections
        sample_state["section_drafts"] = {
            "hook": "Hook content",
            "problem": "Problem content",
        }

        result = await write_section_node(sample_state)

        assert result["current_phase"] == Phase.ASSEMBLING.value

    @pytest.mark.asyncio
    async def test_empty_plan_fails_gracefully(self, mock_key_manager):
        """Handles empty plan gracefully."""
        state = {
            "plan": {"sections": []},
            "current_section_index": 0,
            "section_drafts": {},
        }

        result = await write_section_node(state)

        # Should transition to assembly (nothing to write)
        assert result["current_phase"] == Phase.ASSEMBLING.value
