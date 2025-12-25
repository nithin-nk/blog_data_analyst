"""
Unit tests for the final assembly node and helper functions.

Tests cover:
- _combine_sections()
- _calculate_reading_time()
- _count_words()
- final_assembly_node() with mocked JobManager
- FinalCriticScore, TransitionFix, FinalCriticResult models
- _build_final_critic_prompt()
- _final_critic()
- _apply_transition_fixes()
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    _build_final_critic_prompt,
    _combine_sections,
    _calculate_reading_time,
    _count_words,
    final_assembly_node,
    MAX_FINAL_CRITIC_ITERATIONS,
)
from src.agent.state import (
    FinalCriticResult,
    FinalCriticScore,
    Phase,
    TransitionFix,
)


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
    """Tests for final_assembly_node (basic functionality with mocked critic)."""

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

    @pytest.fixture
    def mock_passing_critic(self):
        """Create a mock passing critic result."""
        return FinalCriticResult(
            scores=FinalCriticScore(
                coherence=9,
                voice_consistency=9,
                no_redundancy=9,
                narrative_arc=9,
                hook_effectiveness=9,
                conclusion_strength=9,
                overall_polish=9,
            ),
            overall_pass=True,
            transition_fixes=[],
            praise="Well done!",
            issues=[],
            reading_time_minutes=1,
            word_count=50,
        )

    @pytest.mark.asyncio
    async def test_combines_sections_successfully(self, sample_state, mock_passing_critic):
        """Successfully combines all sections."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
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
    async def test_calculates_metadata(self, sample_state, mock_passing_critic):
        """Calculates word count and reading time."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
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
    async def test_saves_to_job_directory(self, sample_state, tmp_path, mock_passing_critic):
        """Saves files when job_id is provided."""
        from src.agent.state import JobManager

        # Create a real job
        job_manager = JobManager(base_dir=tmp_path)
        job_id = job_manager.create_job("Test Blog", "Test context")

        sample_state["job_id"] = job_id

        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.JobManager") as MockJobManager, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_instance.get_job_dir.return_value = tmp_path / "jobs" / job_id
            mock_instance.save_state = MagicMock()
            MockJobManager.return_value = mock_instance

            # Create directories
            (tmp_path / "jobs" / job_id / "drafts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "jobs" / job_id / "feedback").mkdir(parents=True, exist_ok=True)

            result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value

    @pytest.mark.asyncio
    async def test_uses_title_from_plan(self, sample_state, mock_passing_critic):
        """Uses blog_title from plan if available."""
        sample_state["plan"]["blog_title"] = "Plan Title"
        sample_state["title"] = "State Title"

        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
            result = await final_assembly_node(sample_state)

        assert "# Plan Title" in result["combined_draft"]

    @pytest.mark.asyncio
    async def test_falls_back_to_state_title(self, sample_state, mock_passing_critic):
        """Falls back to state title if plan has no blog_title."""
        del sample_state["plan"]["blog_title"]
        sample_state["title"] = "Fallback Title"

        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
            result = await final_assembly_node(sample_state)

        assert "# Fallback Title" in result["combined_draft"]

    @pytest.mark.asyncio
    async def test_preserves_markdown_formatting(self, sample_state, mock_passing_critic):
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
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic
            mock_km.return_value = MagicMock()
            result = await final_assembly_node(sample_state)

        combined = result["combined_draft"]
        assert "**bold**" in combined
        assert "```python" in combined
        assert "- Item 1" in combined


# =============================================================================
# Final Critic Model Tests
# =============================================================================


class TestFinalCriticModels:
    """Tests for final critic Pydantic models."""

    def test_final_critic_score_valid(self):
        """FinalCriticScore accepts valid scores."""
        score = FinalCriticScore(
            coherence=8,
            voice_consistency=9,
            no_redundancy=7,
            narrative_arc=8,
            hook_effectiveness=9,
            conclusion_strength=8,
            overall_polish=8,
        )
        assert score.coherence == 8
        assert score.voice_consistency == 9

    def test_final_critic_score_bounds(self):
        """FinalCriticScore enforces 1-10 bounds."""
        with pytest.raises(ValueError):
            FinalCriticScore(
                coherence=0,  # Invalid: below 1
                voice_consistency=8,
                no_redundancy=8,
                narrative_arc=8,
                hook_effectiveness=8,
                conclusion_strength=8,
                overall_polish=8,
            )

        with pytest.raises(ValueError):
            FinalCriticScore(
                coherence=11,  # Invalid: above 10
                voice_consistency=8,
                no_redundancy=8,
                narrative_arc=8,
                hook_effectiveness=8,
                conclusion_strength=8,
                overall_polish=8,
            )

    def test_transition_fix_model(self):
        """TransitionFix has required fields."""
        fix = TransitionFix(
            between=["section_1", "section_2"],
            issue="Abrupt change in topic",
            suggestion="Add a connecting sentence",
        )
        assert fix.between == ["section_1", "section_2"]
        assert "Abrupt" in fix.issue
        assert "connecting" in fix.suggestion

    def test_final_critic_result_model(self):
        """FinalCriticResult combines all fields correctly."""
        scores = FinalCriticScore(
            coherence=9,
            voice_consistency=8,
            no_redundancy=8,
            narrative_arc=9,
            hook_effectiveness=8,
            conclusion_strength=9,
            overall_polish=8,
        )
        result = FinalCriticResult(
            scores=scores,
            overall_pass=True,
            transition_fixes=[],
            praise="Great coherence and narrative arc.",
            issues=[],
            reading_time_minutes=5,
            word_count=1000,
        )
        assert result.overall_pass is True
        assert result.word_count == 1000
        assert result.reading_time_minutes == 5

    def test_final_critic_result_with_fixes(self):
        """FinalCriticResult can contain transition fixes."""
        scores = FinalCriticScore(
            coherence=6,
            voice_consistency=8,
            no_redundancy=8,
            narrative_arc=7,
            hook_effectiveness=8,
            conclusion_strength=8,
            overall_polish=8,
        )
        fix = TransitionFix(
            between=["hook", "problem"],
            issue="Missing connection",
            suggestion="Add bridging text",
        )
        result = FinalCriticResult(
            scores=scores,
            overall_pass=False,
            transition_fixes=[fix],
            praise="Good voice.",
            issues=["Coherence needs improvement"],
            reading_time_minutes=3,
            word_count=600,
        )
        assert result.overall_pass is False
        assert len(result.transition_fixes) == 1
        assert result.transition_fixes[0].between == ["hook", "problem"]


class TestBuildFinalCriticPrompt:
    """Tests for _build_final_critic_prompt helper."""

    def test_prompt_includes_blog_title(self):
        """Prompt includes blog title."""
        prompt = _build_final_critic_prompt(
            draft="# Test Blog\n\nContent here.",
            plan={"sections": []},
            blog_title="Test Blog",
        )
        assert "Test Blog" in prompt

    def test_prompt_includes_section_ids(self):
        """Prompt includes section IDs from plan."""
        plan = {
            "sections": [
                {"id": "hook", "role": "hook", "optional": False},
                {"id": "problem", "role": "problem", "optional": False},
            ]
        }
        prompt = _build_final_critic_prompt(
            draft="Content",
            plan=plan,
            blog_title="Test",
        )
        assert "hook" in prompt
        assert "problem" in prompt

    def test_prompt_excludes_optional_sections(self):
        """Prompt excludes optional sections."""
        plan = {
            "sections": [
                {"id": "required", "role": "problem", "optional": False},
                {"id": "optional_extra", "role": "deep_dive", "optional": True},
            ]
        }
        prompt = _build_final_critic_prompt(
            draft="Content",
            plan=plan,
            blog_title="Test",
        )
        assert "required" in prompt
        assert "optional_extra" not in prompt

    def test_prompt_includes_word_count(self):
        """Prompt includes actual word count."""
        prompt = _build_final_critic_prompt(
            draft="one two three four five",
            plan={"sections": []},
            blog_title="Test",
        )
        assert "5" in prompt  # Word count

    def test_prompt_includes_all_dimensions(self):
        """Prompt includes all 7 evaluation dimensions."""
        prompt = _build_final_critic_prompt(
            draft="Content",
            plan={"sections": []},
            blog_title="Test",
        )
        assert "coherence" in prompt.lower()
        assert "voice_consistency" in prompt.lower()
        assert "no_redundancy" in prompt.lower()
        assert "narrative_arc" in prompt.lower()
        assert "hook_effectiveness" in prompt.lower()
        assert "conclusion_strength" in prompt.lower()
        assert "overall_polish" in prompt.lower()


class TestFinalCriticFunction:
    """Tests for _final_critic function with mocked LLM."""

    @pytest.fixture
    def mock_key_manager(self):
        """Create mock KeyManager."""
        manager = MagicMock()
        manager.get_current_key.return_value = "test-api-key"
        manager.record_usage = MagicMock()
        manager.rotate_key = MagicMock()
        return manager

    @pytest.fixture
    def mock_critic_result(self):
        """Create mock FinalCriticResult."""
        return FinalCriticResult(
            scores=FinalCriticScore(
                coherence=9,
                voice_consistency=8,
                no_redundancy=8,
                narrative_arc=9,
                hook_effectiveness=8,
                conclusion_strength=9,
                overall_polish=8,
            ),
            overall_pass=True,
            transition_fixes=[],
            praise="Well structured blog.",
            issues=[],
            reading_time_minutes=5,
            word_count=1000,
        )

    @pytest.mark.asyncio
    async def test_final_critic_returns_result(self, mock_key_manager, mock_critic_result):
        """_final_critic returns FinalCriticResult."""
        from src.agent.nodes import _final_critic

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_critic_result
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _final_critic(
                draft="# Test\n\nContent",
                plan={"sections": []},
                blog_title="Test",
                key_manager=mock_key_manager,
            )

        assert isinstance(result, FinalCriticResult)
        assert result.overall_pass is True

    @pytest.mark.asyncio
    async def test_final_critic_handles_failure(self, mock_key_manager):
        """_final_critic returns default result on complete failure."""
        from src.agent.nodes import _final_critic

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API error")
            mock_llm_class.return_value = mock_llm

            result = await _final_critic(
                draft="# Test\n\nContent here",
                plan={"sections": []},
                blog_title="Test",
                key_manager=mock_key_manager,
                max_retries=1,
            )

        # Should return default passing result
        assert isinstance(result, FinalCriticResult)
        assert result.overall_pass is True
        assert "Final critic evaluation failed" in result.issues


class TestApplyTransitionFixes:
    """Tests for _apply_transition_fixes function."""

    @pytest.fixture
    def mock_key_manager(self):
        """Create mock KeyManager."""
        manager = MagicMock()
        manager.get_current_key.return_value = "test-api-key"
        manager.record_usage = MagicMock()
        manager.rotate_key = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_returns_original_when_no_fixes(self, mock_key_manager):
        """Returns original draft when no fixes needed."""
        from src.agent.nodes import _apply_transition_fixes

        draft = "# Test\n\nOriginal content"
        result = await _apply_transition_fixes(
            draft=draft,
            fixes=[],
            key_manager=mock_key_manager,
        )
        assert result == draft

    @pytest.mark.asyncio
    async def test_applies_transition_fixes(self, mock_key_manager):
        """Applies fixes and returns refined draft."""
        from src.agent.nodes import _apply_transition_fixes

        fixes = [
            TransitionFix(
                between=["hook", "problem"],
                issue="Abrupt transition",
                suggestion="Add connecting sentence",
            )
        ]

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_result = MagicMock()
            mock_result.content = "# Test\n\nImproved content with better transitions"
            mock_llm.invoke.return_value = mock_result
            mock_llm_class.return_value = mock_llm

            result = await _apply_transition_fixes(
                draft="# Test\n\nOriginal content",
                fixes=fixes,
                key_manager=mock_key_manager,
            )

        assert "Improved content" in result

    @pytest.mark.asyncio
    async def test_handles_failure_returns_original(self, mock_key_manager):
        """Returns original draft on failure."""
        from src.agent.nodes import _apply_transition_fixes

        fixes = [
            TransitionFix(
                between=["a", "b"],
                issue="Issue",
                suggestion="Fix",
            )
        ]
        original_draft = "# Original\n\nContent"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("API error")
            mock_llm_class.return_value = mock_llm

            result = await _apply_transition_fixes(
                draft=original_draft,
                fixes=fixes,
                key_manager=mock_key_manager,
                max_retries=1,
            )

        assert result == original_draft


class TestFinalAssemblyNodeWithCritic:
    """Tests for final_assembly_node with critic loop."""

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

    @pytest.fixture
    def mock_passing_critic_result(self):
        """Create a passing critic result."""
        return FinalCriticResult(
            scores=FinalCriticScore(
                coherence=9,
                voice_consistency=9,
                no_redundancy=9,
                narrative_arc=9,
                hook_effectiveness=9,
                conclusion_strength=9,
                overall_polish=9,
            ),
            overall_pass=True,
            transition_fixes=[],
            praise="Excellent blog!",
            issues=[],
            reading_time_minutes=1,
            word_count=50,
        )

    @pytest.fixture
    def mock_failing_critic_result(self):
        """Create a failing critic result with fixes."""
        return FinalCriticResult(
            scores=FinalCriticScore(
                coherence=6,
                voice_consistency=8,
                no_redundancy=8,
                narrative_arc=6,
                hook_effectiveness=8,
                conclusion_strength=8,
                overall_polish=8,
            ),
            overall_pass=False,
            transition_fixes=[
                TransitionFix(
                    between=["hook", "problem"],
                    issue="Weak transition",
                    suggestion="Add bridge",
                )
            ],
            praise="Good voice.",
            issues=["Improve coherence"],
            reading_time_minutes=1,
            word_count=50,
        )

    @pytest.mark.asyncio
    async def test_final_assembly_with_passing_critic(self, sample_state, mock_passing_critic_result):
        """Assembly node runs critic and passes on first iteration."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic_result
            mock_km.return_value = MagicMock()

            result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value
        assert "final_review" in result
        assert result["final_review"]["overall_pass"] is True
        # Should only call critic once since it passed
        assert mock_critic.call_count == 1

    @pytest.mark.asyncio
    async def test_final_assembly_applies_fixes_on_failure(
        self, sample_state, mock_failing_critic_result, mock_passing_critic_result
    ):
        """Assembly node applies fixes when critic fails."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes._apply_transition_fixes") as mock_fixes, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            # First call fails, second call passes
            mock_critic.side_effect = [mock_failing_critic_result, mock_passing_critic_result]
            mock_fixes.return_value = "Improved content"
            mock_km.return_value = MagicMock()

            result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value
        # Should call critic twice
        assert mock_critic.call_count == 2
        # Should call fixes once
        assert mock_fixes.call_count == 1

    @pytest.mark.asyncio
    async def test_final_assembly_respects_max_iterations(
        self, sample_state, mock_failing_critic_result
    ):
        """Assembly node stops after max iterations."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes._apply_transition_fixes") as mock_fixes, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            # Always fails
            mock_critic.return_value = mock_failing_critic_result
            mock_fixes.return_value = "Still failing content"
            mock_km.return_value = MagicMock()

            result = await final_assembly_node(sample_state)

        assert result["current_phase"] == Phase.REVIEWING.value
        # Should call critic MAX_FINAL_CRITIC_ITERATIONS times
        assert mock_critic.call_count == MAX_FINAL_CRITIC_ITERATIONS
        # Should only apply fixes on iterations before the last
        assert mock_fixes.call_count == MAX_FINAL_CRITIC_ITERATIONS - 1

    @pytest.mark.asyncio
    async def test_final_assembly_includes_critic_scores_in_metadata(
        self, sample_state, mock_passing_critic_result
    ):
        """Metadata includes critic scores."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.return_value = mock_passing_critic_result
            mock_km.return_value = MagicMock()

            result = await final_assembly_node(sample_state)

        metadata = result["metadata"]
        assert "final_critic_scores" in metadata
        assert "final_critic_avg_score" in metadata
        assert "final_critic_pass" in metadata
        assert metadata["final_critic_pass"] is True

    @pytest.mark.asyncio
    async def test_final_markdown_may_differ_from_combined(
        self, sample_state, mock_failing_critic_result, mock_passing_critic_result
    ):
        """final_markdown can differ from combined_draft after fixes."""
        with patch("src.agent.nodes._final_critic") as mock_critic, \
             patch("src.agent.nodes._apply_transition_fixes") as mock_fixes, \
             patch("src.agent.nodes.KeyManager.from_env") as mock_km:
            mock_critic.side_effect = [mock_failing_critic_result, mock_passing_critic_result]
            mock_fixes.return_value = "# Test Blog Title\n\nImproved content"
            mock_km.return_value = MagicMock()

            result = await final_assembly_node(sample_state)

        # combined_draft is original, final_markdown is improved
        assert "Improved content" in result["final_markdown"]
        assert result["combined_draft"] != result["final_markdown"]
