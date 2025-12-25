"""
Unit tests for preview validation with LLM-based uniqueness checking and feedback.

All LLM calls are mocked for fast, deterministic testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    _build_replanning_feedback_prompt,
    _build_uniqueness_prompt,
    _check_sections_uniqueness_llm,
    _generate_replanning_feedback_llm,
    preview_validation_node,
)
from src.agent.graph import preview_validation_router
from src.agent.state import (
    Phase,
    ReplanningFeedback,
    SectionFeasibilityScore,
    SectionSuggestion,
    UniquenessAnalysisResult,
    UniquenessCheck,
)
from src.agent.key_manager import KeyManager


@pytest.fixture
def sample_sections():
    """Sample blog sections for testing."""
    return [
        {
            "id": "section_1",
            "title": "Introduction to Semantic Caching",
            "role": "hook",
            "gap_addressed": "practical_implementation",
            "gap_justification": "Shows how to implement, not just theory",
            "search_queries": ["semantic caching tutorial", "implement semantic cache"],
        },
        {
            "id": "section_2",
            "title": "Why Traditional Caching Fails",
            "role": "problem",
            "gap_addressed": "tradeoffs_analysis",
            "gap_justification": "Explains limitations of existing approaches",
            "search_queries": ["cache limitations", "traditional caching problems"],
        },
    ]


@pytest.fixture
def sample_content_strategy():
    """Sample content strategy with gaps and articles."""
    return {
        "gaps": [
            {
                "gap": "practical_implementation",
                "description": "Step-by-step implementation guide",
            },
            {
                "gap": "tradeoffs_analysis",
                "description": "Honest analysis of when NOT to use semantic caching",
            },
        ],
        "analyzed_articles": [
            {
                "title": "Semantic Caching Overview",
                "url": "https://example.com/semantic-caching",
                "main_angle": "High-level concept explanation",
                "key_points": ["What is semantic caching", "Basic benefits"],
                "strengths": ["Clear definitions"],
                "weaknesses": ["No implementation details"],
            },
        ],
    }


@pytest.fixture
def mock_key_manager():
    """Mock KeyManager."""
    manager = MagicMock()
    manager.get_best_key.return_value = "test-api-key"
    manager.record_usage = MagicMock()
    manager.mark_rate_limited = MagicMock()
    manager.get_next_key = MagicMock(return_value=None)
    return manager


class TestUniquenessPromptBuilding:
    """Test uniqueness prompt construction."""

    def test_build_uniqueness_prompt(self, sample_sections, sample_content_strategy):
        """Test that uniqueness prompt includes all necessary context."""
        prompt = _build_uniqueness_prompt(sample_sections, sample_content_strategy)

        # Check sections are included
        assert "section_1" in prompt
        assert "Introduction to Semantic Caching" in prompt
        assert "practical_implementation" in prompt

        # Check articles are included
        assert "Semantic Caching Overview" in prompt
        assert "https://example.com/semantic-caching" in prompt

        # Check instructions are clear
        assert "uniqueness" in prompt.lower()
        assert "overlap" in prompt.lower()
        assert "0-100%" in prompt or "percentage" in prompt.lower()


class TestUniquenessLLMCall:
    """Test LLM-based uniqueness checking."""

    @pytest.mark.asyncio
    async def test_check_uniqueness_llm_all_unique(
        self, sample_sections, sample_content_strategy, mock_key_manager
    ):
        """Test uniqueness check when all sections are unique."""
        # Mock LLM response
        mock_result = UniquenessAnalysisResult(
            uniqueness_checks=[
                UniquenessCheck(
                    section_id="section_1",
                    section_title="Introduction to Semantic Caching",
                    overlap_percentage=15.0,
                    overlapping_articles=[],
                    is_unique=True,
                    concerns="Low overlap, addresses practical implementation gap",
                ),
                UniquenessCheck(
                    section_id="section_2",
                    section_title="Why Traditional Caching Fails",
                    overlap_percentage=25.0,
                    overlapping_articles=[],
                    is_unique=True,
                    concerns="Moderate overlap but focuses on tradeoffs",
                ),
            ],
            overall_assessment="All sections are sufficiently unique",
        )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_result
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _check_sections_uniqueness_llm(
                sample_sections, sample_content_strategy, mock_key_manager
            )

            # Verify result
            assert len(result.uniqueness_checks) == 2
            assert all(check.is_unique for check in result.uniqueness_checks)
            assert result.overall_assessment == "All sections are sufficiently unique"

            # Verify LLM was called correctly
            mock_llm_class.assert_called_once()
            assert mock_llm_class.call_args[1]["model"] == "gemini-2.5-flash"
            assert mock_llm_class.call_args[1]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_check_uniqueness_llm_with_duplicates(
        self, sample_sections, sample_content_strategy, mock_key_manager
    ):
        """Test uniqueness check when some sections are too similar."""
        # Mock LLM response with duplicates
        mock_result = UniquenessAnalysisResult(
            uniqueness_checks=[
                UniquenessCheck(
                    section_id="section_1",
                    section_title="Introduction to Semantic Caching",
                    overlap_percentage=85.0,
                    overlapping_articles=["https://example.com/semantic-caching"],
                    is_unique=False,
                    concerns="Very high overlap with example.com article",
                ),
                UniquenessCheck(
                    section_id="section_2",
                    section_title="Why Traditional Caching Fails",
                    overlap_percentage=30.0,
                    overlapping_articles=[],
                    is_unique=True,
                    concerns="Acceptable overlap, unique angle on tradeoffs",
                ),
            ],
            overall_assessment="Section 1 has high overlap and should be revised",
        )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_result
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _check_sections_uniqueness_llm(
                sample_sections, sample_content_strategy, mock_key_manager
            )

            # Verify result
            assert len(result.uniqueness_checks) == 2
            assert not result.uniqueness_checks[0].is_unique
            assert result.uniqueness_checks[0].overlap_percentage == 85.0
            assert result.uniqueness_checks[1].is_unique

    @pytest.mark.asyncio
    async def test_check_uniqueness_llm_retry_on_error(
        self, sample_sections, sample_content_strategy, mock_key_manager
    ):
        """Test that uniqueness check retries on failure."""
        mock_result = UniquenessAnalysisResult(
            uniqueness_checks=[
                UniquenessCheck(
                    section_id="section_1",
                    section_title="Test Section",
                    overlap_percentage=20.0,
                    overlapping_articles=[],
                    is_unique=True,
                    concerns="Low overlap",
                )
            ],
            overall_assessment="Unique",
        )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()

            # Fail twice, then succeed
            mock_structured.invoke.side_effect = [
                Exception("Rate limit"),
                Exception("Temporary error"),
                mock_result,
            ]

            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _check_sections_uniqueness_llm(
                sample_sections, sample_content_strategy, mock_key_manager, max_retries=3
            )

            # Should succeed after retries
            assert result == mock_result
            assert mock_structured.invoke.call_count == 3


class TestReplanningFeedbackPromptBuilding:
    """Test replanning feedback prompt construction."""

    def test_build_replanning_feedback_prompt(
        self, sample_sections, sample_content_strategy
    ):
        """Test that replanning feedback prompt includes validation failures."""
        plan = {"sections": sample_sections, "topic": "Semantic Caching", "target_length": "medium"}

        weak_sections = [
            SectionFeasibilityScore(
                section_id="section_1",
                sample_queries_tested=["semantic caching", "implement cache"],
                sources_found=2,
                snippet_quality="poor",
                information_availability=35,
                is_feasible=False,
                concerns=["Low search results", "Limited information online"],
            )
        ]

        non_unique = [
            UniquenessCheck(
                section_id="section_2",
                section_title="Why Traditional Caching Fails",
                overlap_percentage=80.0,
                overlapping_articles=["https://example.com/semantic-caching"],
                is_unique=False,
                concerns="High overlap with existing article",
            )
        ]

        prompt = _build_replanning_feedback_prompt(
            plan,
            sample_content_strategy,
            gap_validation=None,
            weak_sections=weak_sections,
            non_unique_sections=non_unique,
        )

        # Check plan context
        assert "Semantic Caching" in prompt
        assert "medium" in prompt

        # Check gaps are included
        assert "practical_implementation" in prompt
        assert "tradeoffs_analysis" in prompt

        # Check weak sections
        assert "section_1" in prompt
        assert "35" in prompt or "information" in prompt.lower()

        # Check non-unique sections
        assert "section_2" in prompt
        assert "80" in prompt or "overlap" in prompt.lower()

        # Check instructions for concrete suggestions
        assert "concrete" in prompt.lower() or "specific" in prompt.lower()


class TestReplanningFeedbackLLMCall:
    """Test LLM-based replanning feedback generation."""

    @pytest.mark.asyncio
    async def test_generate_replanning_feedback_llm(
        self, sample_sections, sample_content_strategy, mock_key_manager
    ):
        """Test feedback generation with concrete suggestions."""
        plan = {"sections": sample_sections, "topic": "Semantic Caching"}

        weak_sections = [
            SectionFeasibilityScore(
                section_id="section_1",
                sample_queries_tested=["semantic caching tutorial"],
                sources_found=3,
                snippet_quality="weak",
                information_availability=40,
                is_feasible=False,
                concerns=["Insufficient information"],
            )
        ]

        # Mock LLM response
        mock_result = ReplanningFeedback(
            summary="Section 1 lacks available information and should be revised",
            section_suggestions=[
                SectionSuggestion(
                    section_id="section_1",
                    issue="Low information availability (40/100)",
                    suggested_title="Real-World Semantic Caching Examples",
                    suggested_angle="Focus on case studies instead of implementation details",
                    alternative_gap="practical_implementation",
                )
            ],
            general_guidance="Focus on topics with more available research and examples",
        )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.ainvoke = AsyncMock(return_value=mock_result)
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_replanning_feedback_llm(
                plan,
                sample_content_strategy,
                mock_key_manager,
                weak_sections=weak_sections,
            )

            # Verify result
            assert result.summary == "Section 1 lacks available information and should be revised"
            assert len(result.section_suggestions) == 1
            assert result.section_suggestions[0].section_id == "section_1"
            assert "Real-World" in result.section_suggestions[0].suggested_title
            assert result.general_guidance is not None

            # Verify LLM was called
            mock_llm_class.assert_called_once()
            assert mock_llm_class.call_args[1]["model"] == "gemini-2.5-flash"


class TestPreviewValidationNode:
    """Test preview validation node."""

    @pytest.mark.asyncio
    async def test_preview_validation_all_pass(self, mock_key_manager):
        """Test validation node when all sections pass."""
        state = {
            "plan": {
                "sections": [
                    {
                        "id": "section_1",
                        "title": "Test Section",
                        "gap_addressed": "gap1",
                        "search_queries": ["query1"],
                    }
                ],
            },
            "content_strategy": {
                "gaps": [{"gap": "gap1", "description": "Test gap"}],
                "analyzed_articles": [],
            },
            "planning_iteration": 0,
            "key_manager": mock_key_manager,
        }

        # Mock feasibility and uniqueness checks
        with patch("src.agent.nodes._test_section_feasibility_parallel") as mock_feasibility:
            with patch("src.agent.nodes._check_sections_uniqueness_llm") as mock_uniqueness:
                mock_feasibility.return_value = [
                    SectionFeasibilityScore(
                        section_id="section_1",
                        sample_queries_tested=["query1"],
                        sources_found=10,
                        snippet_quality="excellent",
                        information_availability=85,
                        is_feasible=True,
                        concerns=[],
                    )
                ]

                mock_uniqueness.return_value = UniquenessAnalysisResult(
                    uniqueness_checks=[
                        UniquenessCheck(
                            section_id="section_1",
                            section_title="Test Section",
                            overlap_percentage=20.0,
                            overlapping_articles=[],
                            is_unique=True,
                            concerns="",
                        )
                    ],
                    overall_assessment="All unique",
                )

                result = await preview_validation_node(state)

                # Should advance to SECTION_SELECTION
                assert result["current_phase"] == Phase.SECTION_SELECTION.value
                assert result["preview_validation_result"]["all_sections_pass"] is True

                # Should have scratchpad entry
                assert "preview_validation_scratchpad" in result
                assert len(result["preview_validation_scratchpad"]) == 1
                scratchpad_entry = result["preview_validation_scratchpad"][0]
                assert scratchpad_entry["iteration"] == 0
                assert scratchpad_entry["passed"] is True
                assert "plan_snapshot" in scratchpad_entry
                assert "feasibility_scores" in scratchpad_entry
                assert "uniqueness_checks" in scratchpad_entry

    @pytest.mark.asyncio
    async def test_preview_validation_trigger_replanning(self, mock_key_manager):
        """Test validation node when sections fail and replanning is triggered."""
        state = {
            "plan": {
                "sections": [
                    {
                        "id": "section_1",
                        "title": "Test Section",
                        "gap_addressed": "gap1",
                        "search_queries": ["query1"],
                    }
                ],
                "topic": "Test Topic",
            },
            "content_strategy": {
                "gaps": [{"gap": "gap1", "description": "Test gap"}],
                "analyzed_articles": [],
            },
            "planning_iteration": 0,
            "key_manager": mock_key_manager,
        }

        # Mock feasibility check to fail
        with patch("src.agent.nodes._test_section_feasibility_parallel") as mock_feasibility:
            with patch("src.agent.nodes._check_sections_uniqueness_llm") as mock_uniqueness:
                with patch(
                    "src.agent.nodes._generate_replanning_feedback_llm"
                ) as mock_feedback:
                    mock_feasibility.return_value = [
                        SectionFeasibilityScore(
                            section_id="section_1",
                            sample_queries_tested=["query1"],
                            sources_found=2,
                            snippet_quality="poor",
                            information_availability=30,
                            is_feasible=False,
                            concerns=["Low info"],
                        )
                    ]

                    mock_uniqueness.return_value = UniquenessAnalysisResult(
                        uniqueness_checks=[
                            UniquenessCheck(
                            section_id="section_1",
                            section_title="Test Section",
                            overlap_percentage=20.0,
                            overlapping_articles=[],
                            is_unique=True,
                            concerns="",
                        )
                        ],
                        overall_assessment="",
                    )

                    mock_feedback.return_value = ReplanningFeedback(
                        summary="Section 1 failed",
                        section_suggestions=[
                            SectionSuggestion(
                                section_id="section_1",
                                issue="Low info",
                                suggested_title="Alternative",
                                suggested_angle="Different angle",
                            )
                        ],
                        general_guidance="Try different topics",
                    )

                    result = await preview_validation_node(state)

                    # Should go back to PLANNING
                    assert result["current_phase"] == Phase.PLANNING.value
                    assert result["planning_iteration"] == 1
                    assert "replanning_feedback" in result

                    # Should have scratchpad entry with failure
                    assert "preview_validation_scratchpad" in result
                    assert len(result["preview_validation_scratchpad"]) == 1
                    scratchpad_entry = result["preview_validation_scratchpad"][0]
                    assert scratchpad_entry["iteration"] == 0
                    assert scratchpad_entry["passed"] is False
                    assert len(scratchpad_entry["rejected_section_ids"]) == 1
                    assert "section_1" in scratchpad_entry["rejected_section_ids"]
                    assert scratchpad_entry["feedback_text"] != ""


class TestPreviewValidationRouter:
    """Test preview validation router."""

    def test_router_proceed_to_research(self):
        """Test router when validation passes."""
        state = {"current_phase": Phase.SECTION_SELECTION.value}
        assert preview_validation_router(state) == "section_selection"

    def test_router_replan(self):
        """Test router when validation fails."""
        state = {"current_phase": Phase.PLANNING.value}
        assert preview_validation_router(state) == "planning"

    def test_router_default_fallback(self):
        """Test router with unexpected phase."""
        state = {"current_phase": "UNEXPECTED"}
        assert preview_validation_router(state) == "section_selection"
