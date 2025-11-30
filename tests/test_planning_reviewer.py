"""
Tests for outline reviewer module.

Tests the outline quality review and iteration functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.planning.outline_reviewer import OutlineReviewer, OutlineReview
from src.planning.outline_generator import (
    OutlineGenerator,
    BlogOutline,
    OutlineSection,
    OutlineMetadata,
)
from src.research.content_extractor import AggregatedExtractedContent, ExtractedContent


@pytest.fixture
def sample_outline():
    """Create a sample blog outline for testing."""
    return BlogOutline(
        topic="Test Topic",
        sections=[
            OutlineSection(
                heading="Introduction",
                summary="Introduction to the topic",
                references=["https://example.com/1"],
            ),
            OutlineSection(
                heading="Core Concepts",
                summary="Explaining the core concepts",
                references=["https://example.com/2"],
            ),
        ],
        metadata=OutlineMetadata(
            target_audience="Developers",
            difficulty="Intermediate",
            estimated_reading_time="10 minutes",
        ),
    )


@pytest.fixture
def sample_research_data():
    """Create sample research data."""
    return AggregatedExtractedContent(
        topic="Test Topic",
        statistics={"total_urls": 1, "successful": 1},
        contents=[
            ExtractedContent(
                url="https://example.com",
                title="Test Article",
                snippet="Test snippet",
                headings=["Section 1", "Section 2"],
                success=True,
            )
        ],
    )


@pytest.fixture
def mock_reviewer():
    """Create a mock reviewer with mocked LLM."""
    reviewer = OutlineReviewer()
    reviewer._llm = MagicMock()
    return reviewer


class TestOutlineReview:
    """Test OutlineReview model."""

    def test_outline_review_float_scores(self):
        """Test that OutlineReview accepts float scores."""
        review = OutlineReview(
            score=8.5,
            completeness_score=9.0,
            logical_flow_score=8.0,
            depth_score=8.5,
            balance_score=9.0,
            audience_fit_score=8.0,
            strengths=["Clear structure", "Good flow"],
            weaknesses=["Needs more depth"],
            specific_feedback="Add more code examples",
        )
        
        assert review.score == 8.5
        assert review.completeness_score == 9.0
        assert isinstance(review.score, float)

    def test_outline_review_score_validation(self):
        """Test score validation (0.0-10.0 range)."""
        # Valid scores
        review = OutlineReview(
            score=10.0,
            completeness_score=0.0,
            logical_flow_score=5.5,
            depth_score=7.2,
            balance_score=8.9,
            audience_fit_score=9.3,
            strengths=[],
            weaknesses=[],
            specific_feedback="",
        )
        assert review.score == 10.0
        
        # Invalid score - above range
        with pytest.raises(Exception):  # Pydantic ValidationError
            OutlineReview(
                score=11.0,
                completeness_score=5.0,
                logical_flow_score=5.0,
                depth_score=5.0,
                balance_score=5.0,
                audience_fit_score=5.0,
                strengths=[],
                weaknesses=[],
                specific_feedback="",
            )


class TestOutlineReviewer:
    """Test OutlineReviewer class."""

    @pytest.mark.asyncio
    async def test_review_outline(self, mock_reviewer, sample_outline):
        """Test reviewing an outline."""
        # Mock LLM response
        mock_review = OutlineReview(
            score=8.5,
            completeness_score=9.0,
            logical_flow_score=8.0,
            depth_score=8.5,
            balance_score=9.0,
            audience_fit_score=8.0,
            strengths=["Clear structure", "Good progression"],
            weaknesses=["Needs more code examples"],
            specific_feedback="Add practical code examples in sections 2 and 3",
        )
        
        mock_reviewer._llm.ainvoke = AsyncMock(return_value=mock_review)
        
        # Test review
        review = await mock_reviewer.review(sample_outline)
        
        assert review.score == 8.5
        assert len(review.strengths) == 2
        assert len(review.weaknesses) == 1
        assert "code examples" in review.specific_feedback

    @pytest.mark.asyncio
    async def test_save_reviews(self, mock_reviewer, tmp_path):
        """Test saving reviews to YAML."""
        reviews = [
            OutlineReview(
                score=7.5,
                completeness_score=7.0,
                logical_flow_score=8.0,
                depth_score=7.5,
                balance_score=8.0,
                audience_fit_score=7.0,
                strengths=["Good structure"],
                weaknesses=["Lacks depth"],
                specific_feedback="Add more details",
            ),
            OutlineReview(
                score=9.0,
                completeness_score=9.0,
                logical_flow_score=9.0,
                depth_score=8.5,
                balance_score=9.0,
                audience_fit_score=9.0,
                strengths=["Excellent structure", "Great depth"],
                weaknesses=[],
                specific_feedback="Looking good",
            ),
        ]
        
        file_path = tmp_path / "reviews.yaml"
        await mock_reviewer.save_reviews(reviews, file_path)
        
        assert file_path.exists()
        
        # Verify YAML content
        import yaml
        with open(file_path) as f:
            data = yaml.safe_load(f)
        
        assert data["total_iterations"] == 2
        assert data["final_score"] == 9.0
        assert len(data["iterations"]) == 2
        assert data["iterations"][0]["overall_score"] == 7.5
        assert data["iterations"][1]["overall_score"] == 9.0

    def test_display_review(self, mock_reviewer):
        """Test displaying review feedback."""
        review = OutlineReview(
            score=8.5,
            completeness_score=9.0,
            logical_flow_score=8.0,
            depth_score=8.5,
            balance_score=9.0,
            audience_fit_score=8.0,
            strengths=["Clear structure"],
            weaknesses=["Needs examples"],
            specific_feedback="Add code examples",
        )
        
        # Mock console
        mock_console = MagicMock()
        
        # Should not raise any errors
        mock_reviewer.display_review(review, iteration=1, console=mock_console)
        
        # Verify console.print was called
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_regenerate_with_feedback(
        self, mock_reviewer, sample_outline, sample_research_data
    ):
        """Test regenerating outline with feedback."""
        mock_generator = MagicMock()
        improved_outline = BlogOutline(
            topic="Test Topic",
            sections=[
                OutlineSection(
                    heading="Introduction",
                    summary="Improved introduction with examples",
                    references=["https://example.com/1"],
                ),
            ],
            metadata=OutlineMetadata(
                target_audience="Developers",
                difficulty="Intermediate",
                estimated_reading_time="12 minutes",
            ),
        )
        mock_generator.generate = AsyncMock(return_value=improved_outline)
        
        review = OutlineReview(
            score=7.0,
            completeness_score=7.0,
            logical_flow_score=8.0,
            depth_score=6.0,
            balance_score=7.0,
            audience_fit_score=7.0,
            strengths=["Good flow"],
            weaknesses=["Lacks practical examples"],
            specific_feedback="Add more code examples and practical use cases",
        )
        
        result = await mock_reviewer._regenerate_with_feedback(
            topic="Test Topic",
            research_data=sample_research_data,
            generator=mock_generator,
            review=review,
        )
        
        assert result == improved_outline
        # Verify generator was called with feedback
        mock_generator.generate.assert_called_once()
        call_args = mock_generator.generate.call_args
        assert call_args[1]["feedback"] is not None
        assert call_args[1]["feedback"]["score"] == 7.0
        assert "practical examples" in call_args[1]["feedback"]["weaknesses"][0]


class TestOutlineGeneratorFeedbackIntegration:
    """Test OutlineGenerator with feedback parameter."""

    @pytest.mark.asyncio
    async def test_generate_without_feedback(self, sample_research_data):
        """Test generating outline without feedback."""
        generator = OutlineGenerator()
        generator._llm = MagicMock()
        
        mock_outline = BlogOutline(
            topic="Test",
            sections=[],
            metadata=OutlineMetadata(
                target_audience="Devs",
                difficulty="Intermediate",
                estimated_reading_time="10 min",
            ),
        )
        generator._llm.ainvoke = AsyncMock(return_value=mock_outline)
        
        result = await generator.generate("Test Topic", sample_research_data)
        
        assert result == mock_outline
        # Verify prompt doesn't contain feedback
        call_args = generator._llm.ainvoke.call_args[0][0]
        user_prompt = call_args[1]["content"]
        assert "PREVIOUS ITERATION FEEDBACK" not in user_prompt

    @pytest.mark.asyncio
    async def test_generate_with_feedback(self, sample_research_data):
        """Test generating outline with feedback."""
        generator = OutlineGenerator()
        generator._llm = MagicMock()
        
        mock_outline = BlogOutline(
            topic="Test",
            sections=[],
            metadata=OutlineMetadata(
                target_audience="Devs",
                difficulty="Intermediate",
                estimated_reading_time="10 min",
            ),
        )
        generator._llm.ainvoke = AsyncMock(return_value=mock_outline)
        
        feedback = {
            "score": 7.5,
            "strengths": ["Clear structure"],
            "weaknesses": ["Needs more depth"],
            "specific_feedback": "Add more code examples",
        }
        
        result = await generator.generate("Test Topic", sample_research_data, feedback=feedback)
        
        assert result == mock_outline
        # Verify prompt contains feedback
        call_args = generator._llm.ainvoke.call_args[0][0]
        user_prompt = call_args[1]["content"]
        assert "PREVIOUS ITERATION FEEDBACK" in user_prompt
        assert "7.5/10.0" in user_prompt
        assert "Clear structure" in user_prompt
        assert "Needs more depth" in user_prompt
        assert "Add more code examples" in user_prompt


class TestReviewIteration:
    """Test the full review and iteration workflow."""

    @pytest.mark.asyncio
    async def test_iteration_stops_when_threshold_met(
        self, sample_research_data
    ):
        """Test that iteration stops when score meets threshold."""
        reviewer = OutlineReviewer()
        generator = OutlineGenerator()
        
        # Mock LLM responses
        mock_outline = BlogOutline(
            topic="Test",
            sections=[],
            metadata=OutlineMetadata(
                target_audience="Devs",
                difficulty="Intermediate",
                estimated_reading_time="10 min",
            ),
        )
        
        high_score_review = OutlineReview(
            score=9.5,
            completeness_score=9.5,
            logical_flow_score=9.5,
            depth_score=9.0,
            balance_score=9.5,
            audience_fit_score=9.5,
            strengths=["Excellent"],
            weaknesses=[],
            specific_feedback="Great work!",
        )
        
        reviewer._llm = MagicMock()
        reviewer._llm.ainvoke = AsyncMock(return_value=high_score_review)
        
        generator._llm = MagicMock()
        generator._llm.ainvoke = AsyncMock(return_value=mock_outline)
        
        # Should only do 1 iteration since score >= threshold
        outline, reviews = await reviewer.review_and_iterate(
            topic="Test",
            research_data=sample_research_data,
            generator=generator,
            max_iterations=3,
            quality_threshold=9.0,
        )
        
        assert len(reviews) == 1
        assert reviews[0].score == 9.5

    @pytest.mark.asyncio
    async def test_iteration_continues_when_threshold_not_met(
        self, sample_research_data
    ):
        """Test that iteration continues when score is below threshold."""
        reviewer = OutlineReviewer()
        generator = OutlineGenerator()
        
        mock_outline = BlogOutline(
            topic="Test",
            sections=[],
            metadata=OutlineMetadata(
                target_audience="Devs",
                difficulty="Intermediate",
                estimated_reading_time="10 min",
            ),
        )
        
        low_review = OutlineReview(
            score=7.0,
            completeness_score=7.0,
            logical_flow_score=7.0,
            depth_score=7.0,
            balance_score=7.0,
            audience_fit_score=7.0,
            strengths=["Good"],
            weaknesses=["Needs work"],
            specific_feedback="Add more",
        )
        
        high_review = OutlineReview(
            score=9.5,
            completeness_score=9.5,
            logical_flow_score=9.5,
            depth_score=9.0,
            balance_score=9.5,
            audience_fit_score=9.5,
            strengths=["Excellent"],
            weaknesses=[],
            specific_feedback="Perfect!",
        )
        
        reviewer._llm = MagicMock()
        # First call returns low score, second returns high score
        reviewer._llm.ainvoke = AsyncMock(side_effect=[low_review, high_review])
        
        generator._llm = MagicMock()
        generator._llm.ainvoke = AsyncMock(return_value=mock_outline)
        
        outline, reviews = await reviewer.review_and_iterate(
            topic="Test",
            research_data=sample_research_data,
            generator=generator,
            max_iterations=3,
            quality_threshold=9.0,
        )
        
        # Should do 2 iterations
        assert len(reviews) == 2
        assert reviews[0].score == 7.0
        assert reviews[1].score == 9.5
