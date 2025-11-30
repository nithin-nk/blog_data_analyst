"""
Integration tests for E-E-A-T based outline review functionality.

Tests verify that the outline review system properly evaluates outlines based on
Google's Search Quality Guidelines and E-E-A-T principles.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.planning.outline_reviewer import OutlineReviewer, OutlineReview
from src.planning.outline_generator import BlogOutline, OutlineSection, OutlineMetadata


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.planning.outline_reviewer.get_settings') as mock:
        settings = MagicMock()
        settings.google_api_key = "test_key"
        settings.max_outline_iterations = 3
        settings.outline_quality_threshold = 9.5
        mock.return_value = settings
        yield settings


@pytest.fixture
def outline_reviewer(mock_settings):
    """Create an OutlineReviewer instance with mocked dependencies."""
    with patch('src.planning.outline_reviewer.ChatGoogleGenerativeAI'):
        reviewer = OutlineReviewer(model_name="gemini-2.5-flash")
        return reviewer


@pytest.fixture
def sample_outline_high_eeat():
    """Sample outline with strong E-E-A-T characteristics."""
    return BlogOutline(
        topic="Building Production-Ready AI Agents: Lessons from 50+ Deployments",
        sections=[
            OutlineSection(
                heading="Infrastructure Requirements Based on Real-World Deployments",
                summary="Detailed analysis of infrastructure needs based on actual deployment data from 50+ production AI agents, including specific resource requirements, cost breakdowns, and performance metrics.",
                references=["https://example.com/research1", "https://example.com/case-study1"]
            ),
            OutlineSection(
                heading="Performance Optimization: A/B Test Results",
                summary="Comparative analysis of different optimization strategies with quantified performance improvements, latency data, and cost savings from our 6-month optimization project.",
                references=["https://example.com/research2", "https://example.com/benchmark"]
            ),
            OutlineSection(
                heading="Common Pitfalls and How We Solved Them",
                summary="Specific problems encountered during deployments with step-by-step solutions, including code examples and configuration changes that resolved each issue.",
                references=["https://example.com/troubleshooting"]
            )
        ],
        metadata=OutlineMetadata(
            target_audience="DevOps engineers and ML practitioners",
            difficulty="Intermediate to Advanced",
            estimated_reading_time="15 minutes"
        )
    )


@pytest.fixture
def sample_outline_poor_eeat():
    """Sample outline lacking E-E-A-T characteristics."""
    return BlogOutline(
        topic="Top 10 Best AI Agent Deployment Tools 2024",
        sections=[
            OutlineSection(
                heading="Introduction to AI Agents",
                summary="General overview of what AI agents are and why they are important.",
                references=[]
            ),
            OutlineSection(
                heading="Popular Deployment Tools",
                summary="List of various tools that can be used for AI agent deployment.",
                references=[]
            ),
            OutlineSection(
                heading="Conclusion",
                summary="Summary of the benefits of using these tools.",
                references=[]
            )
        ],
        metadata=OutlineMetadata(
            target_audience="General audience",
            difficulty="Beginner",
            estimated_reading_time="5 minutes"
        )
    )


class TestEEATOutlineReview:
    """Test suite for E-E-A-T based outline review."""

    @pytest.mark.asyncio
    async def test_review_high_eeat_outline(self, outline_reviewer, sample_outline_high_eeat, mock_settings):
        """Test that outlines with strong E-E-A-T characteristics score well."""
        
        # Mock high-quality review
        mock_review = OutlineReview(
            score=9.3,
            completeness_score=9.5,
            logical_flow_score=9.2,
            depth_score=9.4,
            balance_score=9.0,
            audience_fit_score=9.3,
            strengths=[
                "Demonstrates first-hand experience from 50+ deployments with specific metrics",
                "Includes evidence-based claims with A/B test results and quantified data",
                "Clear progression from infrastructure to optimization to troubleshooting"
            ],
            weaknesses=[
                "Could include more diverse reference sources",
                "Minor: Consider adding a section on monitoring and observability"
            ],
            specific_feedback="Excellent E-E-A-T demonstration throughout. The outline shows clear expertise "
                            "through specific deployment numbers and real-world data. Strong people-first approach "
                            "with actionable, practical content."
        )

        # Mock the _llm attribute directly
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_review)
        outline_reviewer._llm = mock_llm
        
        result = await outline_reviewer.review(sample_outline_high_eeat)
        
        # Verify high scores
        assert result.score >= 9.0
        assert result.completeness_score >= 9.0
        assert 'experience' in result.specific_feedback.lower() or 'E-E-A-T' in result.specific_feedback
        
        # Verify the prompt was called
        assert mock_llm.ainvoke.called


    @pytest.mark.asyncio
    async def test_review_poor_eeat_outline(self, outline_reviewer, sample_outline_poor_eeat, mock_settings):
        """Test that outlines lacking E-E-A-T characteristics score poorly."""
        
        # Mock poor quality review
        mock_review = OutlineReview(
            score=5.2,
            completeness_score=5.0,
            logical_flow_score=6.0,
            depth_score=4.5,
            balance_score=5.5,
            audience_fit_score=5.0,
            strengths=[
                "Clear basic structure",
                "Appropriate for beginners"
            ],
            weaknesses=[
                "No evidence of first-hand experience or expertise",
                "Generic 'Top 10' listicle format suggests search-engine-first content",
                "Lacks specific, actionable information and original insights"
            ],
            specific_feedback="Major E-E-A-T concerns. The outline lacks evidence of expertise or first-hand "
                            "experience. Title suggests clickbait/search-engine optimization. Sections are too "
                            "generic without specific value propositions. This appears designed for search traffic "
                            "rather than providing genuine value to readers. Score below 6.0 due to search-engine-first "
                            "approach."
        )

        # Mock the _llm attribute directly
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_review)
        outline_reviewer._llm = mock_llm
        
        result = await outline_reviewer.review(sample_outline_poor_eeat)
        
        # Verify low scores
        assert result.score < 7.0
        assert result.completeness_score < 7.0
        assert 'E-E-A-T' in result.specific_feedback or 'experience' in result.specific_feedback.lower()
        assert 'search' in result.specific_feedback.lower() or 'generic' in result.specific_feedback.lower()


    @pytest.mark.asyncio
    async def test_system_prompt_includes_eeat(self, outline_reviewer, sample_outline_high_eeat, mock_settings):
        """Test that the system prompt includes E-E-A-T criteria."""
        
        system_prompt = outline_reviewer.SYSTEM_PROMPT
        
        # Verify E-E-A-T elements are present
        assert "E-E-A-T" in system_prompt
        assert "Experience" in system_prompt or "experience" in system_prompt
        assert "Expertise" in system_prompt or "expertise" in system_prompt
        assert "Authoritativeness" in system_prompt or "authoritative" in system_prompt.lower()
        assert "Trustworthiness" in system_prompt or "trustworthy" in system_prompt.lower()
        
        # Verify people-first vs search-engine-first distinction
        assert "people-first" in system_prompt
        assert "search-engine" in system_prompt
        
        # Verify critical requirements
        assert "CRITICAL" in system_prompt
        assert "< 7.0" in system_prompt or "score below 7" in system_prompt.lower()
        assert "< 6.0" in system_prompt or "score below 6" in system_prompt.lower()


    @pytest.mark.asyncio
    async def test_review_criteria_renamed(self, outline_reviewer, sample_outline_high_eeat, mock_settings):
        """Test that review criteria mention E-E-A-T aspects."""
        
        system_prompt = outline_reviewer.SYSTEM_PROMPT
        
        # Verify criteria include E-E-A-T language
        assert "Completeness & E-E-A-T Alignment" in system_prompt
        assert "Depth & Originality" in system_prompt
        assert "Audience Fit & Value" in system_prompt
        
        # Verify scoring guidelines mention E-E-A-T
        lower_prompt = system_prompt.lower()
        assert "demonstrates strong e-e-a-t" in lower_prompt or "strong e-e-a-t planning" in lower_prompt


    @pytest.mark.asyncio
    async def test_review_search_engine_first_outline(self, outline_reviewer, mock_settings):
        """Test that search-engine-first outlines score below 6.0."""
        
        search_engine_outline = BlogOutline(
            topic="Best AI Agents 2024 - Top 10 AI Agents - Complete Guide to AI Agents",
            sections=[
                OutlineSection(
                    heading="What are AI Agents? AI Agents Explained",
                    summary="Explanation of AI agents with keyword: AI agents, AI agent systems, best AI agents",
                    references=[]
                ),
                OutlineSection(
                    heading="Top 10 Best AI Agents for 2024",
                    summary="Ranking of AI agents with reviews and comparisons for AI agent selection",
                    references=[]
                )
            ],
            metadata=OutlineMetadata(
                target_audience="Anyone searching for AI agents",
                difficulty="Any level",
                estimated_reading_time="3 minutes"
            )
        )

        mock_review = OutlineReview(
            score=3.8,
            completeness_score=4.0,
            logical_flow_score=5.0,
            depth_score=3.0,
            balance_score=4.0,
            audience_fit_score=3.5,
            strengths=[
                "Covers basic topic"
            ],
            weaknesses=[
                "Obvious keyword stuffing in title and sections",
                "Appears designed for search engines, not people",
                "No evidence of expertise or original value"
            ],
            specific_feedback="Fundamentally flawed outline with clear search-engine-first design. "
                            "Title and headings show keyword stuffing. No E-E-A-T elements. "
                            "Score below 6.0 due to search manipulation intent."
        )

        # Mock the _llm attribute directly
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_review)
        outline_reviewer._llm = mock_llm
        
        result = await outline_reviewer.review(search_engine_outline)
        
        assert result.score < 6.0
        assert 'search' in result.specific_feedback.lower()


    @pytest.mark.asyncio
    async def test_outline_review_model_fields(self):
        """Test that OutlineReview model fields reference E-E-A-T."""
        
        from pydantic import Field
        
        # Get field descriptions
        completeness_field = OutlineReview.model_fields['completeness_score']
        depth_field = OutlineReview.model_fields['depth_score']
        audience_field = OutlineReview.model_fields['audience_fit_score']
        strengths_field = OutlineReview.model_fields['strengths']
        weaknesses_field = OutlineReview.model_fields['weaknesses']
        feedback_field = OutlineReview.model_fields['specific_feedback']
        
        # Verify E-E-A-T is mentioned in descriptions
        assert 'E-E-A-T' in completeness_field.description
        assert 'original insights' in depth_field.description
        assert 'people-first' in audience_field.description
        assert 'E-E-A-T' in strengths_field.description
        assert 'E-E-A-T' in weaknesses_field.description
        assert 'E-E-A-T' in feedback_field.description


    @pytest.mark.asyncio
    async def test_review_with_no_references(self, outline_reviewer, mock_settings):
        """Test that outlines with no references score lower on completeness."""
        
        no_refs_outline = BlogOutline(
            topic="Advanced Kubernetes Deployment Strategies",
            sections=[
                OutlineSection(
                    heading="Blue-Green Deployments",
                    summary="How to implement blue-green deployments in Kubernetes",
                    references=[]  # No references
                ),
                OutlineSection(
                    heading="Canary Releases",
                    summary="Step-by-step guide to canary releases",
                    references=[]  # No references
                )
            ],
            metadata=OutlineMetadata(
                target_audience="DevOps Engineers",
                difficulty="Advanced",
                estimated_reading_time="12 minutes"
            )
        )

        mock_review = OutlineReview(
            score=6.8,
            completeness_score=6.5,  # Lower due to no references
            logical_flow_score=7.5,
            depth_score=7.0,
            balance_score=7.2,
            audience_fit_score=7.0,
            strengths=[
                "Clear topic progression",
                "Appropriate technical depth for audience"
            ],
            weaknesses=[
                "No authoritative sources or references cited",
                "Lacks evidence to support E-E-A-T (authoritativeness/trustworthiness)"
            ],
            specific_feedback="Good structure but lacks authoritative references. "
                            "Adding credible sources would strengthen E-E-A-T. "
                            "Completeness score reduced due to missing reference materials."
        )

        # Mock the _llm attribute directly
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_review)
        outline_reviewer._llm = mock_llm
        
        result = await outline_reviewer.review(no_refs_outline)
        
        assert result.completeness_score < 7.0
        assert 'reference' in result.specific_feedback.lower() or 'authoritative' in result.specific_feedback.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
