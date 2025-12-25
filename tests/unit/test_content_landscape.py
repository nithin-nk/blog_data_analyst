"""Unit tests for content_landscape_analysis_node and related functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.nodes import (
    content_landscape_analysis_node,
    _select_top_urls,
    _format_article_summaries,
    _analyze_single_article,
    _synthesize_content_strategy,
    _format_content_gaps,
    _format_differentiation_requirements,
)
from src.agent.state import (
    Phase,
    BlogAgentState,
    ContentGap,
    ExistingArticleSummary,
    ContentStrategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_state() -> BlogAgentState:
    """Sample state for testing with topic context from discovery."""
    return {
        "job_id": "test-job",
        "title": "Semantic Caching for LLM Applications",
        "context": "Saw GPTCache on Twitter. Redis vector search for caching.",
        "topic_context": [
            {
                "title": "GPTCache Overview",
                "url": "https://example1.com/gptcache",
                "snippet": "GPTCache is a semantic caching system for LLM APIs.",
            },
            {
                "title": "Redis Vector Search",
                "url": "https://example2.com/redis",
                "snippet": "Redis supports vector similarity search.",
            },
            {
                "title": "LLM Optimization",
                "url": "https://example3.com/optimize",
                "snippet": "Techniques for reducing LLM API costs.",
            },
        ],
    }


@pytest.fixture
def sample_topic_context():
    """Sample topic context for helper function tests."""
    return [
        {"title": "Article 1", "url": "https://a.com", "snippet": "Snippet 1"},
        {"title": "Article 2", "url": "https://b.com", "snippet": "Snippet 2"},
        {"title": "Article 3", "url": "https://c.com", "snippet": "Snippet 3"},
    ]


@pytest.fixture
def mock_article_summary():
    """Mock ExistingArticleSummary for testing."""
    return ExistingArticleSummary(
        url="https://example.com",
        title="Test Article",
        main_angle="Beginner-focused tutorial",
        strengths=["Clear explanation", "Good code examples"],
        weaknesses=["No production considerations", "Missing edge cases"],
        key_points_covered=["Basic setup", "Simple usage"],
    )


@pytest.fixture
def mock_content_strategy(mock_article_summary):
    """Mock ContentStrategy for testing."""
    return ContentStrategy(
        unique_angle="Production-focused guide with benchmarks",
        target_persona="senior_engineer",
        reader_problem="Reduce LLM API costs by 50%",
        gaps_to_fill=[
            ContentGap(
                gap_type="missing_edge_cases",
                description="No articles cover cache invalidation",
                opportunity="Add comprehensive cache invalidation section",
            ),
        ],
        existing_content_summary="Most articles focus on basic setup and ignore production concerns.",
        analyzed_articles=[mock_article_summary, mock_article_summary, mock_article_summary],
        differentiation_requirements=[
            "Include performance benchmarks",
            "Cover cache invalidation strategies",
        ],
    )


# =============================================================================
# Pydantic Model Validation Tests
# =============================================================================


class TestContentGapModel:
    """Tests for ContentGap Pydantic model."""

    def test_valid_content_gap(self):
        """Creates valid ContentGap instance."""
        gap = ContentGap(
            gap_type="missing_topic",
            description="No coverage of cache invalidation",
            opportunity="Add dedicated section on invalidation strategies",
        )
        assert gap.gap_type == "missing_topic"
        assert "invalidation" in gap.description.lower()

    def test_gap_types(self):
        """Accepts various gap types."""
        for gap_type in [
            "missing_topic",
            "insufficient_depth",
            "no_examples",
            "missing_edge_cases",
        ]:
            gap = ContentGap(
                gap_type=gap_type,
                description="Test description",
                opportunity="Test opportunity",
            )
            assert gap.gap_type == gap_type


class TestExistingArticleSummaryModel:
    """Tests for ExistingArticleSummary Pydantic model."""

    def test_valid_article_summary(self):
        """Creates valid ExistingArticleSummary instance."""
        summary = ExistingArticleSummary(
            url="https://example.com/article",
            title="Semantic Caching Guide",
            main_angle="Beginner tutorial",
            strengths=["Clear", "Concise"],
            weaknesses=["Too basic"],
            key_points_covered=["Setup", "Basic usage"],
        )
        assert summary.url == "https://example.com/article"
        assert len(summary.strengths) == 2

    def test_article_summary_with_empty_lists(self):
        """Accepts empty strengths/weaknesses lists."""
        summary = ExistingArticleSummary(
            url="https://example.com",
            title="Title",
            main_angle="Angle",
            strengths=[],
            weaknesses=[],
            key_points_covered=[],
        )
        assert summary.strengths == []


class TestContentStrategyModel:
    """Tests for ContentStrategy Pydantic model."""

    def test_valid_content_strategy(self, mock_article_summary):
        """Creates valid ContentStrategy instance."""
        strategy = ContentStrategy(
            unique_angle="Production-focused with benchmarks",
            target_persona="senior_engineer",
            reader_problem="Reduce API costs",
            gaps_to_fill=[
                ContentGap(
                    gap_type="no_examples",
                    description="Lack of code",
                    opportunity="Add examples",
                )
            ],
            existing_content_summary="Articles are basic",
            analyzed_articles=[mock_article_summary] * 3,  # Min 3 required
            differentiation_requirements=["Include benchmarks"],
        )
        assert strategy.unique_angle == "Production-focused with benchmarks"
        assert len(strategy.gaps_to_fill) == 1

    def test_content_strategy_min_articles(self, mock_article_summary):
        """Accepts minimum 3 analyzed articles."""
        articles = [mock_article_summary] * 3
        strategy = ContentStrategy(
            unique_angle="Test",
            target_persona="junior_engineer",
            reader_problem="Learn caching",
            gaps_to_fill=[
                ContentGap(gap_type="t", description="d", opportunity="o")
            ],
            existing_content_summary="Summary",
            analyzed_articles=articles,
            differentiation_requirements=[],
        )
        assert len(strategy.analyzed_articles) == 3

    def test_content_strategy_max_gaps(self, mock_article_summary):
        """Accepts maximum 5 content gaps."""
        gaps = [
            ContentGap(gap_type=f"type{i}", description=f"d{i}", opportunity=f"o{i}")
            for i in range(5)
        ]
        strategy = ContentStrategy(
            unique_angle="Test",
            target_persona="data_scientist",
            reader_problem="Problem",
            gaps_to_fill=gaps,
            existing_content_summary="Summary",
            analyzed_articles=[mock_article_summary] * 3,
            differentiation_requirements=[],
        )
        assert len(strategy.gaps_to_fill) == 5


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestSelectTopUrls:
    """Tests for _select_top_urls helper."""

    def test_returns_max_count_urls(self, sample_topic_context):
        """Returns at most max_count URLs."""
        result = _select_top_urls(sample_topic_context, max_count=2)
        assert len(result) == 2

    def test_returns_all_if_less_than_max(self, sample_topic_context):
        """Returns all URLs if count is less than max."""
        result = _select_top_urls(sample_topic_context, max_count=10)
        assert len(result) == 3

    def test_handles_empty_list(self):
        """Handles empty topic context."""
        result = _select_top_urls([], max_count=10)
        assert result == []

    def test_default_max_count(self, sample_topic_context):
        """Uses default max_count of 10."""
        result = _select_top_urls(sample_topic_context)
        assert len(result) == 3  # Only 3 available


class TestFormatArticleSummaries:
    """Tests for _format_article_summaries helper."""

    def test_formats_single_article(self, mock_article_summary):
        """Formats a single article summary."""
        result = _format_article_summaries([mock_article_summary])
        assert "Article 1:" in result
        assert "Test Article" in result
        assert "Beginner-focused tutorial" in result

    def test_formats_multiple_articles(self, mock_article_summary):
        """Formats multiple article summaries with numbering."""
        articles = [mock_article_summary, mock_article_summary]
        result = _format_article_summaries(articles)
        assert "Article 1:" in result
        assert "Article 2:" in result

    def test_handles_empty_list(self):
        """Handles empty article list."""
        result = _format_article_summaries([])
        assert result == ""

    def test_includes_strengths_and_weaknesses(self, mock_article_summary):
        """Includes strengths and weaknesses in output."""
        result = _format_article_summaries([mock_article_summary])
        assert "Clear explanation" in result
        assert "No production considerations" in result


class TestFormatContentGaps:
    """Tests for _format_content_gaps helper."""

    def test_formats_gaps(self):
        """Formats content gaps correctly."""
        gaps = [
            {
                "gap_type": "missing_topic",
                "description": "No invalidation coverage",
                "opportunity": "Add section",
            }
        ]
        result = _format_content_gaps(gaps)
        assert "[missing_topic]" in result
        assert "No invalidation coverage" in result
        assert "Add section" in result

    def test_handles_empty_gaps(self):
        """Returns message for empty gaps."""
        result = _format_content_gaps([])
        assert "No specific gaps" in result

    def test_formats_multiple_gaps(self):
        """Formats multiple gaps with numbering."""
        gaps = [
            {"gap_type": "type1", "description": "d1", "opportunity": "o1"},
            {"gap_type": "type2", "description": "d2", "opportunity": "o2"},
        ]
        result = _format_content_gaps(gaps)
        assert "1." in result
        assert "2." in result


class TestFormatDifferentiationRequirements:
    """Tests for _format_differentiation_requirements helper."""

    def test_formats_requirements(self):
        """Formats requirements as bullet list."""
        requirements = ["Include benchmarks", "Cover edge cases"]
        result = _format_differentiation_requirements(requirements)
        assert "- Include benchmarks" in result
        assert "- Cover edge cases" in result

    def test_handles_empty_requirements(self):
        """Returns message for empty requirements."""
        result = _format_differentiation_requirements([])
        assert "No specific requirements" in result


# =============================================================================
# Analyze Single Article Tests
# =============================================================================


class TestAnalyzeSingleArticle:
    """Tests for _analyze_single_article helper."""

    @pytest.mark.asyncio
    async def test_returns_article_summary(self, mock_article_summary):
        """Returns ExistingArticleSummary on success."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        article = {
            "url": "https://example.com",
            "title": "Test Article",
            "content": "Article content here...",
        }

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_article_summary
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _analyze_single_article(
                article=article,
                blog_title="Test Blog",
                key_manager=mock_key_manager,
            )

            assert isinstance(result, ExistingArticleSummary)
            assert result.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_records_usage_on_success(self, mock_article_summary):
        """Records API usage after successful analysis."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        article = {"url": "https://example.com", "title": "Test", "content": "Content"}

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_article_summary
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            await _analyze_single_article(
                article=article,
                blog_title="Test",
                key_manager=mock_key_manager,
            )

            mock_key_manager.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_all_failures(self):
        """Returns None when all retries fail."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"
        mock_key_manager.get_next_key.return_value = None

        article = {"url": "https://example.com", "title": "Test", "content": "Content"}

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = Exception("LLM error")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _analyze_single_article(
                article=article,
                blog_title="Test",
                key_manager=mock_key_manager,
                max_retries=2,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_rotates_key_on_rate_limit(self, mock_article_summary):
        """Rotates key on 429 error."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = "key2"

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Resource Exhausted")
            return mock_article_summary

        article = {"url": "https://example.com", "title": "Test", "content": "Content"}

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = invoke_side_effect
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _analyze_single_article(
                article=article,
                blog_title="Test",
                key_manager=mock_key_manager,
            )

            mock_key_manager.mark_rate_limited.assert_called_once_with("key1")
            assert result is not None


# =============================================================================
# Synthesize Content Strategy Tests
# =============================================================================


class TestSynthesizeContentStrategy:
    """Tests for _synthesize_content_strategy helper."""

    @pytest.mark.asyncio
    async def test_returns_content_strategy(
        self, mock_article_summary, mock_content_strategy
    ):
        """Returns ContentStrategy on success."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_content_strategy
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _synthesize_content_strategy(
                blog_title="Test Blog",
                context="Test context",
                analyzed_articles=[mock_article_summary],
                key_manager=mock_key_manager,
            )

            assert isinstance(result, ContentStrategy)
            assert result.unique_angle == "Production-focused guide with benchmarks"

    @pytest.mark.asyncio
    async def test_raises_on_all_failures(self, mock_article_summary):
        """Raises RuntimeError when all retries fail."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"
        mock_key_manager.get_next_key.return_value = None

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = Exception("LLM error")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            with pytest.raises(RuntimeError):
                await _synthesize_content_strategy(
                    blog_title="Test",
                    context="Context",
                    analyzed_articles=[mock_article_summary],
                    key_manager=mock_key_manager,
                    max_retries=2,
                )


# =============================================================================
# Content Landscape Analysis Node Tests
# =============================================================================


class TestContentLandscapeAnalysisNode:
    """Tests for content_landscape_analysis_node function."""

    @pytest.mark.asyncio
    async def test_returns_content_strategy(
        self, sample_state, mock_article_summary, mock_content_strategy
    ):
        """Node returns content_strategy on success."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch(
                        "src.agent.nodes._synthesize_content_strategy"
                    ) as mock_synth:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_km.from_env.return_value = MagicMock()
                            mock_fetch.return_value = {
                                "success": True,
                                "content": "Article content",
                                "title": "Test",
                            }
                            mock_analyze.return_value = mock_article_summary
                            mock_synth.return_value = mock_content_strategy

                            mock_jm_instance = MagicMock()
                            mock_jm_instance.get_job_dir.return_value = MagicMock()
                            mock_jm.return_value = mock_jm_instance

                            result = await content_landscape_analysis_node(
                                sample_state
                            )

                            assert "content_strategy" in result
                            assert result["content_strategy"] is not None

    @pytest.mark.asyncio
    async def test_advances_to_planning_phase(
        self, sample_state, mock_article_summary, mock_content_strategy
    ):
        """Phase advances to PLANNING on success."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch(
                        "src.agent.nodes._synthesize_content_strategy"
                    ) as mock_synth:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_km.from_env.return_value = MagicMock()
                            mock_fetch.return_value = {
                                "success": True,
                                "content": "Content",
                            }
                            mock_analyze.return_value = mock_article_summary
                            mock_synth.return_value = mock_content_strategy

                            mock_jm_instance = MagicMock()
                            mock_jm_instance.get_job_dir.return_value = MagicMock()
                            mock_jm.return_value = mock_jm_instance

                            result = await content_landscape_analysis_node(
                                sample_state
                            )

                            assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_fails_without_title(self):
        """Node fails if title is missing."""
        state: BlogAgentState = {"context": "Some context", "topic_context": []}

        result = await content_landscape_analysis_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_skips_analysis_without_topic_context(self):
        """Node skips analysis if no topic context available."""
        state: BlogAgentState = {
            "title": "Test Title",
            "context": "Context",
            "topic_context": [],
        }

        result = await content_landscape_analysis_node(state)

        # Should return default strategy, not None
        assert result["content_strategy"] is not None
        assert isinstance(result["content_strategy"], dict)
        assert result["content_strategy"]["unique_angle"] != ""
        assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_handles_fetch_failures(
        self, sample_state, mock_article_summary, mock_content_strategy
    ):
        """Node continues if some URL fetches fail."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch(
                        "src.agent.nodes._synthesize_content_strategy"
                    ) as mock_synth:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_km.from_env.return_value = MagicMock()
                            # First fetch fails, rest succeed
                            mock_fetch.side_effect = [
                                {"success": False, "error": "Timeout"},
                                {"success": True, "content": "Content 2"},
                                {"success": True, "content": "Content 3"},
                            ]
                            mock_analyze.return_value = mock_article_summary
                            mock_synth.return_value = mock_content_strategy

                            mock_jm_instance = MagicMock()
                            mock_jm_instance.get_job_dir.return_value = MagicMock()
                            mock_jm.return_value = mock_jm_instance

                            result = await content_landscape_analysis_node(
                                sample_state
                            )

                            # Should still succeed with 2 articles
                            assert result["current_phase"] == Phase.PLANNING.value

    @pytest.mark.asyncio
    async def test_creates_minimal_strategy_with_few_articles(self, sample_state):
        """Creates minimal strategy if fewer than 3 articles analyzed."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch("src.agent.nodes.JobManager") as mock_jm:
                        mock_km.from_env.return_value = MagicMock()
                        mock_fetch.return_value = {
                            "success": True,
                            "content": "Content",
                        }
                        # Only 1 article analyzed successfully
                        mock_analyze.side_effect = [
                            ExistingArticleSummary(
                                url="https://example.com",
                                title="Test",
                                main_angle="Test",
                                strengths=[],
                                weaknesses=[],
                                key_points_covered=[],
                            ),
                            None,  # Second fails
                            None,  # Third fails
                        ]

                        mock_jm_instance = MagicMock()
                        mock_jm_instance.get_job_dir.return_value = MagicMock()
                        mock_jm.return_value = mock_jm_instance

                        result = await content_landscape_analysis_node(sample_state)

                        # Should create minimal strategy
                        assert result["content_strategy"] is not None
                        strategy = result["content_strategy"]
                        assert "practical implementation" in strategy["unique_angle"].lower()

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self, sample_state):
        """Node handles RuntimeError gracefully by returning default strategy."""
        with patch("src.agent.nodes.JobManager") as mock_jm:
            with patch("builtins.open", MagicMock()):
                mock_instance = MagicMock()
                mock_instance.get_job_dir.return_value = MagicMock()
                mock_jm.return_value = mock_instance

                with patch("src.agent.nodes.KeyManager") as mock_km:
                    mock_km.from_env.side_effect = RuntimeError("No API keys")

                    result = await content_landscape_analysis_node(sample_state)

                    # Should return default strategy and continue, not fail
                    assert result["current_phase"] == Phase.PLANNING.value
                    assert result["content_strategy"] is not None
                    assert "Content analysis failed" in result["content_strategy"]["existing_content_summary"]


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestContentLandscapeCheckpointing:
    """Tests for checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_saves_checkpoint_on_success(
        self, sample_state, mock_article_summary, mock_content_strategy
    ):
        """Checkpoint is saved after successful analysis."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch(
                        "src.agent.nodes._synthesize_content_strategy"
                    ) as mock_synth:
                        with patch("src.agent.nodes.JobManager") as mock_jm_class:
                            mock_km.from_env.return_value = MagicMock()
                            mock_fetch.return_value = {
                                "success": True,
                                "content": "Content",
                            }
                            mock_analyze.return_value = mock_article_summary
                            mock_synth.return_value = mock_content_strategy

                            mock_jm = MagicMock()
                            mock_job_dir = MagicMock()
                            mock_job_dir.__truediv__ = MagicMock(
                                return_value=MagicMock()
                            )
                            mock_jm.get_job_dir.return_value = mock_job_dir
                            mock_jm_class.return_value = mock_jm

                            with patch("builtins.open", MagicMock()):
                                with patch("json.dump"):
                                    await content_landscape_analysis_node(
                                        sample_state
                                    )

                            mock_jm.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_without_job_id(
        self, mock_article_summary, mock_content_strategy
    ):
        """No checkpoint if job_id not in state."""
        state: BlogAgentState = {
            "title": "Test",
            "context": "Context",
            "topic_context": [
                {"title": "T", "url": "https://a.com", "snippet": "S"}
            ],
            # No job_id
        }

        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes.fetch_url_content") as mock_fetch:
                with patch(
                    "src.agent.nodes._analyze_single_article"
                ) as mock_analyze:
                    with patch(
                        "src.agent.nodes._synthesize_content_strategy"
                    ) as mock_synth:
                        with patch("src.agent.nodes.JobManager") as mock_jm_class:
                            mock_km.from_env.return_value = MagicMock()
                            mock_fetch.return_value = {
                                "success": True,
                                "content": "Content",
                            }
                            mock_analyze.return_value = mock_article_summary
                            mock_synth.return_value = mock_content_strategy

                            await content_landscape_analysis_node(state)

                            # JobManager should not be instantiated
                            mock_jm_class.assert_not_called()
