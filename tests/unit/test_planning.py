"""Unit tests for planning_node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.nodes import (
    planning_node,
    _generate_blog_plan,
    _format_topic_context_snippets,
    _build_planning_prompt,
    TARGET_WORDS_MAP,
)
from src.agent.state import Phase, BlogAgentState, BlogPlan, PlanSection


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_state() -> BlogAgentState:
    """Sample state for testing with topic_context from discovery."""
    return {
        "job_id": "test-job",
        "title": "Semantic Caching for LLM Applications",
        "context": "Saw GPTCache on Twitter. Redis vector search for caching.",
        "target_length": "medium",
        "topic_context": [
            {"title": "GPTCache Guide", "url": "https://gptcache.io", "snippet": "Cache LLM responses"},
            {"title": "Redis Vector", "url": "https://redis.io/vector", "snippet": "Vector similarity search"},
        ],
    }


@pytest.fixture
def mock_blog_plan():
    """Mock BlogPlan response with required and optional sections."""
    return BlogPlan(
        blog_title="Semantic Caching for LLM Applications",
        target_words=1500,
        sections=[
            PlanSection(
                id="hook",
                title=None,
                role="hook",
                search_queries=["semantic caching benefits"],
                needs_code=False,
                needs_diagram=False,
                target_words=150,
                optional=False,
            ),
            PlanSection(
                id="problem",
                title="The LLM Cost Problem",
                role="problem",
                search_queries=["LLM API costs", "repeated queries"],
                needs_code=False,
                needs_diagram=False,
                target_words=200,
                optional=False,
            ),
            PlanSection(
                id="why",
                title="Why Semantic Caching Works",
                role="why",
                search_queries=["semantic similarity caching", "vector embeddings cache"],
                needs_code=False,
                needs_diagram=True,
                target_words=300,
                optional=False,
            ),
            PlanSection(
                id="gptcache",
                title="Building with GPTCache",
                role="deep_dive",
                search_queries=["GPTCache tutorial", "GPTCache setup"],
                needs_code=True,
                needs_diagram=False,
                target_words=350,
                optional=False,
            ),
            PlanSection(
                id="redis_vector",
                title="Redis Vector Search",
                role="deep_dive",
                search_queries=["Redis vector search setup"],
                needs_code=True,
                needs_diagram=True,
                target_words=350,
                optional=False,
            ),
            # Optional sections - user can choose to include
            PlanSection(
                id="benchmarking",
                title="Benchmarking Cache Performance",
                role="deep_dive",
                search_queries=["semantic cache benchmarks"],
                needs_code=True,
                needs_diagram=False,
                target_words=250,
                optional=True,
            ),
            PlanSection(
                id="production",
                title="Production Deployment Tips",
                role="deep_dive",
                search_queries=["semantic cache production"],
                needs_code=False,
                needs_diagram=True,
                target_words=250,
                optional=True,
            ),
            PlanSection(
                id="conclusion",
                title="Key Takeaways",
                role="conclusion",
                search_queries=[],
                needs_code=False,
                needs_diagram=False,
                target_words=150,
                optional=False,
            ),
        ]
    )


# =============================================================================
# Target Words Map Tests
# =============================================================================


class TestTargetWordsMap:
    """Tests for TARGET_WORDS_MAP constant."""

    def test_short_length(self):
        """Short length maps to 500 words."""
        assert TARGET_WORDS_MAP["short"] == 500

    def test_medium_length(self):
        """Medium length maps to 1000 words."""
        assert TARGET_WORDS_MAP["medium"] == 1000

    def test_long_length(self):
        """Long length maps to 1500 words."""
        assert TARGET_WORDS_MAP["long"] == 1500


# =============================================================================
# Format Topic Context Snippets Tests
# =============================================================================


class TestFormatTopicContextSnippets:
    """Tests for _format_topic_context_snippets helper."""

    def test_formats_snippets_correctly(self):
        """Formats snippets with title, url, and snippet text."""
        topic_context = [
            {"title": "Article 1", "url": "https://example1.com", "snippet": "Snippet 1"},
            {"title": "Article 2", "url": "https://example2.com", "snippet": "Snippet 2"},
        ]

        result = _format_topic_context_snippets(topic_context)

        assert "1. [Article 1](https://example1.com)" in result
        assert "Snippet 1" in result
        assert "2. [Article 2](https://example2.com)" in result
        assert "Snippet 2" in result

    def test_returns_fallback_for_empty_context(self):
        """Returns fallback message when topic_context is empty."""
        result = _format_topic_context_snippets([])

        assert "No topic research available" in result

    def test_respects_max_snippets(self):
        """Limits snippets to max_snippets parameter."""
        topic_context = [
            {"title": f"Article {i}", "url": f"https://example{i}.com", "snippet": f"Snippet {i}"}
            for i in range(20)
        ]

        result = _format_topic_context_snippets(topic_context, max_snippets=5)

        # Should have snippets 1-5, not 6+
        assert "1. [Article 0]" in result
        assert "5. [Article 4]" in result
        assert "6. [Article 5]" not in result

    def test_handles_missing_fields(self):
        """Handles missing title, url, or snippet gracefully."""
        topic_context = [
            {"title": "", "url": "https://example.com", "snippet": "Test"},
            {"url": "https://nosnippet.com"},  # Missing title and snippet
        ]

        result = _format_topic_context_snippets(topic_context)

        # Should not crash, should contain URLs
        assert "https://example.com" in result


# =============================================================================
# Build Planning Prompt Tests
# =============================================================================


class TestBuildPlanningPrompt:
    """Tests for _build_planning_prompt helper."""

    def test_includes_title_and_context(self):
        """Prompt includes title and context."""
        prompt = _build_planning_prompt(
            title="Test Title",
            context="Test Context",
            target_words=1500,
            topic_snippets="Snippet content",
        )

        assert 'Title: "Test Title"' in prompt
        assert "Context: Test Context" in prompt

    def test_includes_target_words(self):
        """Prompt includes target word count."""
        prompt = _build_planning_prompt(
            title="Test",
            context="Context",
            target_words=2500,
            topic_snippets="Snippets",
        )

        assert "2500" in prompt

    def test_includes_topic_snippets(self):
        """Prompt includes topic research snippets."""
        snippets = "1. [Article](url)\n   Snippet text"
        prompt = _build_planning_prompt(
            title="Test",
            context="Context",
            target_words=1500,
            topic_snippets=snippets,
        )

        assert snippets in prompt

    def test_includes_blog_structure_rules(self):
        """Prompt includes blog structure requirements."""
        prompt = _build_planning_prompt(
            title="Test",
            context="Context",
            target_words=1500,
            topic_snippets="Snippets",
        )

        assert "Hook" in prompt
        assert "Problem" in prompt
        assert "Conclusion" in prompt


# =============================================================================
# Planning Node Tests
# =============================================================================


class TestPlanningNode:
    """Tests for planning_node function."""

    @pytest.mark.asyncio
    async def test_returns_plan_and_advances_phase(
        self,
        sample_state,
        mock_blog_plan,
    ):
        """Node returns plan and advances to RESEARCHING phase."""
        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm:
                    mock_gen.return_value = mock_blog_plan
                    mock_km.from_env.return_value = MagicMock()
                    mock_jm.return_value = MagicMock()

                    result = await planning_node(sample_state)

                    assert "plan" in result
                    assert result["current_phase"] == Phase.RESEARCHING.value
                    assert len(result["plan"]["sections"]) == 8  # 6 required + 2 optional

    @pytest.mark.asyncio
    async def test_plan_includes_optional_sections(
        self,
        sample_state,
        mock_blog_plan,
    ):
        """Plan includes both required and optional sections."""
        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm:
                    mock_gen.return_value = mock_blog_plan
                    mock_km.from_env.return_value = MagicMock()
                    mock_jm.return_value = MagicMock()

                    result = await planning_node(sample_state)

                    sections = result["plan"]["sections"]
                    required = [s for s in sections if not s["optional"]]
                    optional = [s for s in sections if s["optional"]]

                    assert len(required) == 6  # hook, problem, why, 2 deep_dive, conclusion
                    assert len(optional) == 2  # 2 extra deep_dive options

    @pytest.mark.asyncio
    async def test_fails_without_title(self):
        """Node fails if title is missing."""
        state: BlogAgentState = {"context": "Some context"}

        result = await planning_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_fails_with_empty_title(self):
        """Node fails if title is empty string."""
        state: BlogAgentState = {"title": "", "context": "Some context"}

        result = await planning_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "Title is required" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_uses_default_medium_length(self, mock_blog_plan):
        """Uses medium (1500 words) when target_length not specified."""
        state: BlogAgentState = {
            "title": "Test Title",
            "context": "Test context",
            # No target_length specified
        }

        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                mock_gen.return_value = mock_blog_plan
                mock_km.from_env.return_value = MagicMock()

                await planning_node(state)

                # Should be called with target_words=1500 (medium)
                call_args = mock_gen.call_args
                assert call_args.kwargs.get("target_words") == 1500

    @pytest.mark.asyncio
    async def test_respects_target_length_short(self, mock_blog_plan):
        """Uses short (800 words) when specified."""
        state: BlogAgentState = {
            "title": "Test Title",
            "context": "Test context",
            "target_length": "short",
        }

        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                mock_gen.return_value = mock_blog_plan
                mock_km.from_env.return_value = MagicMock()

                await planning_node(state)

                call_args = mock_gen.call_args
                assert call_args.kwargs.get("target_words") == 800

    @pytest.mark.asyncio
    async def test_respects_target_length_long(self, mock_blog_plan):
        """Uses long (2500 words) when specified."""
        state: BlogAgentState = {
            "title": "Test Title",
            "context": "Test context",
            "target_length": "long",
        }

        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                mock_gen.return_value = mock_blog_plan
                mock_km.from_env.return_value = MagicMock()

                await planning_node(state)

                call_args = mock_gen.call_args
                assert call_args.kwargs.get("target_words") == 2500

    @pytest.mark.asyncio
    async def test_handles_empty_topic_context(self, mock_blog_plan):
        """Works with empty topic_context."""
        state: BlogAgentState = {
            "title": "Test Title",
            "context": "Test context",
            "topic_context": [],
        }

        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                mock_gen.return_value = mock_blog_plan
                mock_km.from_env.return_value = MagicMock()

                result = await planning_node(state)

                assert result["current_phase"] == Phase.RESEARCHING.value
                assert "plan" in result

    @pytest.mark.asyncio
    async def test_handles_plan_generation_failure(self, sample_state):
        """Node handles LLM failure gracefully."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
                mock_km.from_env.return_value = MagicMock()
                mock_gen.side_effect = RuntimeError("All API keys exhausted")

                result = await planning_node(sample_state)

                assert result["current_phase"] == Phase.FAILED.value
                assert "exhausted" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_handles_unexpected_error(self, sample_state):
        """Node handles unexpected errors gracefully."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            mock_km.from_env.side_effect = ValueError("No API keys found")

            result = await planning_node(sample_state)

            assert result["current_phase"] == Phase.FAILED.value
            assert "error" in result.get("error_message", "").lower()


# =============================================================================
# Generate Blog Plan Tests
# =============================================================================


class TestGenerateBlogPlan:
    """Tests for _generate_blog_plan helper."""

    @pytest.mark.asyncio
    async def test_returns_blog_plan_model(self, mock_blog_plan):
        """Returns BlogPlan Pydantic model."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_blog_plan
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_blog_plan(
                title="Test",
                context="Context",
                target_words=1500,
                topic_context=[],
                key_manager=mock_key_manager,
            )

            assert isinstance(result, BlogPlan)
            assert len(result.sections) == 8  # 6 required + 2 optional

    @pytest.mark.asyncio
    async def test_records_usage_on_success(self):
        """Records API usage after successful call."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        mock_plan = BlogPlan(
            blog_title="Test",
            target_words=100,
            sections=[
                PlanSection(
                    id="test", title="Test", role="hook",
                    search_queries=[], needs_code=False,
                    needs_diagram=False, target_words=100
                )
            ]
        )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_plan
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            await _generate_blog_plan(
                title="Test",
                context="Context",
                target_words=1500,
                topic_context=[],
                key_manager=mock_key_manager,
            )

            mock_key_manager.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotates_key_on_rate_limit(self):
        """Rotates to next key on 429 error."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = "key2"

        call_count = 0
        mock_plan = BlogPlan(
            blog_title="Test",
            target_words=100,
            sections=[
                PlanSection(
                    id="test", title="Test", role="hook",
                    search_queries=[], needs_code=False,
                    needs_diagram=False, target_words=100
                )
            ]
        )

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Resource Exhausted")
            return mock_plan

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = invoke_side_effect
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_blog_plan(
                title="Test",
                context="Context",
                target_words=1500,
                topic_context=[],
                key_manager=mock_key_manager,
            )

            mock_key_manager.mark_rate_limited.assert_called_once_with("key1")
            assert len(result.sections) == 1

    @pytest.mark.asyncio
    async def test_raises_when_all_keys_exhausted(self):
        """Raises RuntimeError when all keys are exhausted."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = None  # No more keys

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = Exception("429 quota exceeded")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            with pytest.raises(RuntimeError, match="exhausted"):
                await _generate_blog_plan(
                    title="Test",
                    context="Context",
                    target_words=1500,
                    topic_context=[],
                    key_manager=mock_key_manager,
                )


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestPlanningCheckpointing:
    """Tests for checkpoint/resume functionality in planning."""

    @pytest.mark.asyncio
    async def test_saves_checkpoint_on_success(
        self,
        sample_state,
        mock_blog_plan,
    ):
        """Checkpoint is saved after successful planning."""
        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm_class:
                    mock_gen.return_value = mock_blog_plan
                    mock_km.from_env.return_value = MagicMock()

                    mock_jm = MagicMock()
                    mock_jm_class.return_value = mock_jm

                    await planning_node(sample_state)

                    mock_jm.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_without_job_id(self, mock_blog_plan):
        """No checkpoint if job_id not in state."""
        state: BlogAgentState = {
            "title": "Test",
            "context": "Context",
            # No job_id
        }

        with patch("src.agent.nodes._generate_blog_plan") as mock_gen:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm_class:
                    mock_gen.return_value = mock_blog_plan
                    mock_km.from_env.return_value = MagicMock()

                    await planning_node(state)

                    # JobManager should not be instantiated
                    mock_jm_class.assert_not_called()
