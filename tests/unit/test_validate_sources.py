"""Unit tests for validate_sources_node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.nodes import (
    validate_sources_node,
    _validate_section_sources,
    _build_validation_prompt,
    _generate_alternative_queries,
)
from src.agent.state import (
    AlternativeQueries,
    Phase,
    BlogAgentState,
    SourceValidation,
    SourceValidationList,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_plan():
    """Sample plan with sections."""
    return {
        "blog_title": "Semantic Caching for LLM Applications",
        "target_words": 1500,
        "sections": [
            {
                "id": "problem",
                "title": "The LLM Cost Problem",
                "role": "problem",
                "search_queries": ["LLM API costs"],
                "optional": False,
            },
            {
                "id": "why",
                "title": "Why Semantic Caching Works",
                "role": "why",
                "search_queries": ["semantic similarity"],
                "optional": False,
            },
        ],
    }


@pytest.fixture
def sample_research_cache():
    """Sample research cache with fetched content."""
    return {
        "hash1": {
            "url": "https://example1.com",
            "title": "Article 1",
            "content": "Content about LLM costs and caching strategies...",
            "tokens_estimate": 100,
        },
        "hash2": {
            "url": "https://example2.com",
            "title": "Article 2",
            "content": "Content about semantic similarity search...",
            "tokens_estimate": 150,
        },
        "hash3": {
            "url": "https://example3.com",
            "title": "Article 3",
            "content": "More content about vector embeddings...",
            "tokens_estimate": 120,
        },
    }


@pytest.fixture
def sample_state(sample_plan, sample_research_cache) -> BlogAgentState:
    """Sample state for testing."""
    return {
        "job_id": "test-job",
        "title": "Semantic Caching for LLM Applications",
        "plan": sample_plan,
        "research_cache": sample_research_cache,
    }


@pytest.fixture
def mock_validation_result():
    """Mock LLM validation result."""
    return SourceValidationList(
        sources=[
            SourceValidation(
                url="https://example1.com",
                relevant=True,
                quality="high",
                use=True,
                reason="Highly relevant to LLM costs",
            ),
            SourceValidation(
                url="https://example2.com",
                relevant=True,
                quality="medium",
                use=True,
                reason="Good coverage of semantic search",
            ),
            SourceValidation(
                url="https://example3.com",
                relevant=False,
                quality="low",
                use=False,
                reason="Not relevant to this section",
            ),
        ]
    )


# =============================================================================
# Build Validation Prompt Tests
# =============================================================================


class TestBuildValidationPrompt:
    """Tests for _build_validation_prompt helper."""

    def test_includes_blog_title(self):
        """Prompt includes blog title."""
        prompt = _build_validation_prompt(
            blog_title="Test Blog",
            section_title="Test Section",
            section_role="problem",
            sources=[{"url": "https://test.com", "title": "T", "content": "C"}],
        )

        assert "Test Blog" in prompt

    def test_includes_section_info(self):
        """Prompt includes section title and role."""
        prompt = _build_validation_prompt(
            blog_title="Blog",
            section_title="My Section",
            section_role="deep_dive",
            sources=[{"url": "https://test.com", "title": "T", "content": "C"}],
        )

        assert "My Section" in prompt
        assert "deep_dive" in prompt

    def test_includes_all_sources(self):
        """Prompt includes all provided sources."""
        sources = [
            {"url": "https://site1.com", "title": "Title 1", "content": "Content 1"},
            {"url": "https://site2.com", "title": "Title 2", "content": "Content 2"},
        ]

        prompt = _build_validation_prompt(
            blog_title="Blog",
            section_title="Section",
            section_role="problem",
            sources=sources,
        )

        assert "https://site1.com" in prompt
        assert "https://site2.com" in prompt
        assert "Title 1" in prompt
        assert "Title 2" in prompt

    def test_truncates_long_content(self):
        """Long content is truncated in prompt."""
        long_content = "A" * 2000  # 2000 chars
        sources = [{"url": "https://test.com", "title": "T", "content": long_content}]

        prompt = _build_validation_prompt(
            blog_title="Blog",
            section_title="Section",
            section_role="problem",
            sources=sources,
        )

        # Content should be truncated to ~1000 chars
        assert len(prompt) < len(long_content) + 500

    def test_includes_blog_structure_context(self):
        """Prompt includes all section titles when all_sections provided."""
        all_sections = [
            {"title": "The Problem", "role": "problem", "optional": False},
            {"title": "Building with GPTCache", "role": "deep_dive", "optional": False},
            {"title": "Advanced Tuning", "role": "deep_dive", "optional": True},
            {"title": "Conclusion", "role": "conclusion", "optional": False},
        ]

        prompt = _build_validation_prompt(
            blog_title="Semantic Caching",
            section_title="Building with GPTCache",
            section_role="deep_dive",
            sources=[{"url": "https://test.com", "title": "T", "content": "C"}],
            all_sections=all_sections,
        )

        # Should include blog structure
        assert "Blog Structure" in prompt
        assert "The Problem" in prompt
        assert "Building with GPTCache" in prompt
        assert "Advanced Tuning" in prompt
        assert "[optional]" in prompt  # Optional marker
        assert "Conclusion" in prompt


# =============================================================================
# Validate Section Sources Tests
# =============================================================================


class TestValidateSectionSources:
    """Tests for _validate_section_sources helper."""

    @pytest.mark.asyncio
    async def test_returns_validated_sources(self, mock_validation_result):
        """Returns list of validated source dicts."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        sources = [
            {"url": "https://example1.com", "title": "T1", "content": "C1", "tokens_estimate": 100},
            {"url": "https://example2.com", "title": "T2", "content": "C2", "tokens_estimate": 100},
            {"url": "https://example3.com", "title": "T3", "content": "C3", "tokens_estimate": 100},
        ]

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_validation_result
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _validate_section_sources(
                blog_title="Test Blog",
                section={"id": "test", "title": "Test Section", "role": "problem"},
                sources=sources,
                key_manager=mock_key_manager,
            )

            # Should return only sources where use=True (2 of 3)
            assert len(result) == 2
            assert all("quality" in s for s in result)
            assert all("reason" in s for s in result)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_sources(self):
        """Returns empty list when no sources provided."""
        mock_key_manager = MagicMock()

        result = await _validate_section_sources(
            blog_title="Test",
            section={"id": "test", "title": "Test", "role": "problem"},
            sources=[],
            key_manager=mock_key_manager,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_records_api_usage(self, mock_validation_result):
        """Records API usage after successful call."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        sources = [{"url": "https://test.com", "title": "T", "content": "C", "tokens_estimate": 50}]

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = SourceValidationList(
                sources=[SourceValidation(
                    url="https://test.com", relevant=True, quality="high", use=True, reason="Good"
                )]
            )
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            await _validate_section_sources(
                blog_title="Test",
                section={"id": "test", "title": "Test", "role": "problem"},
                sources=sources,
                key_manager=mock_key_manager,
            )

            mock_key_manager.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotates_key_on_rate_limit(self):
        """Rotates to next key on 429 error."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = "key2"

        sources = [{"url": "https://test.com", "title": "T", "content": "C", "tokens_estimate": 50}]

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Resource Exhausted")
            return SourceValidationList(
                sources=[SourceValidation(
                    url="https://test.com", relevant=True, quality="high", use=True, reason="Good"
                )]
            )

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = invoke_side_effect
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _validate_section_sources(
                blog_title="Test",
                section={"id": "test", "title": "Test", "role": "problem"},
                sources=sources,
                key_manager=mock_key_manager,
            )

            mock_key_manager.mark_rate_limited.assert_called_once_with("key1")
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_raises_when_all_keys_exhausted(self):
        """Raises RuntimeError when all keys are exhausted."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "key1"
        mock_key_manager.get_next_key.return_value = None

        sources = [{"url": "https://test.com", "title": "T", "content": "C", "tokens_estimate": 50}]

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = Exception("429 quota exceeded")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            with pytest.raises(RuntimeError, match="exhausted"):
                await _validate_section_sources(
                    blog_title="Test",
                    section={"id": "test", "title": "Test", "role": "problem"},
                    sources=sources,
                    key_manager=mock_key_manager,
                )


# =============================================================================
# Validate Sources Node Tests
# =============================================================================


class TestValidateSourcesNode:
    """Tests for validate_sources_node function."""

    @pytest.mark.asyncio
    async def test_returns_validated_sources_and_advances_phase(self, sample_state):
        """Node returns validated_sources and advances to WRITING."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm:
                    # Return 4+ sources to avoid triggering retry logic
                    mock_validate.return_value = [
                        {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                        {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                    ]
                    mock_km.from_env.return_value = MagicMock()
                    mock_jm.return_value = MagicMock()

                    result = await validate_sources_node(sample_state)

                    assert "validated_sources" in result
                    assert result["current_phase"] == Phase.WRITING.value

    @pytest.mark.asyncio
    async def test_fails_without_sections(self):
        """Node fails if no sections in plan."""
        state: BlogAgentState = {
            "plan": {"sections": []},
            "research_cache": {},
        }

        result = await validate_sources_node(state)

        assert result["current_phase"] == Phase.FAILED.value
        assert "No sections" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_validates_all_sections(self, sample_state):
        """Validates sources for all sections."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                # Return 4+ sources to avoid triggering retry logic
                mock_validate.return_value = [
                    {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                    {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                    {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                    {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                ]
                mock_km.from_env.return_value = MagicMock()

                await validate_sources_node(sample_state)

                # Should validate 2 sections
                assert mock_validate.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_empty_cache(self):
        """Handles empty research cache gracefully."""
        state: BlogAgentState = {
            "plan": {
                "blog_title": "Test",
                "sections": [{"id": "test", "title": "Test", "role": "problem"}],
            },
            "research_cache": {},  # Empty cache
        }

        with patch("src.agent.nodes.KeyManager") as mock_km:
            mock_km.from_env.return_value = MagicMock()

            result = await validate_sources_node(state)

            # Should succeed with empty validated sources
            assert result["current_phase"] == Phase.WRITING.value
            assert result["validated_sources"]["test"] == []

    @pytest.mark.asyncio
    async def test_saves_checkpoint_on_success(self, sample_state):
        """Checkpoint is saved after successful validation."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm_class:
                    # Return 4+ sources to avoid triggering retry logic
                    mock_validate.return_value = [
                        {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                        {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                    ]
                    mock_km.from_env.return_value = MagicMock()
                    mock_jm = MagicMock()
                    mock_jm_class.return_value = mock_jm

                    await validate_sources_node(sample_state)

                    mock_jm.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_without_job_id(self, sample_plan, sample_research_cache):
        """No checkpoint if job_id not in state."""
        state: BlogAgentState = {
            "plan": sample_plan,
            "research_cache": sample_research_cache,
            # No job_id
        }

        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                with patch("src.agent.nodes.JobManager") as mock_jm_class:
                    # Return 4+ sources to avoid triggering retry logic
                    mock_validate.return_value = [
                        {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                        {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                        {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                    ]
                    mock_km.from_env.return_value = MagicMock()

                    await validate_sources_node(state)

                    mock_jm_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_validation_failure(self, sample_state):
        """Node handles LLM failure gracefully."""
        with patch("src.agent.nodes.KeyManager") as mock_km:
            with patch("src.agent.nodes._validate_section_sources") as mock_validate:
                mock_km.from_env.return_value = MagicMock()
                mock_validate.side_effect = RuntimeError("All API keys exhausted")

                result = await validate_sources_node(sample_state)

                assert result["current_phase"] == Phase.FAILED.value
                assert "exhausted" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_uses_blog_title_from_plan(self, sample_state):
        """Uses blog_title from plan for validation."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes.KeyManager") as mock_km:
                # Return 4+ sources to avoid triggering retry logic
                mock_validate.return_value = [
                    {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                    {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                    {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                    {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                ]
                mock_km.from_env.return_value = MagicMock()

                await validate_sources_node(sample_state)

                # Check that blog_title was passed to validation
                call_args = mock_validate.call_args_list[0]
                assert call_args.kwargs.get("blog_title") == "Semantic Caching for LLM Applications"


# =============================================================================
# Generate Alternative Queries Tests
# =============================================================================


class TestGenerateAlternativeQueries:
    """Tests for _generate_alternative_queries helper."""

    @pytest.mark.asyncio
    async def test_returns_alternative_queries(self):
        """Returns list of alternative queries."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = AlternativeQueries(
                queries=["alternative query 1", "alternative query 2"]
            )
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            result = await _generate_alternative_queries(
                blog_title="Test Blog",
                section={"id": "test", "title": "Test Section", "role": "problem"},
                original_queries=["original query"],
                key_manager=mock_key_manager,
            )

            assert len(result) == 2
            assert "alternative query 1" in result
            assert "alternative query 2" in result

    @pytest.mark.asyncio
    async def test_records_api_usage(self):
        """Records API usage after successful call."""
        mock_key_manager = MagicMock()
        mock_key_manager.get_best_key.return_value = "test_key"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = AlternativeQueries(
                queries=["alt1", "alt2"]
            )
            mock_llm.with_structured_output.return_value = mock_structured
            mock_llm_class.return_value = mock_llm

            await _generate_alternative_queries(
                blog_title="Test",
                section={"id": "test", "title": "Test", "role": "problem"},
                original_queries=["original"],
                key_manager=mock_key_manager,
            )

            mock_key_manager.record_usage.assert_called_once()


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestValidateSourcesRetryLogic:
    """Tests for retry logic in validate_sources_node."""

    @pytest.mark.asyncio
    async def test_retries_when_sources_insufficient(self, sample_state):
        """Retries with alternative queries when validated sources < 4."""
        call_count = 0

        def validate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call returns insufficient sources
            if call_count <= 2:  # 2 sections, first validation each
                return [{"url": "https://test1.com", "quality": "high", "reason": "Good"}]
            # Retry returns more sources
            return [
                {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                {"url": "https://test4.com", "quality": "high", "reason": "Great"},
            ]

        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes._generate_alternative_queries") as mock_gen_alt:
                with patch("src.agent.nodes._research_section") as mock_research:
                    with patch("src.agent.nodes.KeyManager") as mock_km:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_validate.side_effect = validate_side_effect
                            mock_gen_alt.return_value = ["alternative query"]
                            mock_research.return_value = (
                                [{"url": "https://new.com", "content": "New content"}],
                                {"newhash": {"url": "https://new.com", "content": "New content"}},
                            )
                            mock_km.from_env.return_value = MagicMock()
                            mock_jm.return_value = MagicMock()

                            result = await validate_sources_node(sample_state)

                            # Should have retried (generated alternative queries)
                            assert mock_gen_alt.call_count >= 1
                            assert result["current_phase"] == Phase.WRITING.value

    @pytest.mark.asyncio
    async def test_does_not_retry_when_sources_sufficient(self, sample_state):
        """Does not retry when validated sources >= 4."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes._generate_alternative_queries") as mock_gen_alt:
                with patch("src.agent.nodes.KeyManager") as mock_km:
                    with patch("src.agent.nodes.JobManager") as mock_jm:
                        # Return sufficient sources immediately
                        mock_validate.return_value = [
                            {"url": "https://test1.com", "quality": "high", "reason": "Good"},
                            {"url": "https://test2.com", "quality": "high", "reason": "Good"},
                            {"url": "https://test3.com", "quality": "medium", "reason": "OK"},
                            {"url": "https://test4.com", "quality": "high", "reason": "Great"},
                        ]
                        mock_km.from_env.return_value = MagicMock()
                        mock_jm.return_value = MagicMock()

                        result = await validate_sources_node(sample_state)

                        # Should NOT have generated alternative queries
                        mock_gen_alt.assert_not_called()
                        assert result["current_phase"] == Phase.WRITING.value

    @pytest.mark.asyncio
    async def test_merges_retry_sources_with_original(self, sample_state):
        """Retry sources are merged with original validated sources."""
        call_count = 0

        def validate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First section, first validation
                return [{"url": "https://original.com", "quality": "high", "reason": "Original"}]
            elif call_count == 2:  # First section, retry validation
                return [
                    {"url": "https://new1.com", "quality": "high", "reason": "New 1"},
                    {"url": "https://new2.com", "quality": "medium", "reason": "New 2"},
                    {"url": "https://new3.com", "quality": "high", "reason": "New 3"},
                ]
            else:  # Second section - return sufficient
                return [
                    {"url": "https://s1.com", "quality": "high", "reason": "S1"},
                    {"url": "https://s2.com", "quality": "high", "reason": "S2"},
                    {"url": "https://s3.com", "quality": "medium", "reason": "S3"},
                    {"url": "https://s4.com", "quality": "high", "reason": "S4"},
                ]

        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes._generate_alternative_queries") as mock_gen_alt:
                with patch("src.agent.nodes._research_section") as mock_research:
                    with patch("src.agent.nodes.KeyManager") as mock_km:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_validate.side_effect = validate_side_effect
                            mock_gen_alt.return_value = ["alt query"]
                            mock_research.return_value = (
                                [{"url": "https://new.com", "content": "New content"}],
                                {"newhash": {"url": "https://new.com", "content": "New content"}},
                            )
                            mock_km.from_env.return_value = MagicMock()
                            mock_jm.return_value = MagicMock()

                            result = await validate_sources_node(sample_state)

                            # First section should have 4 sources (1 original + 3 from retry)
                            validated = result["validated_sources"]
                            assert len(validated["problem"]) == 4
                            # Verify original source is preserved
                            urls = [s["url"] for s in validated["problem"]]
                            assert "https://original.com" in urls

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, sample_state):
        """Stops retrying after MAX_VALIDATION_RETRIES attempts."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes._generate_alternative_queries") as mock_gen_alt:
                with patch("src.agent.nodes._research_section") as mock_research:
                    with patch("src.agent.nodes.KeyManager") as mock_km:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            # Always return insufficient sources
                            mock_validate.return_value = [
                                {"url": "https://test.com", "quality": "high", "reason": "Only one"}
                            ]
                            mock_gen_alt.return_value = ["alt query"]
                            mock_research.return_value = ([], {})
                            mock_km.from_env.return_value = MagicMock()
                            mock_jm.return_value = MagicMock()

                            result = await validate_sources_node(sample_state)

                            # Should have tried max retries per section (2 sections * 2 retries = 4)
                            # But since each section has 2 max retries:
                            assert mock_gen_alt.call_count == 4  # 2 retries per 2 sections
                            # Should still complete (not fail)
                            assert result["current_phase"] == Phase.WRITING.value

    @pytest.mark.asyncio
    async def test_updated_cache_returned_after_retries(self, sample_state):
        """Research cache is updated and returned after retries."""
        with patch("src.agent.nodes._validate_section_sources") as mock_validate:
            with patch("src.agent.nodes._generate_alternative_queries") as mock_gen_alt:
                with patch("src.agent.nodes._research_section") as mock_research:
                    with patch("src.agent.nodes.KeyManager") as mock_km:
                        with patch("src.agent.nodes.JobManager") as mock_jm:
                            mock_validate.return_value = [
                                {"url": "https://test.com", "quality": "high", "reason": "One"}
                            ]
                            mock_gen_alt.return_value = ["alt query"]
                            # Return new cache entries from research
                            mock_research.return_value = (
                                [],
                                {"newhash123": {"url": "https://new.com", "content": "New"}},
                            )
                            mock_km.from_env.return_value = MagicMock()
                            mock_jm.return_value = MagicMock()

                            result = await validate_sources_node(sample_state)

                            # Research cache should include new entries from retries
                            assert "research_cache" in result
                            assert "newhash123" in result["research_cache"]
