"""Unit tests for dynamic research functionality in section refine loop."""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.nodes import (
    _build_refiner_prompt,
    _build_scratchpad_entry,
    _refine_section,
)
from src.agent.state import CriticIssue, CriticScore, SectionCriticResult


class TestDynamicResearch:
    """Tests for dynamic research functionality."""

    def test_build_refiner_prompt_with_additional_sources(self):
        """Refiner prompt includes additional sources section when provided."""
        section = {
            "id": "caching",
            "title": "Caching Strategies",
            "role": "implementation",
            "target_words": 300,
        }
        content = "Current section content about caching."
        issues = [
            {
                "dimension": "completeness",
                "location": "paragraph 2",
                "problem": "Missing Redis vs Memcached comparison",
                "suggestion": "Add comparison",
            }
        ]
        scores = {
            "technical_accuracy": 7,
            "completeness": 6,
            "code_quality": 8,
            "clarity": 8,
            "voice": 7,
            "originality": 7,
            "length": 8,
            "diagram_quality": 10,
        }
        additional_sources = [
            {
                "url": "https://example.com/redis-vs-memcached",
                "title": "Redis vs Memcached Comparison",
                "content": "Redis supports data persistence while Memcached does not. Redis has richer data structures...",
            },
            {
                "url": "https://example.com/caching-guide",
                "title": "Caching Best Practices",
                "content": "When choosing a caching solution, consider your use case requirements...",
            },
        ]

        prompt = _build_refiner_prompt(
            section, content, issues, scores, additional_sources=additional_sources
        )

        # Check additional sources section exists
        assert "Additional Research (Just Fetched)" in prompt
        assert "Redis vs Memcached Comparison" in prompt
        assert "https://example.com/redis-vs-memcached" in prompt
        assert "Redis supports data persistence" in prompt
        assert "Caching Best Practices" in prompt
        assert "https://example.com/caching-guide" in prompt

        # Check instruction to use sources
        assert "incorporate missing information" in prompt.lower() or \
               "use these sources" in prompt.lower()

    def test_build_refiner_prompt_without_additional_sources(self):
        """Refiner prompt excludes additional sources section when not provided."""
        section = {"id": "test", "target_words": 200}
        content = "Test content"
        issues = []
        scores = {
            "technical_accuracy": 7,
            "completeness": 7,
            "code_quality": 7,
            "clarity": 7,
            "voice": 7,
            "originality": 7,
            "length": 7,
            "diagram_quality": 10,
        }

        prompt = _build_refiner_prompt(section, content, issues, scores)

        # Should not have additional sources section
        assert "Additional Research" not in prompt

    @pytest.mark.asyncio
    async def test_refine_section_with_additional_sources(self):
        """_refine_section passes additional_sources to prompt builder."""
        section = {"id": "test", "target_words": 200}
        content = "Test content"
        critic_result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=7,
                completeness=6,
                code_quality=7,
                clarity=7,
                voice=7,
                originality=7,
                length=7,
                diagram_quality=10,
            ),
            overall_pass=False,
            issues=[
                CriticIssue(
                    dimension="completeness",
                    location="overall",
                    problem="Missing info",
                    suggestion="Add info",
                )
            ],
            fact_check_needed=[],
        )
        additional_sources = [
            {"url": "https://example.com", "title": "Test", "content": "Test content"}
        ]

        mock_key_manager = MagicMock()
        mock_key_manager.get_current_key.return_value = "test_key"

        mock_response = MagicMock()
        mock_response.content = "Refined with additional sources"

        with patch("src.agent.nodes.ChatGoogleGenerativeAI") as mock_llm, \
             patch("src.agent.nodes._build_refiner_prompt") as mock_build_prompt:

            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            mock_build_prompt.return_value = "Test prompt"

            result = await _refine_section(
                section,
                content,
                critic_result,
                mock_key_manager,
                additional_sources=additional_sources,
            )

            # Verify additional_sources was passed to prompt builder
            mock_build_prompt.assert_called_once()
            call_kwargs = mock_build_prompt.call_args[1]
            assert call_kwargs["additional_sources"] == additional_sources

            # Verify refined content returned
            assert result == "Refined with additional sources"

    def test_build_scratchpad_entry_with_research_tracking(self):
        """Scratchpad entry includes research tracking when provided."""
        critic_result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=8,
                completeness=8,
                code_quality=8,
                clarity=8,
                voice=8,
                originality=8,
                length=8,
                diagram_quality=10,
            ),
            overall_pass=True,
            issues=[],
            fact_check_needed=[],
        )

        research_queries = [
            "Redis vs Memcached comparison 2025",
            "Redis Memcached use cases",
        ]
        sources_fetched = 6

        entry = _build_scratchpad_entry(
            attempt=1,
            critic_result=critic_result,
            research_queries=research_queries,
            sources_fetched=sources_fetched,
        )

        # Check research tracking fields
        assert entry["research_performed"] is True
        assert entry["research_queries"] == research_queries
        assert entry["sources_fetched"] == 6

    def test_build_scratchpad_entry_without_research(self):
        """Scratchpad entry marks research_performed as False when no research."""
        critic_result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=7,
                completeness=7,
                code_quality=7,
                clarity=7,
                voice=7,
                originality=7,
                length=7,
                diagram_quality=10,
            ),
            overall_pass=False,
            issues=[],
            fact_check_needed=[],
        )

        entry = _build_scratchpad_entry(
            attempt=1,
            critic_result=critic_result,
            research_queries=None,
            sources_fetched=0,
        )

        # Check research_performed is False
        assert entry["research_performed"] is False
        assert "research_queries" not in entry
        assert "sources_fetched" not in entry

    def test_build_scratchpad_entry_initial_attempt_no_research(self):
        """Initial scratchpad entry (attempt 0) has research_performed=False."""
        critic_result = SectionCriticResult(
            scores=CriticScore(
                technical_accuracy=7,
                completeness=6,
                code_quality=5,
                clarity=8,
                voice=8,
                originality=7,
                length=7,
                diagram_quality=10,
            ),
            overall_pass=False,
            issues=[
                CriticIssue(
                    dimension="completeness",
                    location="overall",
                    problem="Missing examples",
                    suggestion="Add examples",
                )
            ],
            fact_check_needed=[],
        )

        entry = _build_scratchpad_entry(attempt=0, critic_result=critic_result)

        # Initial write should have research_performed=False
        assert entry["attempt"] == 0
        assert entry["research_performed"] is False
        assert entry["score"] == 7.25  # (7+6+5+8+8+7+7+10)/8 = 7.25
