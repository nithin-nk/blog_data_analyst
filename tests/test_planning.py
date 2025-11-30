"""
Tests for the Question Generator module.

Tests cover:
- Question generation with topic only
- Question generation with topic + context
- Structured output validation
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import asyncio
import yaml

from src.planning.question_generator import (
    QuestionGenerator,
    ResearchQuestions,
)


class TestResearchQuestions:
    """Tests for the ResearchQuestions Pydantic model."""
    
    def test_valid_questions(self):
        """Test creating valid research questions."""
        questions = ResearchQuestions(
            questions=[
                "mem0 AI agent memory Python",
                "how to implement mem0 tutorial",
                "mem0 vs langchain memory comparison",
            ],
            categories_covered=["core concepts", "how-to", "comparisons"],
        )
        assert len(questions.questions) == 3
        assert len(questions.categories_covered) == 3
    
    def test_empty_categories_default(self):
        """Test that categories_covered defaults to empty list."""
        questions = ResearchQuestions(
            questions=["test query"],
        )
        assert questions.categories_covered == []
    
    def test_minimum_questions(self):
        """Test that at least one question is required."""
        with pytest.raises(ValueError):
            ResearchQuestions(questions=[])


class TestQuestionGenerator:
    """Tests for the QuestionGenerator class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings with API key."""
        with patch("src.planning.question_generator.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = "test-api-key"
            mock.return_value = settings
            yield settings
    
    @pytest.fixture
    def generator(self, mock_settings):
        """Create a QuestionGenerator instance with mocked settings."""
        return QuestionGenerator()
    
    def test_init_default_model(self, generator):
        """Test default model is gemini-2.0-flash."""
        assert generator.model_name == "gemini-2.0-flash"
    
    def test_init_custom_model(self, mock_settings):
        """Test custom model can be specified."""
        gen = QuestionGenerator(model_name="gemini-1.5-pro")
        assert gen.model_name == "gemini-1.5-pro"
    
    def test_question_categories_defined(self, generator):
        """Test that question categories are defined."""
        assert len(generator.QUESTION_CATEGORIES) == 7
        assert any("Core concepts" in cat for cat in generator.QUESTION_CATEGORIES)
    
    def test_build_prompt_topic_only(self, generator):
        """Test prompt building with topic only."""
        topic = "AI agent memory systems"
        prompt = generator._build_prompt(topic)
        
        assert "AI agent memory systems" in prompt
        assert "TOPIC:" in prompt
        assert "ADDITIONAL CONTEXT" not in prompt
    
    def test_build_prompt_with_context(self, generator):
        """Test prompt building with topic and context."""
        topic = "AI agent memory systems"
        context = "Focus on Python implementation"
        prompt = generator._build_prompt(topic, context)
        
        assert "AI agent memory systems" in prompt
        assert "TOPIC:" in prompt
        assert "ADDITIONAL CONTEXT" in prompt
        assert "Focus on Python implementation" in prompt
    
    def test_system_prompt_contains_guidelines(self, generator):
        """Test that system prompt contains key guidelines."""
        assert "OPTIMIZED FOR GOOGLE SEARCH" in generator.SYSTEM_PROMPT
        assert "exactly 10 search queries" in generator.SYSTEM_PROMPT
        assert "Core concepts" in generator.SYSTEM_PROMPT
    
    @pytest.mark.asyncio
    async def test_generate_returns_structured_output(self, generator):
        """Test that generate returns ResearchQuestions."""
        mock_result = ResearchQuestions(
            questions=[
                "mem0 AI agent memory implementation",
                "mem0 Python tutorial getting started",
                "mem0 architecture design patterns",
                "mem0 vs langchain memory comparison",
                "mem0 best practices production",
                "what is mem0 AI memory",
                "mem0 real world examples",
                "mem0 custom memory extraction",
                "mem0 integration guide",
                "mem0 performance optimization",
            ],
            categories_covered=[
                "core concepts",
                "how-to",
                "architecture",
                "comparisons",
                "best practices",
                "real-world examples",
            ],
        )
        
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)
        generator._llm = mock_llm
        
        result = await generator.generate("mem0 AI agent memory")
        
        assert isinstance(result, ResearchQuestions)
        assert len(result.questions) == 10
        mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_context(self, generator):
        """Test generate incorporates context."""
        mock_result = ResearchQuestions(
            questions=["python mem0 implementation guide"],
            categories_covered=["how-to"],
        )
        
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)
        generator._llm = mock_llm
        
        await generator.generate(
            topic="mem0 memory system",
            context="Focus on Python examples"
        )
        
        call_args = mock_llm.ainvoke.call_args[0][0]
        user_message = call_args[1]["content"]
        assert "Focus on Python examples" in user_message
    
    @pytest.mark.asyncio
    async def test_generate_handles_llm_error(self, generator):
        """Test that LLM errors are propagated."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        generator._llm = mock_llm
        
        with pytest.raises(Exception, match="API Error"):
            await generator.generate("test topic")
    
    def test_generate_sync_wrapper(self, generator):
        """Test synchronous wrapper calls async generate."""
        mock_result = ResearchQuestions(
            questions=["test query"],
            categories_covered=["test"],
        )
        
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)
        generator._llm = mock_llm
        
        result = generator.generate_sync("test topic")
        
        assert isinstance(result, ResearchQuestions)
    
    def test_llm_lazy_initialization(self, mock_settings):
        """Test LLM is lazily initialized."""
        generator = QuestionGenerator()
        assert generator._llm is None
    
    def test_llm_requires_api_key(self):
        """Test LLM initialization fails without API key."""
        with patch("src.planning.question_generator.get_settings") as mock:
            settings = MagicMock()
            settings.google_api_key = ""
            mock.return_value = settings
            
            generator = QuestionGenerator()
            
            with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
                _ = generator.llm


class TestResearchQuestionsValidation:
    """Additional validation tests for ResearchQuestions."""
    
    def test_questions_max_length(self):
        """Test questions list respects max length."""
        # Should accept up to 15 questions
        questions = ResearchQuestions(
            questions=[f"query {i}" for i in range(15)],
        )
        assert len(questions.questions) == 15
    
    def test_questions_over_max_raises_error(self):
        """Test more than 15 questions raises error."""
        with pytest.raises(ValueError):
            ResearchQuestions(
                questions=[f"query {i}" for i in range(16)],
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_questions_live():
    """
    Integration test: Calls the actual Gemini LLM and prints generated questions for a sample topic.
    Requires GOOGLE_API_KEY to be set in environment or .env.
    """
    from src.planning.question_generator import QuestionGenerator
    
    topic = "How to build memory for AI agents using mem0"
    context = "Focus on the open source Python framework, include practical implementation examples"
    
    generator = QuestionGenerator(model_name="gemini-2.0-flash")
    
    result = await generator.generate(topic, context)
    
    print("\n=== LIVE LLM GENERATED QUESTIONS ===")
    for i, q in enumerate(result.questions, 1):
        print(f"{i:2}. {q}")
    print("Categories covered:", ", ".join(result.categories_covered))
    print("====================================\n")
    
    assert result.questions, "No questions generated by LLM"
    assert isinstance(result.questions, list)
    assert len(result.questions) >= 5  # Should generate at least 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_questions_live_yaml():
    """
    Integration test: Calls the actual Gemini LLM and prints YAML output for a sample technical topic.
    """
    from src.planning.question_generator import QuestionGenerator
    
    topic = "LangChain framework for AI agents"
    context = "Focus on Python implementation, integration with OpenAI, and recent developments."
    
    generator = QuestionGenerator(model_name="gemini-2.0-flash")
    result = await generator.generate(topic, context)
    
    output = {
        "topic": topic,
        "questions": result.questions,
        "categories_covered": result.categories_covered,
    }
    print("\n=== LIVE LLM GENERATED YAML OUTPUT ===")
    print(yaml.dump(output, sort_keys=False, allow_unicode=True))
    print("======================================\n")
    
    assert result.questions, "No questions generated by LLM"
    assert isinstance(result.questions, list)
    assert len(result.questions) >= 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_questions_mem0_memory_yaml():
    """
    Integration test: Calls the actual Gemini LLM and prints YAML output for 'Memory for AI agents using mem0'.
    """
    from src.planning.question_generator import QuestionGenerator
    
    topic = "Memory for AI agents using mem0"
    context = "Focus on open source Python implementation, architecture, best practices, and recent developments."
    
    generator = QuestionGenerator(model_name="gemini-2.0-flash")
    result = await generator.generate(topic, context)
    
    output = {
        "topic": topic,
        "questions": result.questions,
        "categories_covered": result.categories_covered,
    }
    print("\n=== LIVE LLM GENERATED YAML OUTPUT (mem0 memory) ===")
    print(yaml.dump(output, sort_keys=False, allow_unicode=True))
    print("======================================\n")
    
    assert result.questions, "No questions generated by LLM"
    assert isinstance(result.questions, list)
    assert len(result.questions) >= 5
