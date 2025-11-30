"""
Tests for the Outline Generator module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.planning.outline_generator import OutlineGenerator, BlogOutline, OutlineSection, OutlineMetadata
from src.research.content_extractor import AggregatedExtractedContent, ExtractedContent

@pytest.fixture
def mock_research_data():
    """Create mock research data."""
    return AggregatedExtractedContent(
        topic="Test Topic",
        statistics={"successful": 1, "failed": 0, "total_urls": 1},
        contents=[
            ExtractedContent(
                url="http://example.com",
                title="Example Article",
                snippet="This is a snippet.",
                headings=["H1: Intro", "H2: Details"],
                success=True
            )
        ]
    )

@pytest.fixture
def mock_outline_response():
    """Create a mock BlogOutline response."""
    return BlogOutline(
        topic="Test Topic",
        sections=[
            OutlineSection(
                heading="Introduction",
                summary="Intro summary",
                references=["http://example.com"]
            ),
            OutlineSection(
                heading="Body",
                summary="Body summary",
                references=[]
            )
        ],
        metadata=OutlineMetadata(
            target_audience="Developers",
            difficulty="Intermediate",
            estimated_reading_time="5 mins"
        )
    )

@pytest.mark.asyncio
async def test_generate_outline_success(mock_research_data, mock_outline_response):
    """Test successful outline generation."""
    with patch("src.planning.outline_generator.ChatGoogleGenerativeAI") as MockLLM:
        # Setup mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_outline_response)
        MockLLM.return_value = mock_llm_instance
        
        # Initialize generator
        generator = OutlineGenerator(model_name="gemini-2.0-flash")
        
        # Run generation
        outline = await generator.generate("Test Topic", mock_research_data)
        
        # Verify results
        assert isinstance(outline, BlogOutline)
        assert outline.topic == "Test Topic"
        assert len(outline.sections) == 2
        assert outline.sections[0].heading == "Introduction"
        assert outline.metadata.difficulty == "Intermediate"
        
        # Verify LLM was called correctly
        mock_llm_instance.with_structured_output.return_value.ainvoke.assert_called_once()
        call_args = mock_llm_instance.with_structured_output.return_value.ainvoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
        assert "Test Topic" in call_args[1]["content"]
        assert "Example Article" in call_args[1]["content"]

@pytest.mark.asyncio
async def test_generate_outline_error(mock_research_data):
    """Test error handling during generation."""
    with patch("src.planning.outline_generator.ChatGoogleGenerativeAI") as MockLLM:
        # Setup mock LLM to raise exception
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        MockLLM.return_value = mock_llm_instance
        
        generator = OutlineGenerator()
        
        with pytest.raises(Exception, match="API Error"):
            await generator.generate("Test Topic", mock_research_data)

def test_format_research_data(mock_research_data):
    """Test research data formatting."""
    generator = OutlineGenerator()
    formatted = generator._format_research_data(mock_research_data)
    
    assert "Research Summary for 'Test Topic'" in formatted
    assert "Analyzed 1 URLs" in formatted
    assert "Source 1: Example Article" in formatted
    assert "URL: http://example.com" in formatted
    assert "Snippet: This is a snippet." in formatted
    assert "Key Sections:" in formatted
    assert "- H1: Intro" in formatted
