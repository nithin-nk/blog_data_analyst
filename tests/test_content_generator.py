import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from src.generation.content_generator import ContentGenerator

@pytest.fixture
def mock_settings():
    with patch('src.generation.content_generator.get_settings') as mock:
        mock.return_value.google_api_key = "test_key"
        yield mock

@pytest.fixture
def content_generator(mock_settings):
    with patch('src.generation.content_generator.ChatGoogleGenerativeAI') as mock_llm:
        generator = ContentGenerator()
        generator.llm = MagicMock()
        generator.preview_llm = MagicMock()
        return generator

def test_init(content_generator):
    assert content_generator.llm is not None
    assert content_generator.preview_llm is not None

def test_get_research_content(content_generator):
    research_data = {
        'contents': [
            {'url': 'http://example.com/1', 'markdown': 'Content 1'},
            {'url': 'http://example.com/2', 'snippet': 'Snippet 2'},
            {'url': 'http://example.com/3', 'markdown': 'Content 3'},
            {'url': 'http://example.com/4', 'markdown': 'Content 4'},
        ]
    }
    references = ['http://example.com/1', 'http://example.com/2', 'http://example.com/4']
    
    content = content_generator._get_research_content(references, research_data)
    
    assert "Source: http://example.com/1" in content
    assert "Content 1" in content
    assert "Source: http://example.com/2" in content
    assert "Snippet 2" in content
    assert "Source: http://example.com/4" in content
    assert "Content 4" in content
    assert "Source: http://example.com/3" not in content

def test_generate_section_content_success(content_generator):
    section = {'heading': 'Test Heading', 'summary': 'Test Summary'}
    research_content = "Test Research"
    previous_context = "Previous Context"
    topic = "Test Topic"
    
    content_generator.llm.invoke.return_value.content = "Generated Content"
    
    result = content_generator._generate_section_content(section, research_content, previous_context, topic)
    
    assert result == "Generated Content"
    content_generator.llm.invoke.assert_called_once()

def test_generate_section_content_fallback(content_generator):
    section = {'heading': 'Test Heading', 'summary': 'Test Summary'}
    research_content = "Test Research"
    previous_context = "Previous Context"
    topic = "Test Topic"
    
    # Primary fails, backup succeeds
    content_generator.llm.invoke.side_effect = Exception("Rate limit")
    content_generator.preview_llm.invoke.return_value.content = "Fallback Content"
    
    with patch('time.sleep') as mock_sleep: # Don't actually sleep
        result = content_generator._generate_section_content(section, research_content, previous_context, topic)
    
    assert result == "Fallback Content"
    content_generator.llm.invoke.assert_called_once()
    content_generator.preview_llm.invoke.assert_called_once()

def test_generate_blog_post(content_generator):
    outline_yaml = """
    topic: Test Topic
    sections:
      - heading: Section 1
        summary: Summary 1
        references: ["url1"]
      - heading: Section 2
        summary: Summary 2
        references: ["url2"]
    """
    research_yaml = """
    contents:
      - url: url1
        markdown: Content 1
      - url: url2
        markdown: Content 2
    """
    
    with patch('builtins.open', mock_open()) as mock_file:
        # Mock reading files - need to handle multiple files
        # This is tricky with mock_open for multiple files, so we'll mock _load_yaml instead
        with patch.object(content_generator, '_load_yaml') as mock_load:
            mock_load.side_effect = [
                {'topic': 'Test Topic', 'sections': [{'heading': 'S1', 'references': []}, {'heading': 'S2', 'references': []}]}, # outline
                {'contents': []} # research
            ]
            
            content_generator.llm.invoke.return_value.content = "Content"
            
            with patch('time.sleep') as mock_sleep:
                result = content_generator.generate_blog_post(Path('outline.yaml'), Path('research.yaml'))
            
            assert "# Test Topic" in result
            assert "## S1" in result
            assert "## S2" in result
            assert content_generator.llm.invoke.call_count == 2
