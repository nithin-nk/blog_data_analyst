"""
Unit tests for the MetadataGenerator module.

Tests metadata generation, review loop, and saving functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.generation.metadata_generator import (
    MetadataGenerator,
    BlogMetadata,
    MetadataReview,
    MetadataResult,
)


# Sample test content
SAMPLE_TOPIC = "Getting Started with Docker Containers"
SAMPLE_CONTENT = """
# Getting Started with Docker Containers

Docker is a platform for developing, shipping, and running applications in containers.

## What is Docker?

Docker enables you to separate your applications from your infrastructure.

- Containers are lightweight
- They share the host OS kernel
- Fast startup times

## Installing Docker

To install Docker on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## Conclusion

Docker simplifies deployment and improves consistency across environments.
"""


class TestBlogMetadata:
    """Tests for BlogMetadata model."""
    
    def test_valid_metadata(self):
        metadata = BlogMetadata(
            titles=["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            tags=["docker", "containers", "devops"],
            search_description="Learn Docker basics in this guide.",
        )
        assert len(metadata.titles) == 5
        assert len(metadata.tags) == 3
        assert len(metadata.search_description) < 160
    
    def test_tags_minimum(self):
        metadata = BlogMetadata(
            titles=["T1", "T2", "T3", "T4", "T5"],
            tags=["tag1", "tag2", "tag3"],
            search_description="Test description",
        )
        assert len(metadata.tags) >= 3
    
    def test_tags_maximum(self):
        metadata = BlogMetadata(
            titles=["T1", "T2", "T3", "T4", "T5"],
            tags=["tag1", "tag2", "tag3", "tag4"],
            search_description="Test description",
        )
        assert len(metadata.tags) <= 4


class TestMetadataReview:
    """Tests for MetadataReview model."""
    
    def test_valid_review(self):
        review = MetadataReview(
            score=8.5,
            feedback=["Improve title clarity", "Add more keywords"],
        )
        assert review.score == 8.5
        assert len(review.feedback) == 2
    
    def test_score_bounds(self):
        review = MetadataReview(score=10.0, feedback=[])
        assert review.score == 10.0
        
        review = MetadataReview(score=1.0, feedback=[])
        assert review.score == 1.0


class TestMetadataResult:
    """Tests for MetadataResult dataclass."""
    
    def test_basic_result(self):
        result = MetadataResult(
            titles=["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            tags=["tag1", "tag2", "tag3"],
            search_description="Test description",
            review_score=8.0,
            iterations=2,
        )
        assert result.selected_title is None
        assert result.review_score == 8.0
    
    def test_with_selected_title(self):
        result = MetadataResult(
            titles=["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            tags=["tag1", "tag2", "tag3"],
            search_description="Test description",
            selected_title="Title 2",
        )
        assert result.selected_title == "Title 2"


class TestMetadataGenerator:
    """Tests for MetadataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a MetadataGenerator instance with mocked settings."""
        with patch('src.generation.metadata_generator.get_settings') as mock_settings:
            settings = Mock()
            settings.google_api_key = "test_key"
            settings.google_api_key_1 = ""
            settings.google_api_key_2 = ""
            settings.max_retries = 1
            settings.retry_delay = 0
            mock_settings.return_value = settings
            return MetadataGenerator()
    
    def test_generator_initialization(self, generator):
        """Test MetadataGenerator initializes correctly."""
        assert generator is not None
        assert generator.model_name == "gemini-2.5-flash"
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_generate_metadata_success(self, mock_llm, generator):
        """Test successful metadata generation."""
        mock_llm.return_value = '''```json
{
    "titles": ["Docker Guide", "Learn Docker", "Docker Basics", "Docker Tutorial", "Master Docker"],
    "tags": ["docker", "containers", "devops"],
    "search_description": "Complete guide to Docker containers."
}
```'''
        
        metadata = generator._generate_metadata(SAMPLE_TOPIC, SAMPLE_CONTENT)
        
        assert len(metadata.titles) == 5
        assert len(metadata.tags) == 3
        assert "Docker" in metadata.titles[0] or "docker" in metadata.tags
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_generate_metadata_fallback(self, mock_llm, generator):
        """Test fallback metadata on generation failure."""
        mock_llm.side_effect = Exception("API Error")
        
        metadata = generator._generate_metadata(SAMPLE_TOPIC, SAMPLE_CONTENT)
        
        assert len(metadata.titles) == 5
        assert SAMPLE_TOPIC in metadata.titles[0]
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_review_metadata_success(self, mock_llm, generator):
        """Test successful metadata review."""
        mock_llm.return_value = '''```json
{
    "score": 8.5,
    "feedback": ["Consider adding more keywords", "Title could be more specific"]
}
```'''
        
        metadata = BlogMetadata(
            titles=["T1", "T2", "T3", "T4", "T5"],
            tags=["tag1", "tag2", "tag3"],
            search_description="Test description",
        )
        
        review = generator._review_metadata(metadata, SAMPLE_TOPIC)
        
        assert review.score == 8.5
        assert len(review.feedback) == 2
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_generate_with_review_passes_first_iteration(self, mock_llm, generator):
        """Test generation passes on first iteration with good score."""
        # First call: generate metadata
        # Second call: review metadata with score >= 9
        mock_llm.side_effect = [
            '''{"titles": ["T1", "T2", "T3", "T4", "T5"], "tags": ["a", "b", "c"], "search_description": "Desc"}''',
            '''{"score": 9.5, "feedback": []}''',
        ]
        
        result = generator.generate_with_review(
            topic=SAMPLE_TOPIC,
            content=SAMPLE_CONTENT,
            max_iterations=3,
            score_threshold=9.0,
        )
        
        assert result.review_score == 9.5
        assert result.iterations == 1
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_generate_with_review_iterates(self, mock_llm, generator):
        """Test generation iterates when score is below threshold."""
        # First iteration: score 7.0
        # Second iteration: score 9.5
        mock_llm.side_effect = [
            '''{"titles": ["T1", "T2", "T3", "T4", "T5"], "tags": ["a", "b", "c"], "search_description": "Desc"}''',
            '''{"score": 7.0, "feedback": ["Improve titles"]}''',
            '''{"titles": ["Better T1", "Better T2", "Better T3", "Better T4", "Better T5"], "tags": ["x", "y", "z"], "search_description": "Better desc"}''',
            '''{"score": 9.5, "feedback": []}''',
        ]
        
        result = generator.generate_with_review(
            topic=SAMPLE_TOPIC,
            content=SAMPLE_CONTENT,
            max_iterations=3,
            score_threshold=9.0,
        )
        
        assert result.review_score == 9.5
        assert result.iterations == 2
    
    @patch('src.generation.metadata_generator.gemini_llm_call')
    def test_generate_with_review_max_iterations(self, mock_llm, generator):
        """Test generation respects max iterations."""
        # All iterations below threshold
        mock_llm.side_effect = [
            '''{"titles": ["T1", "T2", "T3", "T4", "T5"], "tags": ["a", "b", "c"], "search_description": "Desc"}''',
            '''{"score": 6.0, "feedback": ["Improve"]}''',
            '''{"titles": ["T1", "T2", "T3", "T4", "T5"], "tags": ["a", "b", "c"], "search_description": "Desc"}''',
            '''{"score": 7.0, "feedback": ["Improve more"]}''',
            '''{"titles": ["T1", "T2", "T3", "T4", "T5"], "tags": ["a", "b", "c"], "search_description": "Desc"}''',
            '''{"score": 8.0, "feedback": ["Almost there"]}''',
        ]
        
        result = generator.generate_with_review(
            topic=SAMPLE_TOPIC,
            content=SAMPLE_CONTENT,
            max_iterations=3,
            score_threshold=9.0,
        )
        
        assert result.iterations == 3
        assert result.review_score == 8.0  # Best score


class TestMetadataSaving:
    """Tests for saving metadata."""
    
    @pytest.fixture
    def generator(self):
        with patch('src.generation.metadata_generator.get_settings') as mock_settings:
            settings = Mock()
            mock_settings.return_value = settings
            return MetadataGenerator()
    
    def test_save_metadata(self, generator, tmp_path):
        """Test saving metadata to YAML file."""
        result = MetadataResult(
            titles=["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            tags=["docker", "containers", "devops"],
            search_description="Learn Docker in this comprehensive guide.",
            selected_title="Title 2",
            review_score=9.0,
            iterations=2,
        )
        
        output_path = tmp_path / "metadata.yaml"
        generator.save_metadata(result, output_path)
        
        assert output_path.exists()
        
        import yaml
        with open(output_path) as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["title"] == "Title 2"
        assert saved_data["tags"] == ["docker", "containers", "devops"]
        assert saved_data["search_description"] == "Learn Docker in this comprehensive guide."
        assert saved_data["review_score"] == 9.0
        assert saved_data["iterations"] == 2
        assert len(saved_data["all_titles"]) == 5
    
    def test_save_metadata_without_selection(self, generator, tmp_path):
        """Test saving metadata uses first title when none selected."""
        result = MetadataResult(
            titles=["First Title", "Title 2", "Title 3", "Title 4", "Title 5"],
            tags=["tag1", "tag2", "tag3"],
            search_description="Description",
        )
        
        output_path = tmp_path / "metadata.yaml"
        generator.save_metadata(result, output_path)
        
        import yaml
        with open(output_path) as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["title"] == "First Title"


class TestTitleSelection:
    """Tests for title selection prompt."""
    
    @pytest.fixture
    def generator(self):
        with patch('src.generation.metadata_generator.get_settings') as mock_settings:
            settings = Mock()
            mock_settings.return_value = settings
            return MetadataGenerator()
    
    def test_prompt_title_selection_with_console(self, generator):
        """Test title selection with Rich console."""
        from unittest.mock import MagicMock
        
        mock_console = MagicMock()
        
        # Mock the rich.prompt.Prompt.ask to return "2"
        with patch('rich.prompt.Prompt.ask', return_value="2"):
            titles = ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"]
            selected = generator.prompt_title_selection(titles, console=mock_console)
            
            assert selected == "Title 2"
    
    def test_prompt_title_selection_without_console(self, generator, monkeypatch):
        """Test title selection without Rich console (fallback)."""
        # Mock input() to return "3"
        monkeypatch.setattr('builtins.input', lambda _: "3")
        
        titles = ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"]
        selected = generator.prompt_title_selection(titles, console=None)
        
        assert selected == "Title 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
