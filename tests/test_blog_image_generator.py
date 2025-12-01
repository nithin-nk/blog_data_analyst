"""
Tests for the Blog Image Generator module.
"""

import base64
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.media.blog_image_generator import (
    BlogImageGenerator,
    ImageDescription,
    GeneratedBlogImage,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.google_api_key = "test-api-key"
    settings.blog_image_description_model = "gemini-2.5-flash"
    return settings


@pytest.fixture
def blog_image_generator(mock_settings):
    """Create a BlogImageGenerator instance with mocked settings."""
    with patch("src.media.blog_image_generator.get_settings") as mock_get_settings, \
         patch("src.media.blog_image_generator.genai") as mock_genai:
        mock_get_settings.return_value = mock_settings
        mock_genai.Client.return_value = MagicMock()
        return BlogImageGenerator()


@pytest.fixture
def sample_title():
    """Sample blog title."""
    return "Memory for AI Agents Using Mem0"


@pytest.fixture
def sample_content():
    """Sample blog content for testing."""
    return """# Memory for AI Agents Using Mem0

## Introduction
AI agents are becoming increasingly sophisticated, but many still lack
persistent memory capabilities. This tutorial explores how to implement
memory for AI agents using Mem0.

## Understanding Agent Memory
Memory allows agents to retain information across conversations and sessions.
This enables more personalized and context-aware interactions.

## Implementation with Mem0
Here's how to integrate Mem0 into your AI agent architecture.

## Conclusion
Memory is essential for building truly intelligent AI agents.
"""


class TestImageDescription:
    """Test ImageDescription model."""

    def test_image_description_creation(self):
        """Test creating an ImageDescription."""
        desc = ImageDescription(
            description="A futuristic AI brain with neural connections",
            alt_text="AI memory visualization",
            style="illustration",
        )

        assert desc.description == "A futuristic AI brain with neural connections"
        assert desc.alt_text == "AI memory visualization"
        assert desc.style == "illustration"


class TestGeneratedBlogImage:
    """Test GeneratedBlogImage model."""

    def test_generated_blog_image_creation(self):
        """Test creating a GeneratedBlogImage."""
        image = GeneratedBlogImage(
            title="Test Blog",
            description="A test image description",
            alt_text="Test alt text",
            style="photorealistic",
            image_base64="dGVzdA==",
            format="png",
        )

        assert image.title == "Test Blog"
        assert image.format == "png"
        assert image.image_base64 == "dGVzdA=="


class TestGenerateImageDescription:
    """Test image description generation."""

    def test_generate_description_success(
        self, blog_image_generator, sample_title, sample_content
    ):
        """Test successful description generation."""
        with patch("src.media.blog_image_generator.gemini_llm_call") as mock_llm:
            mock_desc = ImageDescription(
                description="A glowing neural network with memory nodes, representing AI agents storing and retrieving information in a digital space.",
                alt_text="AI agent memory visualization with neural network",
                style="illustration",
            )
            mock_llm.return_value = mock_desc

            result = blog_image_generator.generate_image_description(
                sample_title, sample_content
            )

            assert result.description is not None
            assert "neural network" in result.description.lower()
            assert result.style == "illustration"
            mock_llm.assert_called_once()

    def test_generate_description_truncates_long_content(
        self, blog_image_generator, sample_title
    ):
        """Test that long content is truncated."""
        long_content = "x" * 5000  # Longer than 3000 char limit

        with patch("src.media.blog_image_generator.gemini_llm_call") as mock_llm:
            mock_desc = ImageDescription(
                description="Test description",
                alt_text="Test alt",
                style="abstract",
            )
            mock_llm.return_value = mock_desc

            blog_image_generator.generate_image_description(sample_title, long_content)

            # Verify the prompt contains truncated content
            call_args = mock_llm.call_args
            prompt = call_args[0][0][0].content
            # The content preview in prompt should be truncated
            assert len(prompt) < 5000 + 1000  # Content + prompt overhead

    def test_generate_description_fallback_on_error(
        self, blog_image_generator, sample_title, sample_content
    ):
        """Test fallback when LLM call fails."""
        with patch("src.media.blog_image_generator.gemini_llm_call") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = blog_image_generator.generate_image_description(
                sample_title, sample_content
            )

            # Should return fallback description
            assert result.description is not None
            assert sample_title in result.alt_text
            assert result.style == "illustration"


class TestGenerateImage:
    """Test image generation with Gemini."""

    def test_generate_image_success(self, blog_image_generator):
        """Test successful image generation."""
        test_image_bytes = b"fake_image_data"

        # Mock the genai client response
        mock_part = MagicMock()
        mock_part.inline_data.data = test_image_bytes
        mock_part.inline_data.mime_type = "image/png"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        blog_image_generator.client.models.generate_content.return_value = mock_response

        image_bytes, fmt = blog_image_generator.generate_image(
            "A futuristic AI visualization"
        )

        assert image_bytes == test_image_bytes
        assert fmt == "png"
        blog_image_generator.client.models.generate_content.assert_called_once()

    def test_generate_image_no_image_in_response(self, blog_image_generator):
        """Test error when no image in response."""
        # Mock response with no inline_data
        mock_part = MagicMock()
        mock_part.inline_data = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        blog_image_generator.client.models.generate_content.return_value = mock_response

        with pytest.raises(ValueError, match="No image data"):
            blog_image_generator.generate_image("Test prompt")

    def test_generate_image_api_error(self, blog_image_generator):
        """Test error handling when API fails."""
        blog_image_generator.client.models.generate_content.side_effect = Exception(
            "API Error"
        )

        with pytest.raises(Exception, match="API Error"):
            blog_image_generator.generate_image("Test prompt")


class TestGenerateBlogImage:
    """Test the main blog image generation workflow."""

    def test_generate_blog_image_success(
        self, blog_image_generator, sample_title, sample_content
    ):
        """Test successful end-to-end blog image generation."""
        test_image_bytes = b"test_image_data"

        # Mock description generation
        with patch.object(
            blog_image_generator, "generate_image_description"
        ) as mock_desc, patch.object(
            blog_image_generator, "generate_image"
        ) as mock_gen:

            mock_desc.return_value = ImageDescription(
                description="AI memory visualization",
                alt_text="AI agent memory",
                style="illustration",
            )
            mock_gen.return_value = (test_image_bytes, "png")

            result = blog_image_generator.generate_blog_image(
                sample_title, sample_content
            )

            assert result.title == sample_title
            assert result.description == "AI memory visualization"
            assert result.alt_text == "AI agent memory"
            assert result.style == "illustration"
            assert result.format == "png"
            # Verify base64 encoding
            decoded = base64.b64decode(result.image_base64)
            assert decoded == test_image_bytes

    def test_generate_blog_image_with_progress_callback(
        self, blog_image_generator, sample_title, sample_content
    ):
        """Test blog image generation with progress callback."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)

        with patch.object(
            blog_image_generator, "generate_image_description"
        ) as mock_desc, patch.object(
            blog_image_generator, "generate_image"
        ) as mock_gen:

            mock_desc.return_value = ImageDescription(
                description="Test description",
                alt_text="Test alt",
                style="abstract",
            )
            mock_gen.return_value = (b"test", "png")

            blog_image_generator.generate_blog_image(
                sample_title, sample_content, progress_callback=progress_callback
            )

            # Verify progress messages were sent
            assert len(progress_messages) > 0
            assert any("BLOG IMAGE GENERATION" in msg for msg in progress_messages)
            assert any("description" in msg.lower() for msg in progress_messages)

    def test_generate_blog_image_fails_gracefully(
        self, blog_image_generator, sample_title, sample_content
    ):
        """Test that image generation failure is propagated."""
        with patch.object(
            blog_image_generator, "generate_image_description"
        ) as mock_desc, patch.object(
            blog_image_generator, "generate_image"
        ) as mock_gen:

            mock_desc.return_value = ImageDescription(
                description="Test",
                alt_text="Test",
                style="test",
            )
            mock_gen.side_effect = Exception("Generation failed")

            with pytest.raises(Exception, match="Generation failed"):
                blog_image_generator.generate_blog_image(sample_title, sample_content)


class TestSaveToDiagramsYaml:
    """Test saving blog image to diagrams.yaml."""

    def test_save_to_new_file(self, blog_image_generator, tmp_path):
        """Test saving to a new diagrams.yaml file."""
        blog_image = GeneratedBlogImage(
            title="Test Blog",
            description="Test description",
            alt_text="Test alt text",
            style="illustration",
            image_base64="dGVzdA==",
            format="png",
        )

        output_path = tmp_path / "diagrams.yaml"
        blog_image_generator.save_to_diagrams_yaml(blog_image, output_path)

        assert output_path.exists()

        import yaml

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        assert "blog_image" in data
        assert data["blog_image"]["title"] == "Test Blog"
        assert data["blog_image"]["description"] == "Test description"
        assert data["blog_image"]["alt_text"] == "Test alt text"
        assert data["blog_image"]["image_base64"] == "dGVzdA=="

    def test_save_to_existing_file_with_diagrams(self, blog_image_generator, tmp_path):
        """Test saving to existing diagrams.yaml that already has diagrams."""
        import yaml

        # Create existing file with diagram data
        output_path = tmp_path / "diagrams.yaml"
        existing_data = {
            "diagrams": [
                {
                    "heading": "Architecture",
                    "diagram_type": "flowchart",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "image_base64": "existing_diagram_base64",
                    "score": 9.0,
                }
            ],
            "total_diagrams": 1,
        }

        with open(output_path, "w") as f:
            yaml.dump(existing_data, f)

        # Add blog image
        blog_image = GeneratedBlogImage(
            title="Test Blog",
            description="Test description",
            alt_text="Test alt text",
            style="illustration",
            image_base64="new_image_base64",
            format="png",
        )

        blog_image_generator.save_to_diagrams_yaml(blog_image, output_path)

        with open(output_path, "r") as f:
            data = yaml.safe_load(f)

        # Verify existing diagrams are preserved
        assert len(data["diagrams"]) == 1
        assert data["diagrams"][0]["heading"] == "Architecture"
        assert data["total_diagrams"] == 1

        # Verify blog image was added
        assert "blog_image" in data
        assert data["blog_image"]["image_base64"] == "new_image_base64"

    def test_save_creates_parent_directories(self, blog_image_generator, tmp_path):
        """Test that parent directories are created if needed."""
        blog_image = GeneratedBlogImage(
            title="Test",
            description="Test",
            alt_text="Test",
            style="test",
            image_base64="dGVzdA==",
            format="png",
        )

        output_path = tmp_path / "nested" / "path" / "diagrams.yaml"
        blog_image_generator.save_to_diagrams_yaml(blog_image, output_path)

        assert output_path.exists()


class TestBlogImageGeneratorInit:
    """Test BlogImageGenerator initialization."""

    def test_init_without_api_key_raises_error(self):
        """Test that initialization fails without API key."""
        with patch("src.media.blog_image_generator.get_settings") as mock_settings:
            mock_settings.return_value.google_api_key = ""

            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                BlogImageGenerator()

    def test_init_with_valid_api_key(self, mock_settings):
        """Test successful initialization with valid API key."""
        with patch("src.media.blog_image_generator.get_settings") as mock_get_settings, \
             patch("src.media.blog_image_generator.genai") as mock_genai:
            mock_get_settings.return_value = mock_settings

            generator = BlogImageGenerator()

            assert generator.client is not None
            mock_genai.Client.assert_called_once_with(api_key="test-api-key")
