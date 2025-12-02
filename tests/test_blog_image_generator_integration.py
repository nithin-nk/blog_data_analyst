"""
Integration tests for Blog Image Generator using actual LLM calls.

These tests make real API calls to Google Gemini and require:
- GOOGLE_API_KEY environment variable to be set
- Network connectivity

Run with: pytest tests/test_blog_image_generator_integration.py -v -s

Note: Image generation tests may be skipped due to rate limits on free tier.
"""

import base64
import os
import pytest
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.media.blog_image_generator import (
    BlogImageGenerator,
    ImageDescription,
    GeneratedBlogImage,
)


# Skip all tests if GOOGLE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)


def is_rate_limit_error(exc_info) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exc_info.value) if hasattr(exc_info, 'value') else str(exc_info)
    return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str


@pytest.fixture
def blog_image_generator():
    """Create a real BlogImageGenerator instance."""
    return BlogImageGenerator()


@pytest.fixture
def sample_blog_data():
    """Sample blog data for testing."""
    return {
        "title": "Memory for AI Agents Using Mem0",
        "content": """# Memory for AI Agents Using Mem0

## Introduction
AI agents are becoming increasingly sophisticated, but many still lack
persistent memory capabilities. This tutorial explores how to implement
memory for AI agents using Mem0, an open-source memory layer.

## Understanding Agent Memory
Memory allows agents to retain information across conversations and sessions.
There are several types of memory:
- Short-term memory: Recent conversation context
- Long-term memory: Persistent facts and preferences
- Episodic memory: Specific events and experiences

## Why Memory Matters
Without memory, AI agents treat each interaction as completely new.
This leads to:
- Repetitive questions
- Lack of personalization
- Inability to learn from past interactions

## Implementing Mem0
Mem0 provides a simple API for adding memory to your agents:

```python
from mem0 import Memory

m = Memory()
m.add("User prefers Python over JavaScript", user_id="alice")
memories = m.search("programming preferences", user_id="alice")
```

## Architecture Overview
The memory system consists of:
1. Vector database for semantic search
2. Graph database for relationships
3. LLM for memory extraction and retrieval

## Conclusion
Adding memory to AI agents significantly improves user experience
and enables more natural, contextual interactions.
""",
    }


class TestImageDescriptionGeneration:
    """Integration tests for image description generation."""

    def test_generate_description_returns_valid_structure(
        self, blog_image_generator, sample_blog_data
    ):
        """Test that description generation returns valid ImageDescription."""
        result = blog_image_generator.generate_image_description(
            title=sample_blog_data["title"],
            content=sample_blog_data["content"],
        )

        # Verify return type
        assert isinstance(result, ImageDescription)

        # Verify required fields are populated
        assert result.description is not None
        assert len(result.description) > 20  # Should be meaningful

        assert result.alt_text is not None
        assert len(result.alt_text) > 0
        assert len(result.alt_text) <= 150  # Alt text should be concise

        assert result.style is not None
        assert result.style in [
            "illustration",
            "abstract",
            "photorealistic",
            "minimal",
            "3d",
            "digital art",
            "concept art",
        ] or len(result.style) > 0  # Allow other styles

        # Print for manual inspection
        print("\n" + "=" * 60)
        print("GENERATED IMAGE DESCRIPTION")
        print("=" * 60)
        print(f"Style: {result.style}")
        print(f"Alt Text: {result.alt_text}")
        print(f"Description: {result.description}")
        print("=" * 60)

    def test_generate_description_is_contextually_relevant(
        self, blog_image_generator, sample_blog_data
    ):
        """Test that generated description is relevant to blog content."""
        result = blog_image_generator.generate_image_description(
            title=sample_blog_data["title"],
            content=sample_blog_data["content"],
        )

        description_lower = result.description.lower()

        # Should reference concepts from the blog
        relevant_terms = [
            "ai",
            "agent",
            "memory",
            "neural",
            "brain",
            "data",
            "network",
            "digital",
            "technology",
            "intelligence",
            "learning",
            "connection",
        ]

        found_terms = [term for term in relevant_terms if term in description_lower]
        assert (
            len(found_terms) >= 1
        ), f"Description should reference blog concepts. Found: {found_terms}"

        print(f"\nRelevant terms found in description: {found_terms}")


class TestImageGeneration:
    """Integration tests for actual image generation."""

    def test_generate_image_returns_valid_bytes(self, blog_image_generator):
        """Test that image generation returns valid image bytes."""
        description = (
            "A futuristic visualization of AI memory systems, "
            "showing glowing neural pathways and data nodes connected "
            "in a digital network. Modern tech aesthetic with blue "
            "and purple color scheme."
        )

        try:
            image_bytes, fmt = blog_image_generator.generate_image(description)

            # Verify we got image data
            assert image_bytes is not None
            assert len(image_bytes) > 1000  # Should be a real image, not empty

            # Verify format
            assert fmt in ["png", "jpeg", "jpg", "webp"]

            # Verify it's valid image data (PNG starts with specific bytes)
            if fmt == "png":
                assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Invalid PNG header"

            print(f"\nGenerated image: {len(image_bytes):,} bytes, format: {fmt}")
            
        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit (free tier)")
            raise


class TestEndToEndBlogImageGeneration:
    """End-to-end integration tests for complete blog image workflow."""

    def test_full_blog_image_generation(
        self, blog_image_generator, sample_blog_data, tmp_path
    ):
        """Test complete workflow: description + image generation + saving."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        try:
            # Generate blog image
            result = blog_image_generator.generate_blog_image(
                title=sample_blog_data["title"],
                content=sample_blog_data["content"],
                progress_callback=progress_callback,
            )

            # Verify result structure
            assert isinstance(result, GeneratedBlogImage)
            assert result.title == sample_blog_data["title"]
            assert len(result.description) > 20
            assert len(result.alt_text) > 0
            assert result.style is not None
            assert result.format in ["png", "jpeg", "jpg", "webp"]

            # Verify base64 is valid
            assert len(result.image_base64) > 1000
            decoded = base64.b64decode(result.image_base64)
            assert len(decoded) > 1000  # Real image data

            # Verify progress messages were sent
            assert len(progress_messages) > 0
            assert any("BLOG IMAGE GENERATION" in msg for msg in progress_messages)

            # Save to YAML and verify
            diagrams_path = tmp_path / "diagrams.yaml"
            blog_image_generator.save_to_diagrams_yaml(result, diagrams_path)

            assert diagrams_path.exists()

            import yaml

            with open(diagrams_path, "r") as f:
                data = yaml.safe_load(f)

            assert "blog_image" in data
            assert data["blog_image"]["title"] == sample_blog_data["title"]
            assert len(data["blog_image"]["image_base64"]) > 1000

            # Save actual image to file for manual inspection
            image_path = tmp_path / f"generated_blog_image.{result.format}"
            with open(image_path, "wb") as f:
                f.write(decoded)

            print("\n" + "=" * 60)
            print("END-TO-END TEST RESULTS")
            print("=" * 60)
            print(f"Title: {result.title}")
            print(f"Style: {result.style}")
            print(f"Alt Text: {result.alt_text}")
            print(f"Description: {result.description[:100]}...")
            print(f"Image Format: {result.format}")
            print(f"Image Size: {len(decoded):,} bytes")
            print(f"Base64 Length: {len(result.image_base64):,} chars")
            print(f"Saved to: {image_path}")
            print(f"YAML saved to: {diagrams_path}")
            print("=" * 60)
            
        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit (free tier)")
            raise

    def test_different_blog_topics(self, blog_image_generator):
        """Test image generation for different blog topics."""
        test_cases = [
            {
                "title": "Getting Started with Docker Containers",
                "content": """# Getting Started with Docker Containers
                
## Introduction
Docker has revolutionized how we deploy applications.

## What is Docker?
Docker is a containerization platform that packages applications
with their dependencies into lightweight, portable containers.

## Key Concepts
- Images: Read-only templates
- Containers: Running instances of images
- Dockerfile: Build instructions
""",
            },
            {
                "title": "Building REST APIs with FastAPI",
                "content": """# Building REST APIs with FastAPI

## Introduction
FastAPI is a modern Python web framework for building APIs.

## Features
- Automatic API documentation
- Type hints and validation
- Async support
- High performance

## Example
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```
""",
            },
        ]

        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['title']} ---")

            result = blog_image_generator.generate_image_description(
                title=test_case["title"],
                content=test_case["content"],
            )

            assert isinstance(result, ImageDescription)
            assert len(result.description) > 20

            print(f"Style: {result.style}")
            print(f"Description: {result.description[:80]}...")


class TestErrorHandling:
    """Integration tests for error handling with real API."""

    def test_handles_empty_content_gracefully(self, blog_image_generator):
        """Test that empty content doesn't crash."""
        result = blog_image_generator.generate_image_description(
            title="Test Blog Post",
            content="",
        )

        # Should still return a valid description
        assert isinstance(result, ImageDescription)
        assert len(result.description) > 0

    def test_handles_very_short_content(self, blog_image_generator):
        """Test handling of minimal content."""
        result = blog_image_generator.generate_image_description(
            title="Quick Tips",
            content="Some quick tips for developers.",
        )

        assert isinstance(result, ImageDescription)
        assert len(result.description) > 0
        print(f"\nShort content description: {result.description}")
