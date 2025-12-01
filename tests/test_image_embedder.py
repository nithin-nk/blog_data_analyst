"""
Tests for the Image Embedder module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.media.image_embedder import (
    ImageEmbedder,
    ImagePlacement,
    DiagramPlacements,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.image_embedder_model = "gemini-2.5-flash"
    return settings


@pytest.fixture
def image_embedder(mock_settings):
    """Create an ImageEmbedder instance with mocked settings."""
    with patch("src.media.image_embedder.get_settings") as mock_get_settings:
        mock_get_settings.return_value = mock_settings
        return ImageEmbedder()


@pytest.fixture
def sample_content():
    """Sample markdown content for testing."""
    return """# Memory for AI Agents Using Mem0

This is the introduction paragraph.

## Understanding Agent Memory

Memory allows agents to retain information across sessions.
There are several types of memory systems.

### Short-term Memory

Short-term memory stores recent context.

## Implementation with Mem0

Here's how to implement memory with Mem0.

```python
from mem0 import Memory
m = Memory()
```

## Conclusion

Memory is essential for AI agents.
"""


@pytest.fixture
def sample_blog_image():
    """Sample blog image data."""
    return {
        "title": "Memory for AI Agents",
        "description": "AI memory visualization",
        "alt_text": "AI agent memory system",
        "style": "illustration",
        "image_base64": "dGVzdF9pbWFnZV9kYXRh",  # "test_image_data" in base64
        "format": "png",
    }


@pytest.fixture
def sample_diagrams():
    """Sample diagram data."""
    return [
        {
            "heading": "Understanding Agent Memory",
            "diagram_type": "flowchart",
            "description": "Memory system architecture",
            "mermaid_code": "flowchart TD\n    A --> B",
            "image_base64": "ZGlhZ3JhbV8x",  # "diagram_1" in base64
            "score": 9.5,
        },
        {
            "heading": "Implementation with Mem0",
            "diagram_type": "sequenceDiagram",
            "description": "Memory interaction sequence",
            "mermaid_code": "sequenceDiagram\n    A->>B: Store",
            "image_base64": "ZGlhZ3JhbV8y",  # "diagram_2" in base64
            "score": 9.0,
        },
    ]


class TestCreateBase64ImageMarkdown:
    """Test base64 image markdown generation."""

    def test_creates_valid_markdown(self, image_embedder):
        """Test that valid markdown is created."""
        result = image_embedder._create_base64_image_markdown(
            base64_data="dGVzdA==",
            alt_text="Test image",
            fmt="png",
        )

        assert result.startswith("![Test image]")
        assert "data:image/png;base64,dGVzdA==" in result

    def test_handles_different_formats(self, image_embedder):
        """Test different image formats."""
        result = image_embedder._create_base64_image_markdown(
            base64_data="dGVzdA==",
            alt_text="JPEG image",
            fmt="jpeg",
        )

        assert "data:image/jpeg;base64," in result


class TestFindHeadingLine:
    """Test heading line finding."""

    def test_finds_h2_heading(self, image_embedder, sample_content):
        """Test finding H2 heading."""
        line = image_embedder._find_heading_line(
            sample_content, "Understanding Agent Memory"
        )

        assert line is not None
        assert "Understanding Agent Memory" in sample_content.split("\n")[line]

    def test_finds_h3_heading(self, image_embedder, sample_content):
        """Test finding H3 heading."""
        line = image_embedder._find_heading_line(sample_content, "Short-term Memory")

        assert line is not None

    def test_returns_none_for_missing_heading(self, image_embedder, sample_content):
        """Test None returned for non-existent heading."""
        line = image_embedder._find_heading_line(
            sample_content, "Non-existent Heading"
        )

        assert line is None

    def test_case_insensitive_matching(self, image_embedder, sample_content):
        """Test case insensitive heading matching."""
        line = image_embedder._find_heading_line(
            sample_content, "understanding agent memory"
        )

        assert line is not None


class TestFindTitleLine:
    """Test title line finding."""

    def test_finds_h1_title(self, image_embedder, sample_content):
        """Test finding H1 title."""
        line = image_embedder._find_title_line(sample_content)

        assert line == 0
        assert sample_content.split("\n")[line].startswith("# Memory")

    def test_returns_zero_for_no_title(self, image_embedder):
        """Test returns 0 if no H1 found."""
        content = "No title here\n\n## Just a heading"
        line = image_embedder._find_title_line(content)

        assert line == 0


class TestEmbedCoverImage:
    """Test blog cover image embedding."""

    def test_embeds_image_after_title(
        self, image_embedder, sample_content, sample_blog_image
    ):
        """Test cover image is embedded after title."""
        result = image_embedder.embed_cover_image(sample_content, sample_blog_image)

        lines = result.split("\n")
        # Find the title line
        title_idx = next(i for i, l in enumerate(lines) if l.startswith("# Memory"))
        # Image should be within next few lines
        image_lines = lines[title_idx + 1 : title_idx + 5]
        image_found = any("data:image/png;base64" in l for l in image_lines)

        assert image_found

    def test_includes_alt_text(
        self, image_embedder, sample_content, sample_blog_image
    ):
        """Test that alt text is included."""
        result = image_embedder.embed_cover_image(sample_content, sample_blog_image)

        assert "AI agent memory system" in result

    def test_no_change_for_empty_image(self, image_embedder, sample_content):
        """Test no change when no image provided."""
        result = image_embedder.embed_cover_image(sample_content, None)

        assert result == sample_content

    def test_no_change_for_missing_base64(self, image_embedder, sample_content):
        """Test no change when image has no base64 data."""
        result = image_embedder.embed_cover_image(
            sample_content, {"title": "Test", "image_base64": ""}
        )

        assert result == sample_content


class TestEmbedDiagrams:
    """Test diagram embedding."""

    def test_embeds_diagrams_under_headings(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test diagrams are embedded under their headings."""
        with patch.object(
            image_embedder, "_get_diagram_placements_from_llm"
        ) as mock_llm:
            mock_llm.return_value = [
                ImagePlacement(
                    heading="Understanding Agent Memory",
                    placement="after_heading",
                    reasoning="Test",
                ),
                ImagePlacement(
                    heading="Implementation with Mem0",
                    placement="after_heading",
                    reasoning="Test",
                ),
            ]

            result = image_embedder.embed_diagrams(
                sample_content, sample_diagrams, use_llm=True
            )

        # Check both diagrams are embedded
        assert "ZGlhZ3JhbV8x" in result  # First diagram base64
        assert "ZGlhZ3JhbV8y" in result  # Second diagram base64

    def test_embeds_without_llm(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test diagram embedding without LLM."""
        result = image_embedder.embed_diagrams(
            sample_content, sample_diagrams, use_llm=False
        )

        assert "ZGlhZ3JhbV8x" in result
        assert "ZGlhZ3JhbV8y" in result

    def test_skips_diagrams_without_base64(self, image_embedder, sample_content):
        """Test diagrams without base64 are skipped."""
        diagrams = [
            {
                "heading": "Test",
                "diagram_type": "flowchart",
                "description": "Test",
                "image_base64": "",
            }
        ]

        result = image_embedder.embed_diagrams(
            sample_content, diagrams, use_llm=False
        )

        # Should not contain any base64 image data
        assert "data:image" not in result

    def test_returns_original_for_no_diagrams(self, image_embedder, sample_content):
        """Test original content returned when no diagrams."""
        result = image_embedder.embed_diagrams(sample_content, [], use_llm=False)

        assert result == sample_content


class TestGetDiagramPlacementsFromLLM:
    """Test LLM-based diagram placement."""

    def test_calls_llm_with_correct_prompt(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test LLM is called with proper prompt structure."""
        with patch("src.media.image_embedder.gemini_llm_call") as mock_llm:
            mock_placements = DiagramPlacements(
                placements=[
                    ImagePlacement(
                        heading="Understanding Agent Memory",
                        placement="after_heading",
                        reasoning="Test",
                    )
                ]
            )
            mock_llm.return_value = mock_placements

            result = image_embedder._get_diagram_placements_from_llm(
                sample_content, sample_diagrams[:1]
            )

            mock_llm.assert_called_once()
            assert len(result) == 1
            assert result[0].heading == "Understanding Agent Memory"

    def test_fallback_on_llm_error(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test fallback when LLM fails."""
        with patch("src.media.image_embedder.gemini_llm_call") as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = image_embedder._get_diagram_placements_from_llm(
                sample_content, sample_diagrams
            )

            # Should fall back to using diagram headings
            assert len(result) == len(sample_diagrams)
            assert result[0].heading == sample_diagrams[0]["heading"]


class TestLoadDiagramsYaml:
    """Test YAML loading."""

    def test_loads_existing_file(self, image_embedder, tmp_path):
        """Test loading existing diagrams.yaml."""
        import yaml

        yaml_path = tmp_path / "diagrams.yaml"
        test_data = {
            "diagrams": [{"heading": "Test", "image_base64": "abc"}],
            "blog_image": {"title": "Test", "image_base64": "xyz"},
        }

        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        result = image_embedder.load_diagrams_yaml(yaml_path)

        assert len(result["diagrams"]) == 1
        assert result["blog_image"]["title"] == "Test"

    def test_returns_empty_for_missing_file(self, image_embedder, tmp_path):
        """Test returns empty dict for missing file."""
        yaml_path = tmp_path / "nonexistent.yaml"

        result = image_embedder.load_diagrams_yaml(yaml_path)

        assert result["diagrams"] == []
        assert result["blog_image"] is None


class TestEmbedAllImages:
    """Test complete embedding workflow."""

    def test_embeds_both_cover_and_diagrams(
        self, image_embedder, sample_content, sample_blog_image, sample_diagrams, tmp_path
    ):
        """Test both cover image and diagrams are embedded."""
        import yaml

        yaml_path = tmp_path / "diagrams.yaml"
        test_data = {
            "diagrams": sample_diagrams,
            "blog_image": sample_blog_image,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        with patch.object(
            image_embedder, "_get_diagram_placements_from_llm"
        ) as mock_llm:
            mock_llm.return_value = [
                ImagePlacement(
                    heading=d["heading"], placement="after_heading", reasoning="Test"
                )
                for d in sample_diagrams
            ]

            result = image_embedder.embed_all_images(
                content=sample_content,
                diagrams_path=yaml_path,
            )

        # Check cover image is embedded
        assert sample_blog_image["image_base64"] in result
        # Check diagrams are embedded
        assert sample_diagrams[0]["image_base64"] in result
        assert sample_diagrams[1]["image_base64"] in result

    def test_handles_missing_diagrams_file(
        self, image_embedder, sample_content, tmp_path
    ):
        """Test handles missing diagrams.yaml gracefully."""
        yaml_path = tmp_path / "nonexistent.yaml"

        result = image_embedder.embed_all_images(
            content=sample_content,
            diagrams_path=yaml_path,
        )

        # Should return original content unchanged
        assert result == sample_content

    def test_progress_callback_called(
        self, image_embedder, sample_content, sample_blog_image, tmp_path
    ):
        """Test progress callback is invoked."""
        import yaml

        yaml_path = tmp_path / "diagrams.yaml"
        test_data = {"diagrams": [], "blog_image": sample_blog_image}

        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        progress_messages = []

        def callback(msg):
            progress_messages.append(msg)

        image_embedder.embed_all_images(
            content=sample_content,
            diagrams_path=yaml_path,
            progress_callback=callback,
        )

        assert len(progress_messages) > 0
        assert any("IMAGE EMBEDDING" in msg for msg in progress_messages)


class TestFindSectionEnd:
    """Test section end finding."""

    def test_finds_next_heading(self, image_embedder, sample_content):
        """Test finds line before next heading."""
        heading_line = image_embedder._find_heading_line(
            sample_content, "Understanding Agent Memory"
        )
        section_end = image_embedder._find_section_end(sample_content, heading_line)

        lines = sample_content.split("\n")
        # Section should end before "Implementation with Mem0"
        assert section_end < len(lines)
        # The line after section_end should be a heading or we're at the end
        if section_end + 1 < len(lines):
            next_line = lines[section_end + 1]
            assert next_line.startswith("## ") or next_line.startswith("# ")

    def test_returns_last_line_for_final_section(self, image_embedder, sample_content):
        """Test returns last line for final section."""
        heading_line = image_embedder._find_heading_line(sample_content, "Conclusion")
        section_end = image_embedder._find_section_end(sample_content, heading_line)

        lines = sample_content.split("\n")
        assert section_end == len(lines) - 1
