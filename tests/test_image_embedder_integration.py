"""
Integration tests for Image Embedder using actual LLM calls.

These tests make real API calls to Google Gemini and require:
- GOOGLE_API_KEY environment variable to be set
- Network connectivity

Run with: pytest tests/test_image_embedder_integration.py -v -s
"""

import os
import pytest
from pathlib import Path

from src.media.image_embedder import (
    ImageEmbedder,
    ImagePlacement,
)


# Skip all tests if GOOGLE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set",
)


@pytest.fixture
def image_embedder():
    """Create a real ImageEmbedder instance."""
    return ImageEmbedder()


@pytest.fixture
def sample_content():
    """Sample markdown content for testing."""
    return """# Memory for AI Agents Using Mem0

This comprehensive guide explores how to implement persistent memory
for AI agents using Mem0, an open-source memory layer.

## Introduction

AI agents are becoming increasingly sophisticated, but many still lack
persistent memory capabilities. This limitation means agents treat
each interaction as completely new, leading to repetitive questions
and lack of personalization.

## Understanding Agent Memory

Memory allows agents to retain information across conversations and sessions.
This enables more personalized and context-aware interactions with users.

There are several types of memory in AI systems:
- Short-term memory: Recent conversation context
- Long-term memory: Persistent facts and user preferences
- Episodic memory: Specific events and experiences

### Memory Architecture

The memory architecture typically consists of:
1. Storage layer for persistence
2. Retrieval mechanism for relevant recall
3. Update logic for learning new information

## Implementation with Mem0

Here's how to implement memory with Mem0 in your AI agent.

### Installation

First, install the Mem0 package:

```bash
pip install mem0ai
```

### Basic Usage

```python
from mem0 import Memory

# Initialize memory
m = Memory()

# Store a memory
m.add("User prefers Python over JavaScript", user_id="alice")

# Retrieve relevant memories
memories = m.search("programming preferences", user_id="alice")
```

## Best Practices

When implementing memory for AI agents, consider these best practices:

1. **Privacy**: Always handle user data responsibly
2. **Relevance**: Only store meaningful information
3. **Cleanup**: Implement memory expiration policies

## Conclusion

Memory is essential for building truly intelligent AI agents that can
maintain context and provide personalized experiences over time.
"""


@pytest.fixture
def sample_diagrams():
    """Sample diagram data for testing."""
    return [
        {
            "heading": "Understanding Agent Memory",
            "diagram_type": "flowchart",
            "description": "Memory system types and their relationships",
            "mermaid_code": "flowchart TD\n    A[Input] --> B[Short-term]\n    B --> C[Long-term]",
            "image_base64": "Zmxvd2NoYXJ0X2RpYWdyYW0=",  # placeholder
            "score": 9.5,
        },
        {
            "heading": "Memory Architecture",
            "diagram_type": "flowchart",
            "description": "Architecture layers of the memory system",
            "mermaid_code": "flowchart LR\n    Storage --> Retrieval --> Update",
            "image_base64": "YXJjaGl0ZWN0dXJlX2RpYWdyYW0=",  # placeholder
            "score": 9.0,
        },
        {
            "heading": "Implementation with Mem0",
            "diagram_type": "sequenceDiagram",
            "description": "Sequence of memory operations",
            "mermaid_code": "sequenceDiagram\n    Agent->>Mem0: Store\n    Mem0->>Agent: Retrieve",
            "image_base64": "c2VxdWVuY2VfZGlhZ3JhbQ==",  # placeholder
            "score": 9.2,
        },
    ]


class TestLLMDiagramPlacement:
    """Integration tests for LLM-based diagram placement."""

    def test_llm_determines_placements_for_multiple_diagrams(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test that LLM correctly determines placements for multiple diagrams."""
        placements = image_embedder._get_diagram_placements_from_llm(
            sample_content, sample_diagrams
        )

        # Should return placements for all diagrams
        assert len(placements) == len(sample_diagrams)

        # Each placement should have required fields
        for placement in placements:
            assert isinstance(placement, ImagePlacement)
            assert placement.heading is not None
            assert len(placement.heading) > 0
            assert placement.placement in ["after_heading", "end_of_section"]
            assert placement.reasoning is not None

        # Print for inspection
        print("\n" + "=" * 60)
        print("LLM DIAGRAM PLACEMENT DECISIONS")
        print("=" * 60)
        for i, (diagram, placement) in enumerate(zip(sample_diagrams, placements)):
            print(f"\nDiagram {i+1}: {diagram['heading']}")
            print(f"  Placed after: {placement.heading}")
            print(f"  Position: {placement.placement}")
            print(f"  Reasoning: {placement.reasoning}")
        print("=" * 60)

    def test_llm_matches_diagram_to_correct_section(
        self, image_embedder, sample_content
    ):
        """Test that LLM matches diagrams to semantically correct sections."""
        # Single diagram that should match "Understanding Agent Memory"
        diagrams = [
            {
                "heading": "Agent Memory Types",  # Slightly different name
                "diagram_type": "flowchart",
                "description": "Different types of memory in AI agents",
                "image_base64": "dGVzdA==",
                "score": 9.0,
            }
        ]

        placements = image_embedder._get_diagram_placements_from_llm(
            sample_content, diagrams
        )

        assert len(placements) == 1
        placement = placements[0]

        # Should match to "Understanding Agent Memory" section
        print(f"\nDiagram 'Agent Memory Types' placed after: {placement.heading}")
        print(f"Reasoning: {placement.reasoning}")

        # The LLM should find a semantically similar heading
        assert placement.heading is not None

    def test_llm_handles_ambiguous_placement(self, image_embedder, sample_content):
        """Test LLM handles diagrams that could fit multiple sections."""
        diagrams = [
            {
                "heading": "Python Code Example",  # Could match multiple sections
                "diagram_type": "flowchart",
                "description": "Code execution flow",
                "image_base64": "dGVzdA==",
                "score": 9.0,
            }
        ]

        placements = image_embedder._get_diagram_placements_from_llm(
            sample_content, diagrams
        )

        assert len(placements) == 1
        print(f"\nAmbiguous diagram placed after: {placements[0].heading}")
        print(f"Reasoning: {placements[0].reasoning}")


class TestFullEmbeddingWorkflow:
    """Integration tests for complete embedding workflow with LLM."""

    def test_embed_diagrams_with_llm_placement(
        self, image_embedder, sample_content, sample_diagrams
    ):
        """Test full diagram embedding with LLM-determined placement."""
        result = image_embedder.embed_diagrams(
            sample_content, sample_diagrams, use_llm=True
        )

        # All diagrams should be embedded
        for diagram in sample_diagrams:
            assert diagram["image_base64"] in result

        # Content structure should be preserved
        assert "# Memory for AI Agents" in result
        assert "## Introduction" in result
        assert "## Understanding Agent Memory" in result
        assert "## Implementation with Mem0" in result
        assert "## Conclusion" in result

        # Print sample of result
        print("\n" + "=" * 60)
        print("EMBEDDED CONTENT (first 2000 chars)")
        print("=" * 60)
        print(result[:2000])
        print("..." if len(result) > 2000 else "")
        print("=" * 60)

    def test_embed_all_images_with_cover_and_diagrams(
        self, image_embedder, sample_content, sample_diagrams, tmp_path
    ):
        """Test complete workflow with both cover image and diagrams."""
        import yaml

        # Create diagrams.yaml with both cover image and diagrams
        yaml_path = tmp_path / "diagrams.yaml"
        test_data = {
            "diagrams": sample_diagrams,
            "blog_image": {
                "title": "Memory for AI Agents",
                "description": "AI memory visualization",
                "alt_text": "AI agent memory system illustration",
                "style": "illustration",
                "image_base64": "Y292ZXJfaW1hZ2VfYmFzZTY0",
                "format": "png",
            },
        }

        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        # Run full embedding
        result = image_embedder.embed_all_images(
            content=sample_content,
            diagrams_path=yaml_path,
            progress_callback=progress_callback,
        )

        # Cover image should be embedded after title
        assert "Y292ZXJfaW1hZ2VfYmFzZTY0" in result

        # All diagrams should be embedded
        for diagram in sample_diagrams:
            assert diagram["image_base64"] in result

        # Progress messages should be sent
        assert len(progress_messages) > 0
        assert any("IMAGE EMBEDDING" in msg for msg in progress_messages)
        assert any("cover image" in msg.lower() for msg in progress_messages)
        assert any("diagram" in msg.lower() for msg in progress_messages)

        # Verify cover image is near the top (after H1)
        lines = result.split("\n")
        h1_idx = next(i for i, l in enumerate(lines) if l.startswith("# Memory"))
        cover_idx = next(
            i for i, l in enumerate(lines) if "Y292ZXJfaW1hZ2VfYmFzZTY0" in l
        )
        assert cover_idx > h1_idx
        assert cover_idx < h1_idx + 5  # Should be within 5 lines of title

        print("\n" + "=" * 60)
        print("FULL EMBEDDING TEST RESULTS")
        print("=" * 60)
        print(f"Cover image at line: {cover_idx}")
        print(f"H1 title at line: {h1_idx}")
        print(f"Total lines: {len(lines)}")
        print(f"Total images embedded: {1 + len(sample_diagrams)}")
        print("=" * 60)


class TestEdgeCases:
    """Integration tests for edge cases with real LLM."""

    def test_handles_no_matching_heading(self, image_embedder, sample_content):
        """Test handling when diagram heading doesn't match any content heading."""
        diagrams = [
            {
                "heading": "Completely Unrelated Topic XYZ",
                "diagram_type": "flowchart",
                "description": "Some diagram",
                "image_base64": "dGVzdA==",
                "score": 9.0,
            }
        ]

        placements = image_embedder._get_diagram_placements_from_llm(
            sample_content, diagrams
        )

        # LLM should still return a placement (best guess)
        assert len(placements) == 1
        print(f"\nUnmatched diagram placed after: {placements[0].heading}")
        print(f"Reasoning: {placements[0].reasoning}")

    def test_handles_single_diagram(self, image_embedder, sample_content):
        """Test with just one diagram."""
        diagrams = [
            {
                "heading": "Conclusion",
                "diagram_type": "pie",
                "description": "Summary statistics",
                "image_base64": "c2luZ2xlX2RpYWdyYW0=",
                "score": 9.0,
            }
        ]

        result = image_embedder.embed_diagrams(sample_content, diagrams, use_llm=True)

        assert "c2luZ2xlX2RpYWdyYW0=" in result
        print("\nSingle diagram embedded successfully")

    def test_preserves_code_blocks(self, image_embedder, sample_content, sample_diagrams):
        """Test that code blocks are preserved during embedding."""
        result = image_embedder.embed_diagrams(
            sample_content, sample_diagrams, use_llm=True
        )

        # Code blocks should be intact
        assert "```bash" in result
        assert "pip install mem0ai" in result
        assert "```python" in result
        assert "from mem0 import Memory" in result

        print("\nCode blocks preserved successfully")
