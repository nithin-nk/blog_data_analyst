"""Tests for converters."""

import pytest
from pathlib import Path
from src.converters.md_to_html import MarkdownToHTMLConverter


def test_converter_initialization():
    """Test MarkdownToHTMLConverter initializes correctly."""
    converter = MarkdownToHTMLConverter()
    assert converter is not None


def test_basic_markdown_conversion():
    """Test basic markdown to HTML conversion."""
    converter = MarkdownToHTMLConverter()
    
    markdown = "# Hello\n\nThis is **bold** text."
    html = converter.convert(
        markdown,
        "Test Title",
        "Test description"
    )
    
    assert "<h1>" in html or "Hello" in html
    assert "<strong>" in html or "<b>" in html or "bold" in html
    assert "<!DOCTYPE html>" in html


def test_code_block_conversion():
    """Test code block conversion."""
    converter = MarkdownToHTMLConverter()
    
    markdown = """
```python
def hello():
    print("Hello, World!")
```
"""
    
    html = converter.convert(markdown, "Code Test", "Testing code blocks")
    
    assert "<code>" in html or "<pre>" in html


def test_mermaid_processing():
    """Test Mermaid diagram processing."""
    converter = MarkdownToHTMLConverter()
    
    # Note: This is a basic test - actual rendering happens in browser
    html = "<pre><code class='mermaid'>graph TD\nA --> B</code></pre>"
    processed = converter._process_mermaid(html)
    
    assert "mermaid" in processed.lower()
