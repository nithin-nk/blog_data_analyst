"""Tests for YAML parser."""

import pytest
from pathlib import Path
from src.parsers.yaml_parser import YAMLParser, BlogInput, OutlineItem


def test_parse_valid_yaml(tmp_path):
    """Test parsing a valid YAML file."""
    yaml_content = """topic: "Introduction to Machine Learning"
outline:
  - "What is Machine Learning?"
  - "Types of Machine Learning"
  - "Code: Simple ML example"
  - "Mermaid: ML workflow diagram"
metadata:
  target_audience: "Beginners"
  difficulty: "Intermediate"
"""
    
    yaml_file = tmp_path / "test_input.yaml"
    yaml_file.write_text(yaml_content)
    
    result = YAMLParser.parse_file(yaml_file)
    
    assert isinstance(result, BlogInput)
    assert result.topic == "Introduction to Machine Learning"
    assert len(result.outline) == 4
    assert len(result.outline_items) == 4
    assert result.metadata["target_audience"] == "Beginners"


def test_parse_minimal_yaml(tmp_path):
    """Test parsing minimal valid YAML."""
    yaml_content = """topic: "Test Topic"
outline:
  - "Question 1"
"""
    
    yaml_file = tmp_path / "minimal.yaml"
    yaml_file.write_text(yaml_content)
    
    result = YAMLParser.parse_file(yaml_file)
    
    assert result.topic == "Test Topic"
    assert len(result.outline) == 1
    assert result.metadata == {}


def test_parse_missing_file():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        YAMLParser.parse_file(Path("nonexistent.yaml"))


def test_parse_invalid_yaml(tmp_path):
    """Test error on malformed YAML."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("topic: test\noutline:\n  - item\n\t- bad indent")
    
    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        YAMLParser.parse_file(yaml_file)


def test_parse_missing_topic(tmp_path):
    """Test error when topic is missing."""
    yaml_content = """outline:
  - "Question 1"
"""
    
    yaml_file = tmp_path / "no_topic.yaml"
    yaml_file.write_text(yaml_content)
    
    with pytest.raises(ValueError, match="Invalid blog input structure"):
        YAMLParser.parse_file(yaml_file)


def test_parse_empty_outline(tmp_path):
    """Test error when outline is empty."""
    yaml_content = """topic: "Test"
outline: []
"""
    
    yaml_file = tmp_path / "empty_outline.yaml"
    yaml_file.write_text(yaml_content)
    
    with pytest.raises(ValueError, match="Invalid blog input structure"):
        YAMLParser.parse_file(yaml_file)


def test_detect_code_marker():
    """Test detection of Code: marker."""
    question = "Code: How to implement binary search"
    markers = YAMLParser.detect_special_markers(question)
    
    assert markers["code"] is True
    assert markers["mermaid"] is False


def test_detect_mermaid_marker():
    """Test detection of Mermaid: marker."""
    question = "Mermaid: System architecture diagram"
    markers = YAMLParser.detect_special_markers(question)
    
    assert markers["code"] is False
    assert markers["mermaid"] is True


def test_detect_both_markers():
    """Test detection of both markers."""
    question = "Code: Example with Mermaid: diagram"
    markers = YAMLParser.detect_special_markers(question)
    
    assert markers["code"] is True
    assert markers["mermaid"] is True


def test_detect_no_markers():
    """Test no markers detected."""
    question = "What is the best approach?"
    markers = YAMLParser.detect_special_markers(question)
    
    assert markers["code"] is False
    assert markers["mermaid"] is False


def test_detect_case_insensitive():
    """Test marker detection is case insensitive."""
    assert YAMLParser.detect_special_markers("CODE: example")["code"] is True
    assert YAMLParser.detect_special_markers("code: example")["code"] is True
    assert YAMLParser.detect_special_markers("MERMAID: diagram")["mermaid"] is True


def test_clean_markers():
    """Test marker removal."""
    assert YAMLParser.clean_markers("Code: Example code") == "Example code"
    assert YAMLParser.clean_markers("Mermaid: Draw diagram") == "Draw diagram"
    assert YAMLParser.clean_markers("Code: and Mermaid: both") == "and both"
    assert YAMLParser.clean_markers("No markers here") == "No markers here"


def test_outline_item_from_string():
    """Test OutlineItem creation from string."""
    item = OutlineItem.from_string("Code: Implement function")
    
    assert item.text == "Code: Implement function"
    assert item.requires_code is True
    assert item.requires_mermaid is False
    assert item.clean_text == "Implement function"


def test_blog_input_outline_items():
    """Test automatic outline items processing."""
    data = {
        "topic": "Test",
        "outline": [
            "Introduction",
            "Code: Example",
            "Mermaid: Diagram"
        ]
    }
    
    blog_input = BlogInput(**data)
    
    assert len(blog_input.outline_items) == 3
    assert blog_input.outline_items[0].requires_code is False
    assert blog_input.outline_items[1].requires_code is True
    assert blog_input.outline_items[2].requires_mermaid is True


def test_validate_yaml_structure():
    """Test YAML structure validation."""
    # Valid structure
    valid_data = {
        "topic": "Test Topic",
        "outline": ["Question 1", "Question 2"]
    }
    issues = YAMLParser.validate_yaml_structure(valid_data)
    assert len(issues) == 0
    
    # Missing topic
    invalid_data = {"outline": ["Q1"]}
    issues = YAMLParser.validate_yaml_structure(invalid_data)
    assert any("topic" in issue.lower() for issue in issues)
    
    # Missing outline
    invalid_data = {"topic": "Test"}
    issues = YAMLParser.validate_yaml_structure(invalid_data)
    assert any("outline" in issue.lower() for issue in issues)
    
    # Empty outline
    invalid_data = {"topic": "Test", "outline": []}
    issues = YAMLParser.validate_yaml_structure(invalid_data)
    assert any("empty" in issue.lower() for issue in issues)
