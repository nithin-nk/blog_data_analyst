"""Tests for YAML parser."""

import pytest
from pathlib import Path
from src.parsers.yaml_parser import YAMLParser, BlogInput


def test_parse_valid_yaml(tmp_path):
    """Test parsing a valid YAML file."""
    yaml_content = """
topic: "Introduction to Machine Learning"
outline:
  - "What is Machine Learning?"
  - "Types of Machine Learning"
  - "Code: Simple ML example"
  - "Mermaid: ML workflow diagram"
"""
    
    yaml_file = tmp_path / "test_input.yaml"
    yaml_file.write_text(yaml_content)
    
    result = YAMLParser.parse_file(yaml_file)
    
    assert isinstance(result, BlogInput)
    assert result.topic == "Introduction to Machine Learning"
    assert len(result.outline) == 4


def test_parse_missing_file():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        YAMLParser.parse_file(Path("nonexistent.yaml"))


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
