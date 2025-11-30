"""
YAML parser for blog input specifications.

Parses and validates YAML input files containing:
- Topic
- Outline (list of questions/subtopics)
- Special markers (Code:, Mermaid:)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class OutlineItem(BaseModel):
    """Represents a single item in the blog outline."""
    
    question: str = Field(..., description="The question or subtopic to cover")
    requires_code: bool = Field(default=False, description="Whether code examples are needed")
    requires_mermaid: bool = Field(default=False, description="Whether Mermaid diagrams are needed")
    
    @field_validator("question", mode="before")
    @classmethod
    def parse_question(cls, v: str) -> str:
        """Extract question and detect special markers."""
        return v.strip()


class BlogInput(BaseModel):
    """Validated blog input specification."""
    
    topic: str = Field(..., description="Main topic of the blog post")
    outline: List[str] = Field(..., description="List of questions/subtopics to cover")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @field_validator("outline")
    @classmethod
    def validate_outline(cls, v: List[str]) -> List[str]:
        """Ensure outline has at least one item."""
        if not v:
            raise ValueError("Outline must contain at least one question/subtopic")
        return v


class YAMLParser:
    """Parser for YAML blog input files."""
    
    @staticmethod
    def parse_file(file_path: Path) -> BlogInput:
        """
        Parse a YAML file and return validated BlogInput.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            BlogInput: Validated blog input specification
            
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary at root level")
        
        return BlogInput(**data)
    
    @staticmethod
    def detect_special_markers(question: str) -> Dict[str, bool]:
        """
        Detect special markers in outline questions.
        
        Args:
            question: The outline question text
            
        Returns:
            Dict with 'code' and 'mermaid' boolean flags
        """
        question_lower = question.lower()
        return {
            "code": "code:" in question_lower,
            "mermaid": "mermaid:" in question_lower,
        }
