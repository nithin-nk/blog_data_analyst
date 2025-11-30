"""
YAML parser for blog input specifications.

Parses and validates YAML input files containing:
- Topic
- Outline (list of questions/subtopics)
- Special markers (Code:, Mermaid:)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class OutlineItem(BaseModel):
    """Represents a single item in the blog outline with markers."""
    
    text: str = Field(..., description="The question or subtopic text")
    requires_code: bool = Field(default=False, description="Whether code examples are needed")
    requires_mermaid: bool = Field(default=False, description="Whether Mermaid diagrams are needed")
    clean_text: str = Field(default="", description="Text without markers")
    
    @classmethod
    def from_string(cls, question: str) -> "OutlineItem":
        """
        Create OutlineItem from a question string, detecting markers.
        
        Args:
            question: Raw question string from YAML
            
        Returns:
            OutlineItem with detected markers
        """
        question = question.strip()
        markers = YAMLParser.detect_special_markers(question)
        clean = YAMLParser.clean_markers(question)
        
        return cls(
            text=question,
            requires_code=markers["code"],
            requires_mermaid=markers["mermaid"],
            clean_text=clean,
        )


class BlogInput(BaseModel):
    """Validated blog input specification."""
    
    topic: str = Field(..., min_length=3, description="Main topic of the blog post")
    outline: List[str] = Field(..., min_length=1, description="List of questions/subtopics to cover")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    # Processed outline items
    outline_items: List[OutlineItem] = Field(default_factory=list, description="Processed outline items")
    
    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Validate and clean topic."""
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Topic must be at least 3 characters long")
        return v
    
    @field_validator("outline")
    @classmethod
    def validate_outline(cls, v: List[str]) -> List[str]:
        """Ensure outline has at least one item and clean whitespace."""
        if not v:
            raise ValueError("Outline must contain at least one question/subtopic")
        
        # Clean whitespace from all items
        cleaned = [item.strip() for item in v if item.strip()]
        
        if not cleaned:
            raise ValueError("Outline cannot contain only empty strings")
        
        return cleaned
    
    def model_post_init(self, __context: Any) -> None:
        """Process outline items after model initialization."""
        self.outline_items = [OutlineItem.from_string(q) for q in self.outline]


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
        logger.info(f"Parsing YAML file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax: {e}")
            raise ValueError(f"Invalid YAML syntax: {e}")
        
        if not isinstance(data, dict):
            logger.error("YAML root must be a dictionary")
            raise ValueError("YAML file must contain a dictionary at root level")
        
        try:
            blog_input = BlogInput(**data)
            logger.info(f"Successfully parsed blog input: {blog_input.topic}")
            logger.info(f"Outline contains {len(blog_input.outline)} items")
            
            # Log marker detection
            code_count = sum(1 for item in blog_input.outline_items if item.requires_code)
            mermaid_count = sum(1 for item in blog_input.outline_items if item.requires_mermaid)
            logger.info(f"Detected {code_count} code sections, {mermaid_count} Mermaid diagrams")
            
            return blog_input
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise ValueError(f"Invalid blog input structure: {e}")
    
    @staticmethod
    def detect_special_markers(question: str) -> Dict[str, bool]:
        """
        Detect special markers in outline questions.
        
        Markers:
        - Code: or code: - Indicates code example needed
        - Mermaid: or mermaid: - Indicates Mermaid diagram needed
        
        Args:
            question: The outline question text
            
        Returns:
            Dict with 'code' and 'mermaid' boolean flags
        """
        question_lower = question.lower()
        
        # Check for markers at the start or after whitespace
        code_pattern = r'(?:^|\s)code:'
        mermaid_pattern = r'(?:^|\s)mermaid:'
        
        return {
            "code": bool(re.search(code_pattern, question_lower)),
            "mermaid": bool(re.search(mermaid_pattern, question_lower)),
        }
    
    @staticmethod
    def clean_markers(question: str) -> str:
        """
        Remove special markers from question text.
        
        Args:
            question: Question with possible markers
            
        Returns:
            Cleaned question text
        """
        # Remove "Code:" and "Mermaid:" markers (case insensitive)
        cleaned = re.sub(r'(?i)\bcode:\s*', '', question)
        cleaned = re.sub(r'(?i)\bmermaid:\s*', '', cleaned)
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    @staticmethod
    def validate_yaml_structure(data: Dict[str, Any]) -> List[str]:
        """
        Validate YAML structure and return list of issues.
        
        Args:
            data: Parsed YAML data
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check required fields
        if "topic" not in data:
            issues.append("Missing required field: 'topic'")
        elif not isinstance(data["topic"], str):
            issues.append("Field 'topic' must be a string")
        elif len(data["topic"].strip()) < 3:
            issues.append("Field 'topic' must be at least 3 characters")
        
        if "outline" not in data:
            issues.append("Missing required field: 'outline'")
        elif not isinstance(data["outline"], list):
            issues.append("Field 'outline' must be a list")
        elif len(data["outline"]) == 0:
            issues.append("Field 'outline' cannot be empty")
        else:
            # Check outline items
            for i, item in enumerate(data["outline"]):
                if not isinstance(item, str):
                    issues.append(f"Outline item {i+1} must be a string")
                elif not item.strip():
                    issues.append(f"Outline item {i+1} cannot be empty")
        
        # Check metadata if present
        if "metadata" in data and not isinstance(data["metadata"], dict):
            issues.append("Field 'metadata' must be a dictionary")
        
        return issues
