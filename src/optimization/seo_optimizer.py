"""
SEO optimizer for blog content analysis.

Analyzes and suggests improvements for:
- Keyword density
- Readability scores
- Header hierarchy
- Link structure
- Meta tags
"""

from typing import Dict, Any, List
import re
from collections import Counter

import textstat
from bs4 import BeautifulSoup

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class SEOOptimizer:
    """Optimizer for SEO analysis and suggestions."""
    
    def __init__(self) -> None:
        """Initialize the SEO optimizer."""
        logger.info("SEOOptimizer initialized")
    
    def analyze(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive SEO analysis.
        
        Args:
            content: Blog content in markdown
            target_keywords: List of target keywords
            
        Returns:
            Dict containing analysis results and suggestions
        """
        logger.info("Starting SEO analysis")
        
        analysis = {
            "keyword_density": self.analyze_keyword_density(content, target_keywords),
            "readability": self.analyze_readability(content),
            "header_structure": self.analyze_headers(content),
            "word_count": len(content.split()),
            "suggestions": [],
        }
        
        # Generate suggestions based on analysis
        analysis["suggestions"] = self._generate_suggestions(analysis)
        
        logger.info("SEO analysis complete")
        return analysis
    
    def analyze_keyword_density(
        self,
        content: str,
        keywords: List[str],
    ) -> Dict[str, float]:
        """
        Calculate keyword density for target keywords.
        
        Args:
            content: Blog content
            keywords: Target keywords
            
        Returns:
            Dict mapping keywords to their density percentage
        """
        content_lower = content.lower()
        words = re.findall(r'\w+', content_lower)
        total_words = len(words)
        
        if total_words == 0:
            return {kw: 0.0 for kw in keywords}
        
        densities = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / total_words) * 100
            densities[keyword] = round(density, 2)
        
        logger.debug(f"Keyword densities: {densities}")
        return densities
    
    def analyze_readability(self, content: str) -> Dict[str, Any]:
        """
        Analyze content readability using multiple metrics.
        
        Args:
            content: Blog content
            
        Returns:
            Dict with readability scores
        """
        # Clean markdown for readability analysis
        clean_text = self._clean_markdown(content)
        
        scores = {
            "flesch_reading_ease": textstat.flesch_reading_ease(clean_text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(clean_text),
            "gunning_fog": textstat.gunning_fog(clean_text),
            "smog_index": textstat.smog_index(clean_text),
            "automated_readability_index": textstat.automated_readability_index(clean_text),
        }
        
        # Interpret scores
        scores["reading_level"] = self._interpret_readability(scores["flesch_reading_ease"])
        
        logger.debug(f"Readability scores: {scores}")
        return scores
    
    def analyze_headers(self, content: str) -> Dict[str, Any]:
        """
        Analyze header hierarchy and structure.
        
        Args:
            content: Blog content in markdown
            
        Returns:
            Dict with header analysis
        """
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        header_counts = Counter(len(h[0]) for h in headers)
        
        analysis = {
            "total_headers": len(headers),
            "h1_count": header_counts.get(1, 0),
            "h2_count": header_counts.get(2, 0),
            "h3_count": header_counts.get(3, 0),
            "hierarchy_valid": self._validate_hierarchy(headers),
        }
        
        logger.debug(f"Header analysis: {analysis}")
        return analysis
    
    @staticmethod
    def _clean_markdown(content: str) -> str:
        """Remove markdown syntax for readability analysis."""
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove inline code
        content = re.sub(r'`[^`]*`', '', content)
        # Remove links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Remove emphasis
        content = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', content)
        
        return content
    
    @staticmethod
    def _interpret_readability(flesch_score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if flesch_score >= 90:
            return "Very Easy (5th grade)"
        elif flesch_score >= 80:
            return "Easy (6th grade)"
        elif flesch_score >= 70:
            return "Fairly Easy (7th grade)"
        elif flesch_score >= 60:
            return "Standard (8th-9th grade)"
        elif flesch_score >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif flesch_score >= 30:
            return "Difficult (College)"
        else:
            return "Very Difficult (College graduate)"
    
    @staticmethod
    def _validate_hierarchy(headers: List[tuple]) -> bool:
        """Validate header hierarchy (no skipped levels)."""
        if not headers:
            return True
        
        levels = [len(h[0]) for h in headers]
        
        for i in range(len(levels) - 1):
            if levels[i + 1] > levels[i] + 1:
                return False
        
        return True
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate SEO improvement suggestions."""
        suggestions = []
        
        # Word count suggestions
        word_count = analysis["word_count"]
        if word_count < 1000:
            suggestions.append(f"Content is {word_count} words. Aim for 1000+ words for better SEO.")
        
        # Header suggestions
        header_analysis = analysis["header_structure"]
        if header_analysis["h1_count"] == 0:
            suggestions.append("Add an H1 header for the main title.")
        elif header_analysis["h1_count"] > 1:
            suggestions.append("Use only one H1 header per page.")
        
        if not header_analysis["hierarchy_valid"]:
            suggestions.append("Fix header hierarchy - don't skip levels (e.g., H1 to H3).")
        
        # Readability suggestions
        readability = analysis["readability"]
        if readability["flesch_reading_ease"] < 50:
            suggestions.append("Content is difficult to read. Consider simplifying sentences.")
        
        return suggestions
