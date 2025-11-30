"""Tests for optimization modules."""

import pytest
from src.optimization.seo_optimizer import SEOOptimizer
from src.optimization.quality_checker import QualityChecker


def test_seo_optimizer_initialization():
    """Test SEOOptimizer initializes correctly."""
    optimizer = SEOOptimizer()
    assert optimizer is not None


def test_keyword_density():
    """Test keyword density calculation."""
    optimizer = SEOOptimizer()
    content = "Python is great. Python is powerful. Learn Python today."
    keywords = ["Python", "JavaScript"]
    
    densities = optimizer.analyze_keyword_density(content, keywords)
    
    assert "Python" in densities
    assert densities["Python"] > 0
    assert densities["JavaScript"] == 0


def test_header_analysis():
    """Test header structure analysis."""
    optimizer = SEOOptimizer()
    content = """# Main Title
## Section 1
### Subsection 1.1
## Section 2
"""
    
    analysis = optimizer.analyze_headers(content)
    
    assert analysis["h1_count"] == 1
    assert analysis["h2_count"] == 2
    assert analysis["h3_count"] == 1
    assert analysis["hierarchy_valid"] is True


def test_clean_markdown():
    """Test markdown cleaning."""
    content = "# Title\n\nSome **bold** text with `code` and [link](url)."
    clean = SEOOptimizer._clean_markdown(content)
    
    assert "#" not in clean
    assert "**" not in clean
    assert "`" not in clean
    assert "bold" in clean


@pytest.mark.asyncio
async def test_quality_checker_initialization():
    """Test QualityChecker initializes correctly."""
    pytest.skip("Requires API keys")
