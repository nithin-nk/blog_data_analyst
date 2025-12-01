"""
Tests for the Mermaid Diagram Generator module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.generation.diagram_generator import (
    DiagramGenerator,
    DiagramOpportunity,
    DiagramReview,
    GeneratedDiagram,
    DiagramOpportunities
)


@pytest.fixture
def diagram_generator():
    """Create a DiagramGenerator instance with mocked settings."""
    with patch('src.generation.diagram_generator.get_settings') as mock_settings:
        mock_settings.return_value.diagram_identifier_model = "gemini-2.5-flash"
        mock_settings.return_value.diagram_generator_model = "gemini-2.5-flash"
        mock_settings.return_value.diagram_reviewer_model = "gemini-2.5-flash"
        mock_settings.return_value.max_diagram_generation_attempts = 3
        mock_settings.return_value.diagram_quality_threshold = 9.0
        return DiagramGenerator()


@pytest.fixture
def sample_content():
    """Sample blog content for testing."""
    return """# Building Production-Ready AI Agents

## Introduction
Welcome to this tutorial on building AI agents.

## System Architecture
We'll explore the overall system architecture using microservices.
This involves API gateways, message queues, and databases.

## Implementation Details
Here's how to implement the core functionality.

## Deployment Workflow
The deployment process follows a CI/CD pipeline.
"""


@pytest.fixture
def sample_opportunity():
    """Sample diagram opportunity."""
    return DiagramOpportunity(
        heading="System Architecture",
        diagram_type="flowchart",
        description="System architecture showing microservices components",
        reasoning="Helps visualize the complex system structure"
    )


class TestExtractHeadings:
    """Test heading extraction from content."""
    
    def test_extract_h2_headings(self, diagram_generator, sample_content):
        """Test extraction of H2 headings."""
        headings = diagram_generator._extract_headings(sample_content)
        
        assert "Introduction" in headings
        assert "System Architecture" in headings
        assert "Implementation Details" in headings
        assert "Deployment Workflow" in headings
    
    def test_extract_empty_content(self, diagram_generator):
        """Test extraction from empty content."""
        headings = diagram_generator._extract_headings("")
        assert headings == []


class TestIdentifyDiagramOpportunities:
    """Test diagram opportunity identification."""
    
    def test_identify_opportunities_success(self, diagram_generator, sample_content):
        """Test successful identification of diagram opportunities."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_opportunities = DiagramOpportunities(
                opportunities=[
                    DiagramOpportunity(
                        heading="System Architecture",
                        diagram_type="flowchart",
                        description="System components and connections",
                        reasoning="Complex architecture needs visualization"
                    ),
                    DiagramOpportunity(
                        heading="Deployment Workflow",
                        diagram_type="sequenceDiagram",
                        description="CI/CD deployment sequence",
                        reasoning="Shows temporal flow of deployment"
                    )
                ]
            )
            mock_llm.return_value = mock_opportunities
            
            opportunities = diagram_generator.identify_diagram_opportunities(
                "Building AI Agents",
                sample_content
            )
            
            assert len(opportunities) == 2
            assert opportunities[0].heading == "System Architecture"
            assert opportunities[0].diagram_type == "flowchart"
            assert opportunities[1].diagram_type == "sequenceDiagram"
    
    def test_identify_opportunities_no_results(self, diagram_generator, sample_content):
        """Test when no opportunities are identified."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = DiagramOpportunities(opportunities=[])
            
            opportunities = diagram_generator.identify_diagram_opportunities(
                "Simple Topic",
                sample_content
            )
            
            assert len(opportunities) == 0
    
    def test_identify_opportunities_error_handling(self, diagram_generator, sample_content):
        """Test error handling during identification."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            opportunities = diagram_generator.identify_diagram_opportunities(
                "Topic",
                sample_content
            )
            
            assert opportunities == []


class TestGenerateMermaidDiagram:
    """Test mermaid diagram generation."""
    
    def test_generate_initial_diagram(self, diagram_generator, sample_opportunity, sample_content):
        """Test initial diagram generation."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = """flowchart TD
    A[API Gateway] --> B[Service 1]
    A --> C[Service 2]
    B --> D[Database]
    C --> D"""
            
            mermaid_code = diagram_generator.generate_mermaid_diagram(
                sample_opportunity,
                sample_content
            )
            
            assert "flowchart TD" in mermaid_code
            assert "API Gateway" in mermaid_code
    
    def test_generate_with_feedback(self, diagram_generator, sample_opportunity, sample_content):
        """Test diagram regeneration with feedback."""
        previous_code = "flowchart TD\n    A --> B"
        feedback = "Add more descriptive labels and include database connection"
        
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = """flowchart TD
    A[API Gateway] --> B[Auth Service]
    B --> C[(Database)]"""
            
            mermaid_code = diagram_generator.generate_mermaid_diagram(
                sample_opportunity,
                sample_content,
                feedback=feedback,
                previous_code=previous_code
            )
            
            assert "flowchart TD" in mermaid_code
            mock_llm.assert_called_once()
    
    def test_clean_markdown_fences(self, diagram_generator, sample_opportunity, sample_content):
        """Test cleaning of markdown code fences from generated code."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = "```mermaid\nflowchart TD\n    A --> B\n```"
            
            mermaid_code = diagram_generator.generate_mermaid_diagram(
                sample_opportunity,
                sample_content
            )
            
            assert not mermaid_code.startswith("```")
            assert not mermaid_code.endswith("```")
            assert mermaid_code.startswith("flowchart TD")


class TestReviewDiagram:
    """Test diagram quality review."""
    
    def test_review_high_quality_diagram(self, diagram_generator, sample_opportunity, sample_content):
        """Test review of high-quality diagram."""
        mermaid_code = """flowchart TD
    A[API Gateway] --> B[Service 1]
    A --> C[Service 2]"""
        
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_review = DiagramReview(
                score=9.5,
                strengths=["Clear structure", "Good labels"],
                weaknesses=["Minor styling improvements possible"],
                feedback="Excellent diagram, minor styling could be better"
            )
            mock_llm.return_value = mock_review
            
            review = diagram_generator.review_diagram(
                mermaid_code,
                sample_opportunity,
                sample_content
            )
            
            assert review['score'] == 9.5
            assert len(review['strengths']) == 2
            assert 'feedback' in review
    
    def test_review_poor_quality_diagram(self, diagram_generator, sample_opportunity, sample_content):
        """Test review of poor-quality diagram."""
        mermaid_code = "graph A-->B"
        
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_review = DiagramReview(
                score=5.0,
                strengths=["Basic structure present"],
                weaknesses=["No labels", "Incomplete"],
                feedback="Needs descriptive labels and more detail"
            )
            mock_llm.return_value = mock_review
            
            review = diagram_generator.review_diagram(
                mermaid_code,
                sample_opportunity,
                sample_content
            )
            
            assert review['score'] == 5.0
            assert len(review['weaknesses']) > 0
    
    def test_review_error_fallback(self, diagram_generator, sample_opportunity, sample_content):
        """Test fallback when review fails."""
        with patch('src.generation.diagram_generator.gemini_llm_call') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            review = diagram_generator.review_diagram(
                "flowchart TD",
                sample_opportunity,
                sample_content
            )
            
            assert review['score'] == 5.0
            assert 'feedback' in review


class TestGenerateWithFeedbackLoop:
    """Test diagram generation with feedback loop."""
    
    def test_successful_first_attempt(self, diagram_generator, sample_opportunity, sample_content):
        """Test when first attempt meets quality threshold."""
        with patch.object(diagram_generator, 'generate_mermaid_diagram') as mock_gen, \
             patch.object(diagram_generator, 'review_diagram') as mock_review:
            
            mock_gen.return_value = "flowchart TD\n    A --> B"
            mock_review.return_value = {
                'score': 9.5,
                'strengths': [],
                'weaknesses': [],
                'feedback': 'Great!'
            }
            
            result = diagram_generator.generate_with_feedback_loop(
                sample_opportunity,
                sample_content
            )
            
            assert result['score'] == 9.5
            assert mock_gen.call_count == 1
            assert mock_review.call_count == 1
    
    def test_improvement_over_iterations(self, diagram_generator, sample_opportunity, sample_content):
        """Test improvement over multiple iterations."""
        with patch.object(diagram_generator, 'generate_mermaid_diagram') as mock_gen, \
             patch.object(diagram_generator, 'review_diagram') as mock_review:
            
            mock_gen.return_value = "flowchart TD\n    A --> B"
            
            # First two attempts fail, third succeeds
            mock_review.side_effect = [
                {'score': 6.0, 'strengths': [], 'weaknesses': [], 'feedback': 'Needs improvement'},
                {'score': 7.5, 'strengths': [], 'weaknesses': [], 'feedback': 'Better but still lacking'},
                {'score': 9.2, 'strengths': [], 'weaknesses': [], 'feedback': 'Excellent!'}
            ]
            
            result = diagram_generator.generate_with_feedback_loop(
                sample_opportunity,
                sample_content
            )
            
            assert result['score'] == 9.2
            assert mock_gen.call_count == 3
            assert mock_review.call_count == 3
    
    def test_max_attempts_returns_best(self, diagram_generator, sample_opportunity, sample_content):
        """Test that best version is returned after max attempts."""
        with patch.object(diagram_generator, 'generate_mermaid_diagram') as mock_gen, \
             patch.object(diagram_generator, 'review_diagram') as mock_review:
            
            mock_gen.return_value = "flowchart TD\n    A --> B"
            
            # None meet threshold, but second is best
            mock_review.side_effect = [
                {'score': 6.0, 'strengths': [], 'weaknesses': [], 'feedback': 'Poor'},
                {'score': 8.0, 'strengths': [], 'weaknesses': [], 'feedback': 'Better'},
                {'score': 7.0, 'strengths': [], 'weaknesses': [], 'feedback': 'Worse again'}
            ]
            
            result = diagram_generator.generate_with_feedback_loop(
                sample_opportunity,
                sample_content
            )
            
            # Should return the best score (8.0) not the last (7.0)
            assert result['score'] == 8.0
            assert mock_gen.call_count == 3


class TestConvertMermaidToImage:
    """Test mermaid to image conversion."""
    
    def test_convert_success(self, diagram_generator):
        """Test successful conversion to base64 image using Kroki.io API."""
        mermaid_code = "flowchart TD\n    A --> B"
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.content = b'PNG_IMAGE_BYTES_FROM_API'
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            
            base64_image = diagram_generator.convert_mermaid_to_image(mermaid_code)
            
            assert base64_image != ""
            assert isinstance(base64_image, str)
            # Verify it's actually base64 encoded
            import base64 as b64
            try:
                decoded = b64.b64decode(base64_image)
                assert decoded == b'PNG_IMAGE_BYTES_FROM_API'
            except:
                assert False, "Not valid base64"
            
            # Verify API was called
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert "kroki.io" in call_url
    
    def test_convert_error_fallback(self, diagram_generator):
        """Test fallback when conversion fails."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            base64_image = diagram_generator.convert_mermaid_to_image("invalid")
            
            assert base64_image == ""


class TestSaveDiagrams:
    """Test saving diagrams to YAML."""
    
    def test_save_diagrams_to_yaml(self, diagram_generator, tmp_path):
        """Test saving diagrams to YAML file."""
        diagrams = [
            GeneratedDiagram(
                heading="System Architecture",
                diagram_type="flowchart",
                mermaid_code="flowchart TD\n    A --> B",
                image_base64="base64string",
                score=9.5,
                description="System architecture diagram"
            ),
            GeneratedDiagram(
                heading="Deployment",
                diagram_type="sequenceDiagram",
                mermaid_code="sequenceDiagram\n    A->>B: Deploy",
                image_base64="base64string2",
                score=9.0,
                description="Deployment sequence"
            )
        ]
        
        output_path = tmp_path / "diagrams.yaml"
        diagram_generator.save_diagrams(diagrams, output_path)
        
        assert output_path.exists()
        
        import yaml
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data['total_diagrams'] == 2
        assert len(data['diagrams']) == 2
        assert data['diagrams'][0]['heading'] == "System Architecture"
        assert data['diagrams'][1]['score'] == 9.0
