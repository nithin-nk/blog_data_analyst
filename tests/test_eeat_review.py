"""
Integration tests for E-E-A-T based content review functionality.

Tests verify that the content review system properly evaluates content based on
Google's Search Quality Guidelines and E-E-A-T principles.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.generation.content_generator import ContentGenerator, SectionReview


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.generation.content_generator.get_settings') as mock:
        settings = MagicMock()
        settings.google_api_key = "test_key"
        settings.section_reviewer_model = "gemini-2.5-flash"
        settings.max_section_review_iterations = 3
        settings.section_quality_threshold = 9.0
        mock.return_value = settings
        yield settings


@pytest.fixture
def content_generator(mock_settings):
    """Create a ContentGenerator instance with mocked dependencies."""
    with patch('src.generation.content_generator.ChatGoogleGenerativeAI'):
        generator = ContentGenerator()
        return generator


class TestEEATContentReview:
    """Test suite for E-E-A-T based content review."""

    def test_review_high_quality_eeat_content(self, content_generator, mock_settings):
        """Test that high-quality content with good E-E-A-T scores well."""
        topic = "Building Production-Ready AI Agents"
        heading = "Infrastructure Requirements"
        
        # High-quality content with E-E-A-T characteristics
        content = """
Infrastructure Requirements for Production AI Agents

## Critical Components

### Vector Database
* ChromaDB works for prototypes.
* Pinecone handles production scale.
* Weaviate offers hybrid search.

From my experience deploying 50+ agents:
* Response time correlates directly with database choice.
* Cold starts add 200-500ms latency.
* Batch processing reduces costs by 40%.

### Resource Allocation
* CPU: 2-4 cores minimum.
* RAM: 8GB baseline, 16GB recommended.
* GPU: Optional for embeddings, required for local LLMs.

I've tested this configuration across AWS, GCP, and Azure.
Results show consistent performance within 5% variance.

### Monitoring Stack
* Prometheus for metrics collection.
* Grafana for visualization.
* Custom dashboards track token usage.

Each agent generates approximately 1M tokens/day in production.
This data comes from analyzing our deployment logs over 6 months.
"""

        # Mock the LLM call to return a high score
        mock_review = SectionReview(
            score=9.2,
            feedback="Excellent content demonstrating strong E-E-A-T:\n"
                    "* Shows clear first-hand experience ('From my experience deploying 50+ agents')\n"
                    "* Provides specific, evidence-backed claims (numbers, percentages)\n"
                    "* Uses concise bullet points and short sentences\n"
                    "* Demonstrates expertise through practical insights\n"
                    "* People-first approach with actionable information"
        )

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            result = content_generator._review_section_content(topic, heading, content)
            
            # Verify high score
            assert result['score'] >= 9.0
            assert 'E-E-A-T' in result['feedback'] or 'experience' in result['feedback'].lower()
            
            # Verify the prompt includes E-E-A-T criteria
            call_args = mock_llm.call_args
            messages = call_args[0][0]
            prompt_content = messages[0].content
            
            assert 'E-E-A-T' in prompt_content
            assert 'Experience' in prompt_content
            assert 'Expertise' in prompt_content
            assert 'Authoritativeness' in prompt_content
            assert 'Trustworthiness' in prompt_content
            assert 'People-First' in prompt_content


    def test_review_poor_eeat_content(self, content_generator, mock_settings):
        """Test that content lacking E-E-A-T characteristics scores poorly."""
        topic = "AI Agent Deployment"
        heading = "Best Practices"
        
        # Poor content lacking E-E-A-T: generic, no evidence, long sentences
        content = """
When you are deploying AI agents to production environments, there are many different 
considerations that you need to take into account, including but not limited to the 
infrastructure requirements, the scalability concerns, the monitoring and observability 
setup, and the cost optimization strategies that you should be implementing in your 
deployment pipeline to ensure that your agents are running efficiently and effectively.

Many companies use various tools and technologies for deploying their AI agents, and 
there are several popular options available in the market today that you might want to 
consider for your specific use case, depending on your requirements and constraints.

It's important to follow best practices when deploying AI agents, and you should make 
sure that you have proper monitoring in place to track the performance of your agents 
and ensure that they are meeting your business objectives and user needs.
"""

        mock_review = SectionReview(
            score=4.5,
            feedback="Content lacks E-E-A-T and quality:\n"
                    "* No first-hand experience or practical examples\n"
                    "* Uses long, complex sentences (50+ words)\n"
                    "* No bullet points or concise formatting\n"
                    "* Generic statements without evidence or specifics\n"
                    "* Appears to be search-engine-first, not people-first\n"
                    "* No actionable insights or concrete recommendations\n"
                    "* Score below threshold due to lacking expertise demonstration"
        )

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            result = content_generator._review_section_content(topic, heading, content)
            
            # Verify low score
            assert result['score'] < 7.0
            assert any(keyword in result['feedback'].lower() 
                      for keyword in ['experience', 'evidence', 'specific', 'bullet', 'sentence'])


    def test_review_search_engine_first_content(self, content_generator, mock_settings):
        """Test that search-engine-first content scores below 6."""
        topic = "AI Agent Deployment Tools"
        heading = "Top 10 Best AI Agent Deployment Tools 2024"
        
        # Search-engine-first content: keyword stuffing, no real value
        content = """
AI agent deployment tools are essential for deploying AI agents. The best AI agent 
deployment tools help you deploy AI agents efficiently. When choosing AI agent deployment 
tools, consider these top AI agent deployment tools for 2024.

Top AI agent deployment tools include various AI agent deployment tools that offer 
different features. These AI agent deployment tools are widely used for AI agent deployment.

Many developers search for AI agent deployment tools to deploy their AI agents using 
the best AI agent deployment tools available in the market today.
"""

        mock_review = SectionReview(
            score=2.8,
            feedback="Severe quality issues:\n"
                    "* Obvious keyword stuffing ('AI agent deployment tools' repeated 12+ times)\n"
                    "* No original insights or practical information\n"
                    "* Search-engine-first content, not people-first\n"
                    "* No E-E-A-T demonstration whatsoever\n"
                    "* Provides no value to readers\n"
                    "* Appears to manipulate search rankings\n"
                    "* Score below 6 due to search-engine-first approach"
        )

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            result = content_generator._review_section_content(topic, heading, content)
            
            # Verify very low score
            assert result['score'] < 6.0
            assert 'search' in result['feedback'].lower() or 'keyword' in result['feedback'].lower()


    def test_review_criteria_weights(self, content_generator, mock_settings):
        """Test that the review criteria include proper weight distribution."""
        topic = "Test Topic"
        heading = "Test Heading"
        content = "Test content"

        # Mock review
        mock_review = SectionReview(score=8.0, feedback="Test feedback")

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            content_generator._review_section_content(topic, heading, content)
            
            # Verify the prompt includes proper weights
            call_args = mock_llm.call_args
            messages = call_args[0][0]
            prompt_content = messages[0].content
            
            # Check all criteria are present with weights
            assert 'Weight: 25%' in prompt_content  # Style Compliance
            assert 'Weight: 35%' in prompt_content  # E-E-A-T & People-First
            assert 'Weight: 25%' in prompt_content  # Content Quality
            assert 'Weight: 10%' in prompt_content  # Structure & Readability
            assert 'Weight: 5%' in prompt_content   # SEO


    def test_review_with_missing_bullet_points(self, content_generator, mock_settings):
        """Test that content without bullet points scores below 7."""
        topic = "AI Infrastructure"
        heading = "Deployment Guide"
        
        # Content without bullet points
        content = """
Deploying AI agents requires careful planning. First, you need to set up your 
infrastructure. Then, configure your database. After that, deploy your monitoring 
stack. Finally, test everything thoroughly before going to production.

The process can be complex but following these steps will help ensure success. 
Make sure to allocate sufficient resources and plan for scalability from the start.
"""

        mock_review = SectionReview(
            score=6.2,
            feedback="Needs improvement:\n"
                    "* No bullet points used - reduces scannability\n"
                    "* Some sentences are moderately long\n"
                    "* Content is somewhat generic\n"
                    "* Add bullet points for better readability\n"
                    "* Include specific examples and evidence"
        )

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            result = content_generator._review_section_content(topic, heading, content)
            
            # Verify score below 7
            assert result['score'] < 7.0
            assert 'bullet' in result['feedback'].lower()


    def test_review_scoring_guidelines(self, content_generator, mock_settings):
        """Test that the review includes proper scoring guidelines."""
        topic = "Test"
        heading = "Test"
        content = "Test"

        mock_review = SectionReview(score=7.5, feedback="Test")

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            content_generator._review_section_content(topic, heading, content)
            
            # Verify scoring guidelines are in prompt
            call_args = mock_llm.call_args
            messages = call_args[0][0]
            prompt_content = messages[0].content
            
            assert 'SCORING GUIDELINES:' in prompt_content
            assert '9-10: Exceptional' in prompt_content
            assert '7-8: Good quality' in prompt_content
            assert '5-6: Acceptable but needs significant improvements' in prompt_content
            assert '3-4: Poor' in prompt_content
            assert '1-2: Fundamentally flawed' in prompt_content


    def test_review_critical_requirements(self, content_generator, mock_settings):
        """Test that critical review requirements are specified in the prompt."""
        topic = "Test"
        heading = "Test"
        content = "Test"

        mock_review = SectionReview(score=8.0, feedback="Test")

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.return_value = mock_review
            
            content_generator._review_section_content(topic, heading, content)
            
            # Verify critical requirements are in prompt
            call_args = mock_llm.call_args
            messages = call_args[0][0]
            prompt_content = messages[0].content
            
            assert '**CRITICAL**' in prompt_content or 'CRITICAL:' in prompt_content
            assert 'experience/expertise' in prompt_content.lower()
            assert 'score below 7' in prompt_content
            assert 'search-engine-first' in prompt_content
            assert 'score below 6' in prompt_content


    def test_review_error_handling(self, content_generator, mock_settings):
        """Test that review errors are handled gracefully."""
        topic = "Test"
        heading = "Test"
        content = "Test"

        with patch('src.generation.content_generator.gemini_llm_call') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            result = content_generator._review_section_content(topic, heading, content)
            
            # Verify fallback behavior
            assert result['score'] == 5.0
            assert 'Review failed' in result['feedback']
            assert 'API Error' in result['feedback']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
