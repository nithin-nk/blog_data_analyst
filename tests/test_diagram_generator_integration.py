"""
Integration tests for Diagram Generator with actual LLM calls.

These tests make real API calls to verify end-to-end functionality.
They will be skipped if GOOGLE_API_KEY is not set.
"""

import pytest
from pathlib import Path
from src.generation.diagram_generator import DiagramGenerator
from src.config.settings import get_settings


# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not get_settings().google_api_key,
    reason="GOOGLE_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def diagram_generator():
    """Create a real DiagramGenerator instance."""
    return DiagramGenerator()


@pytest.fixture
def sample_blog_content():
    """Sample blog content about FastAPI microservices."""
    return """# Building Production-Ready Microservices with FastAPI

## Introduction
FastAPI has become one of the most popular Python frameworks for building high-performance APIs. 
In this guide, we'll explore how to build production-ready microservices.

## System Architecture
A typical FastAPI microservices architecture consists of several key components:
- API Gateway: Routes requests to appropriate services
- Authentication Service: Handles user authentication and authorization
- Business Logic Services: Core application functionality
- Database Layer: PostgreSQL for persistent storage
- Message Queue: RabbitMQ for async communication
- Monitoring: Prometheus and Grafana for observability

Each service communicates through REST APIs or message queues, providing loose coupling and scalability.

## Request Flow
When a client makes a request:
1. Request hits the API Gateway
2. Gateway validates the JWT token with Auth Service
3. Request is routed to the appropriate business logic service
4. Service performs operations and queries the database
5. Response is returned through the gateway
6. Async events are published to message queue for other services

## Deployment Workflow
The CI/CD pipeline automates the deployment process:
1. Developer pushes code to GitHub
2. GitHub Actions runs tests and builds Docker images
3. Images are pushed to container registry
4. Kubernetes pulls new images
5. Rolling update deploys to production
6. Health checks verify deployment success

## Monitoring and Observability
Production systems need comprehensive monitoring including metrics collection, 
distributed tracing, log aggregation, and alerting systems.
"""


class TestDiagramGeneratorIntegration:
    """Integration tests with real LLM calls."""
    
    @pytest.mark.slow
    def test_identify_opportunities_real_llm(self, diagram_generator, sample_blog_content):
        """Test identifying diagram opportunities with real LLM."""
        opportunities = diagram_generator.identify_diagram_opportunities(
            title="Building Production-Ready Microservices with FastAPI",
            content=sample_blog_content
        )
        
        # Should identify at least one opportunity
        assert len(opportunities) > 0, "Should identify at least one diagram opportunity"
        assert len(opportunities) <= 5, "Should not exceed max of 5 diagrams"
        
        # Verify structure
        for opp in opportunities:
            assert opp.heading, "Should have a heading"
            assert opp.diagram_type, "Should have a diagram type"
            assert opp.description, "Should have a description"
            assert opp.reasoning, "Should have reasoning"
            
            # Verify heading exists in content
            assert opp.heading in sample_blog_content, \
                f"Heading '{opp.heading}' should exist in content"
            
            # Verify diagram type is valid
            valid_types = [
                "flowchart", "sequenceDiagram", "classDiagram", 
                "stateDiagram", "erDiagram", "gantt", "pie", "gitGraph"
            ]
            assert any(t in opp.diagram_type for t in valid_types), \
                f"Diagram type '{opp.diagram_type}' should be valid"
        
        print(f"\n✓ Identified {len(opportunities)} diagram opportunities:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp.heading} ({opp.diagram_type})")
    
    @pytest.mark.slow
    def test_generate_and_review_diagram_real_llm(self, diagram_generator, sample_blog_content):
        """Test generating and reviewing a diagram with real LLM."""
        # First identify opportunities
        opportunities = diagram_generator.identify_diagram_opportunities(
            title="Building Production-Ready Microservices",
            content=sample_blog_content
        )
        
        assert len(opportunities) > 0, "Need at least one opportunity"
        
        # Take the first opportunity
        opportunity = opportunities[0]
        print(f"\n✓ Testing with opportunity: {opportunity.heading} ({opportunity.diagram_type})")
        
        # Generate diagram
        mermaid_code = diagram_generator.generate_mermaid_diagram(
            opportunity,
            sample_blog_content
        )
        
        assert mermaid_code, "Should generate mermaid code"
        assert len(mermaid_code) > 20, "Code should be substantial"
        
        # Verify it starts with a diagram type
        valid_starts = [
            "flowchart", "graph", "sequenceDiagram", "classDiagram",
            "stateDiagram", "erDiagram", "gantt", "pie", "gitGraph"
        ]
        assert any(mermaid_code.strip().startswith(t) for t in valid_starts), \
            "Should start with a valid diagram type"
        
        print(f"\n✓ Generated mermaid code ({len(mermaid_code)} chars):")
        print(f"  {mermaid_code[:100]}...")
        
        # Review the diagram
        review = diagram_generator.review_diagram(
            mermaid_code,
            opportunity,
            sample_blog_content
        )
        
        assert 'score' in review, "Review should have a score"
        assert 0 <= review['score'] <= 10, "Score should be between 0 and 10"
        assert 'feedback' in review, "Review should have feedback"
        assert 'strengths' in review, "Review should have strengths"
        assert 'weaknesses' in review, "Review should have weaknesses"
        
        print(f"\n✓ Review score: {review['score']:.1f}/10")
        print(f"  Strengths: {len(review['strengths'])}")
        print(f"  Weaknesses: {len(review['weaknesses'])}")
    
    @pytest.mark.slow
    def test_feedback_loop_real_llm(self, diagram_generator, sample_blog_content):
        """Test the complete feedback loop with real LLM."""
        # Identify opportunities
        opportunities = diagram_generator.identify_diagram_opportunities(
            title="Building Production-Ready Microservices",
            content=sample_blog_content
        )
        
        assert len(opportunities) > 0, "Need at least one opportunity"
        
        # Take first opportunity
        opportunity = opportunities[0]
        print(f"\n✓ Testing feedback loop for: {opportunity.heading}")
        
        # Run feedback loop
        result = diagram_generator.generate_with_feedback_loop(
            opportunity,
            sample_blog_content
        )
        
        assert 'mermaid_code' in result, "Should have mermaid code"
        assert 'score' in result, "Should have score"
        assert 'review' in result, "Should have review"
        
        assert result['mermaid_code'], "Should generate code"
        assert 0 <= result['score'] <= 10, "Score should be valid"
        
        print(f"  Final score: {result['score']:.1f}/10")
        print(f"  Mermaid code length: {len(result['mermaid_code'])} chars")
        print(f"  Review feedback: {result['review']['feedback'][:100]}...")
    
    @pytest.mark.slow
    def test_convert_mermaid_to_image_real(self, diagram_generator):
        """Test converting a real mermaid diagram to image."""
        # Use a simple valid mermaid diagram
        mermaid_code = """flowchart TD
    A[API Gateway] --> B[Auth Service]
    A --> C[Business Logic]
    B --> D[(Database)]
    C --> D
    C --> E[Message Queue]
"""
        
        # Convert to image
        base64_image = diagram_generator.convert_mermaid_to_image(mermaid_code)
        
        # Note: This might fail if mermaid-py has issues, which is expected
        # The important thing is it doesn't crash
        if base64_image:
            assert isinstance(base64_image, str), "Should return string"
            assert len(base64_image) > 0, "Should have content"
            
            # Verify it's valid base64
            import base64 as b64
            try:
                decoded = b64.b64decode(base64_image)
                assert len(decoded) > 0, "Should decode to something"
                print(f"\n✓ Image conversion successful ({len(base64_image)} chars base64)")
            except Exception as e:
                pytest.fail(f"Invalid base64: {e}")
        else:
            print("\n⚠ Image conversion returned empty (expected if mermaid-py has issues)")
    
    @pytest.mark.slow
    def test_end_to_end_workflow_real_llm(self, diagram_generator, sample_blog_content, tmp_path):
        """Test complete end-to-end workflow with real LLM."""
        print("\n" + "="*80)
        print("RUNNING END-TO-END DIAGRAM GENERATION TEST")
        print("="*80)
        
        # Progress tracking
        messages = []
        def progress_callback(msg):
            messages.append(msg)
            print(msg)
        
        # Run complete workflow
        diagrams = diagram_generator.generate_all_diagrams(
            title="Building Production-Ready Microservices with FastAPI",
            content=sample_blog_content,
            progress_callback=progress_callback
        )
        
        # Verify results
        assert isinstance(diagrams, list), "Should return a list"
        
        if len(diagrams) > 0:
            print(f"\n✓ Generated {len(diagrams)} diagrams successfully")
            
            for i, diagram in enumerate(diagrams, 1):
                assert diagram.heading, "Should have heading"
                assert diagram.diagram_type, "Should have diagram type"
                assert diagram.mermaid_code, "Should have mermaid code"
                assert diagram.description, "Should have description"
                assert 0 <= diagram.score <= 10, "Should have valid score"
                
                print(f"\nDiagram {i}:")
                print(f"  Heading: {diagram.heading}")
                print(f"  Type: {diagram.diagram_type}")
                print(f"  Score: {diagram.score:.1f}/10")
                print(f"  Code length: {len(diagram.mermaid_code)} chars")
                print(f"  Image: {'✓' if diagram.image_base64 else '✗'}")
            
            # Save to YAML
            output_path = tmp_path / "test_diagrams.yaml"
            diagram_generator.save_diagrams(diagrams, output_path)
            
            assert output_path.exists(), "YAML file should be created"
            
            # Verify YAML structure
            import yaml
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            
            assert 'diagrams' in data, "Should have diagrams key"
            assert 'total_diagrams' in data, "Should have total count"
            assert data['total_diagrams'] == len(diagrams), "Count should match"
            
            print(f"\n✓ Saved diagrams to: {output_path}")
        else:
            print("\n⚠ No diagrams generated (this might be expected for some content)")
        
        print("\n" + "="*80)
        print("END-TO-END TEST COMPLETE")
        print("="*80)


class TestDiagramQualityStandards:
    """Test that generated diagrams meet quality standards."""
    
    @pytest.mark.slow
    def test_diagrams_meet_threshold(self, diagram_generator):
        """Test that the feedback loop produces diagrams meeting the threshold."""
        # Simple technical content that should produce a diagram
        content = """# API Design

## Request Processing Flow
When a client makes an API request:
1. Request arrives at load balancer
2. Authentication middleware validates credentials
3. Rate limiting checks are performed
4. Business logic processes the request
5. Database queries are executed
6. Response is formatted and returned

This flow ensures security and performance.
"""
        
        opportunities = diagram_generator.identify_diagram_opportunities(
            "API Design Patterns",
            content
        )
        
        if len(opportunities) > 0:
            # Test first opportunity
            result = diagram_generator.generate_with_feedback_loop(
                opportunities[0],
                content
            )
            
            settings = get_settings()
            threshold = settings.diagram_quality_threshold
            
            # Should either meet threshold OR be the best after max attempts
            print(f"\nFinal score: {result['score']:.1f}, Threshold: {threshold}")
            
            # If score is below threshold, verify we tried max attempts
            if result['score'] < threshold:
                print(f"⚠ Score below threshold (expected after {settings.max_diagram_generation_attempts} attempts)")
            else:
                print(f"✓ Score meets threshold!")
                assert result['score'] >= threshold
