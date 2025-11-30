import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.main import run_research_pipeline
from src.planning.outline_reviewer import OutlineReview
from src.planning.outline_generator import BlogOutline, OutlineSection, OutlineMetadata

@pytest.mark.asyncio
async def test_outline_fallback_logic():
    """
    Test that the pipeline falls back to the best outline when max iterations are reached
    and the quality threshold is not met.
    """
    # Mock data
    topic = "Test Topic"
    context = "Test Context"
    output_dir = MagicMock()
    
    # Create mock outlines for different iterations
    outline_v1 = BlogOutline(
        topic=topic,
        sections=[OutlineSection(heading="V1", summary="Summary V1", references=[])],
        metadata=OutlineMetadata(target_audience="Test Audience", difficulty="Beginner", estimated_reading_time="10 minutes")
    )
    outline_v2 = BlogOutline(
        topic=topic,
        sections=[OutlineSection(heading="V2", summary="Summary V2", references=[])],
        metadata=OutlineMetadata(target_audience="Test Audience", difficulty="Beginner", estimated_reading_time="10 minutes")
    )
    outline_v3 = BlogOutline(
        topic=topic,
        sections=[OutlineSection(heading="V3", summary="Summary V3", references=[])],
        metadata=OutlineMetadata(target_audience="Test Audience", difficulty="Beginner", estimated_reading_time="10 minutes")
    )
    
    # Create mock reviews with different scores
    # Iteration 1: Score 7.0
    review_v1 = OutlineReview(
        score=7.0,
        completeness_score=7.0, logical_flow_score=7.0, depth_score=7.0,
        balance_score=7.0, audience_fit_score=7.0,
        strengths=["Good start"], weaknesses=["Needs work"],
        specific_feedback="Improve V1"
    )
    
    # Iteration 2: Score 8.5 (Best)
    review_v2 = OutlineReview(
        score=8.5,
        completeness_score=8.5, logical_flow_score=8.5, depth_score=8.5,
        balance_score=8.5, audience_fit_score=8.5,
        strengths=["Better"], weaknesses=["Still minor issues"],
        specific_feedback="Improve V2"
    )
    
    # Iteration 3: Score 6.0 (Worse)
    review_v3 = OutlineReview(
        score=6.0,
        completeness_score=6.0, logical_flow_score=6.0, depth_score=6.0,
        balance_score=6.0, audience_fit_score=6.0,
        strengths=["Okay"], weaknesses=["Regression"],
        specific_feedback="Fix V3"
    )
    
    # Mock dependencies
    with patch("src.main.FileHandler") as mock_file_handler, \
         patch("src.main.QuestionGenerator") as mock_q_gen, \
         patch("src.main.SearchAgent") as mock_search_agent, \
         patch("src.main.ContentExtractor") as mock_extractor, \
         patch("src.main.OutlineGenerator") as mock_outline_gen, \
         patch("src.main.OutlineReviewer") as mock_outline_reviewer, \
         patch("src.main.get_settings") as mock_get_settings:
        
        # Configure settings
        mock_settings = MagicMock()
        mock_settings.max_outline_iterations = 3
        mock_settings.outline_quality_threshold = 9.5  # High threshold to force fallback
        mock_settings.outline_reviewer_model = "test-model"
        mock_settings.input_dir = MagicMock()
        mock_get_settings.return_value = mock_settings
        
        # Mock FileHandler
        mock_file_handler.create_blog_structure.return_value = {
            "blog_dir": "test_dir",
            "research_questions": "questions.yaml",
            "search_results": "results.yaml",
            "extracted_content": "content.yaml",
            "outline_reviews": MagicMock(),  # Needs to be a path-like object or mock
            "outline": "outline.yaml"
        }
        
        # Mock QuestionGenerator
        mock_q_gen_instance = mock_q_gen.return_value
        mock_q_gen_instance.generate = AsyncMock(return_value=MagicMock(questions=["Q1"], categories_covered=["Cat1"]))
        
        # Mock SearchAgent
        mock_search_agent_instance = mock_search_agent.return_value
        mock_search_agent_instance.search_multiple = AsyncMock(return_value=MagicMock(
            successful_queries=1, total_results=1, all_urls=["http://test.com"],
            queries=[], url_to_queries={}
        ))
        
        # Mock ContentExtractor
        mock_extractor_instance = mock_extractor.return_value
        mock_extracted_item = MagicMock()
        mock_extracted_item.url = "http://test.com"
        mock_extracted_item.title = "Test Title"
        mock_extracted_item.success = True
        mock_extracted_item.markdown = "Test content"
        mock_extracted_item.headings = ["Heading 1"]
        mock_extracted_item.code_blocks = []

        mock_content = MagicMock()
        mock_content.contents = [mock_extracted_item]
        mock_content.statistics = {}
        mock_extractor_instance.extract_multiple = AsyncMock(return_value=mock_content)
        
        # Mock OutlineGenerator
        mock_outline_gen_instance = mock_outline_gen.return_value
        # Initial generation returns V1
        mock_outline_gen_instance.generate = AsyncMock(side_effect=[outline_v1]) 
        
        # Mock OutlineReviewer
        mock_outline_reviewer_instance = mock_outline_reviewer.return_value
        # Reviews return V1, V2, V3 scores in order
        mock_outline_reviewer_instance.review = AsyncMock(side_effect=[review_v1, review_v2, review_v3])

        # Regenerate returns V2 then V3
        # Note: _regenerate_with_feedback is called after review 1 (to get V2) and review 2 (to get V3)
        mock_outline_reviewer_instance._regenerate_with_feedback = AsyncMock(side_effect=[outline_v2, outline_v3])

        # Mock save_reviews method
        mock_outline_reviewer_instance.save_reviews = AsyncMock()
        
        # Run the pipeline
        result = await run_research_pipeline(
            topic=topic,
            context=context,
            output_dir=output_dir
        )
        
        # Verify results
        assert result["outline"] == outline_v2  # Should be V2 (Score 8.5), not V3 (Score 6.0)
        assert result["final_score"] == 8.5
        
        # Verify interactions
        # Should have called review 3 times
        assert mock_outline_reviewer_instance.review.call_count == 3
        
        # Should have called regenerate 2 times (after iter 1 and iter 2)
        assert mock_outline_reviewer_instance._regenerate_with_feedback.call_count == 2
