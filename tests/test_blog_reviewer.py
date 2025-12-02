"""
Unit tests for the BlogReviewer agent.

Tests the multi-model blog review functionality including:
- Individual model reviews
- Aggregated review results
- Iterative improvement loop
- Review history saving
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.refinement.blog_reviewer import (
    BlogReviewer,
    BlogReview,
    ModelReviewResult,
    AggregatedReviewResult,
    ReviewIterationHistory,
)


# Sample test content
SAMPLE_TITLE = "Getting Started with Docker Containers"
SAMPLE_CONTENT = """
# Getting Started with Docker Containers

Docker is a platform for developing, shipping, and running applications in containers.

## What is Docker?

Docker enables you to separate your applications from your infrastructure.

- Containers are lightweight
- They share the host OS kernel
- Fast startup times

## Installing Docker

To install Docker on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## Conclusion

Docker simplifies deployment and improves consistency across environments.
"""


class TestModelReviewResult:
    """Tests for ModelReviewResult dataclass."""
    
    def test_create_successful_result(self):
        result = ModelReviewResult(
            model_name="gemini-2.5-pro",
            score=8.5,
            feedback=["Add more code examples", "Improve introduction"],
            can_apply_feedback=True,
        )
        assert result.model_name == "gemini-2.5-pro"
        assert result.score == 8.5
        assert len(result.feedback) == 2
        assert result.can_apply_feedback is True
        assert result.error is None
    
    def test_create_error_result(self):
        result = ModelReviewResult(
            model_name="gpt-5-chat",
            score=0.0,
            feedback=[],
            can_apply_feedback=False,
            error="API rate limit exceeded",
        )
        assert result.error == "API rate limit exceeded"
        assert result.score == 0.0


class TestAggregatedReviewResult:
    """Tests for AggregatedReviewResult dataclass."""
    
    def test_passes_threshold_true(self):
        result = AggregatedReviewResult(
            average_score=9.5,
            individual_results=[],
            combined_feedback=["Minor improvement"],
            can_apply_any_feedback=True,
        )
        assert result.passes_threshold is True
    
    def test_passes_threshold_false(self):
        result = AggregatedReviewResult(
            average_score=8.5,
            individual_results=[],
            combined_feedback=["Major improvement needed"],
            can_apply_any_feedback=True,
        )
        assert result.passes_threshold is False
    
    def test_passes_threshold_boundary(self):
        # Exactly 9.0 should NOT pass (requirement is >9)
        result = AggregatedReviewResult(
            average_score=9.0,
            individual_results=[],
            combined_feedback=[],
            can_apply_any_feedback=True,
        )
        assert result.passes_threshold is False


class TestBlogReviewer:
    """Tests for BlogReviewer class."""
    
    @pytest.fixture
    def reviewer(self):
        """Create a BlogReviewer instance with mocked settings."""
        with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
            settings = Mock()
            settings.google_api_key = "test_key_1"
            settings.google_api_key_1 = "test_key_2"
            settings.google_api_key_2 = "test_key_3"
            settings.azure_openai_api_key = "test_azure_key"
            settings.azure_openai_endpoint = "https://test.openai.azure.com"
            settings.azure_api_version = "2025-01-01-preview"
            settings.azure_deployment_name = "gpt-5-chat"
            mock_settings.return_value = settings
            return BlogReviewer()
    
    def test_reviewer_initialization(self, reviewer):
        """Test BlogReviewer initializes correctly."""
        assert reviewer is not None
        assert reviewer._azure_client is None  # Lazy init
    
    def test_get_review_prompt(self, reviewer):
        """Test review prompt generation."""
        prompt = reviewer._get_review_prompt(SAMPLE_TITLE, SAMPLE_CONTENT)
        
        assert SAMPLE_TITLE in prompt
        assert "Docker" in prompt
        assert "EVALUATION CRITERIA" in prompt
        assert "E-E-A-T" in prompt
        assert "score" in prompt.lower()
        assert "feedback" in prompt.lower()
    
    @patch('src.utils.llm_helpers.gemini_llm_call')
    def test_review_with_gemini_success(self, mock_gemini_llm_call, reviewer):
        """Test successful Gemini review."""
        # Mock gemini_llm_call response
        mock_gemini_llm_call.return_value = '''```json
{
    "score": 8.0,
    "feedback": ["Add more examples", "Improve code snippets"],
    "can_apply_feedback": true
}
```'''
        
        result = reviewer._review_with_gemini("gemini-2.5-pro", SAMPLE_TITLE, SAMPLE_CONTENT)
        
        assert result.model_name == "gemini-2.5-pro"
        assert result.score == 8.0
        assert len(result.feedback) == 2
        assert result.can_apply_feedback is True
        assert result.error is None
    
    @patch('src.utils.llm_helpers.gemini_llm_call')
    def test_review_with_gemini_all_keys_fail(self, mock_gemini_llm_call, reviewer):
        """Test Gemini review when all API keys fail."""
        mock_gemini_llm_call.side_effect = Exception("Rate limit exceeded")
        
        result = reviewer._review_with_gemini("gemini-2.5-flash", SAMPLE_TITLE, SAMPLE_CONTENT)
        
        assert result.error is not None
        assert result.score == 0.0
        assert result.can_apply_feedback is False
    
    def test_review_with_azure_openai_success(self, reviewer):
        """Test successful Azure OpenAI review."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{"score": 7.5, "feedback": ["More detail needed"], "can_apply_feedback": true}'''
        mock_client.chat.completions.create.return_value = mock_response
        
        reviewer._azure_client = mock_client
        
        result = reviewer._review_with_azure_openai(SAMPLE_TITLE, SAMPLE_CONTENT)
        
        assert result.model_name == "gpt-5-chat"
        assert result.score == 7.5
        assert "More detail needed" in result.feedback
        assert result.error is None
    
    def test_review_with_azure_openai_failure(self, reviewer):
        """Test Azure OpenAI review failure."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        
        reviewer._azure_client = mock_client
        
        result = reviewer._review_with_azure_openai(SAMPLE_TITLE, SAMPLE_CONTENT)
        
        assert result.error is not None
        assert "Connection failed" in result.error
        assert result.score == 0.0


class TestBlogReviewerAsync:
    """Async tests for BlogReviewer."""
    
    @pytest.fixture
    def reviewer(self):
        """Create a BlogReviewer instance with mocked settings."""
        with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
            settings = Mock()
            settings.google_api_key = "test_key"
            settings.google_api_key_1 = ""
            settings.google_api_key_2 = ""
            settings.azure_openai_api_key = "test_azure_key"
            settings.azure_openai_endpoint = "https://test.openai.azure.com"
            settings.azure_api_version = "2025-01-01-preview"
            settings.azure_deployment_name = "gpt-5-chat"
            mock_settings.return_value = settings
            return BlogReviewer()
    
    @pytest.mark.asyncio
    async def test_review_with_all_models(self, reviewer):
        """Test parallel review with all models."""
        # Mock all review methods
        mock_results = [
            ModelReviewResult("gemini-2.5-pro", 8.5, ["fb1"], True),
            ModelReviewResult("gemini-2.5-flash", 8.0, ["fb2"], True),
            ModelReviewResult("gpt-5-chat", 7.5, ["fb3"], False),
        ]
        
        with patch.object(reviewer, '_review_with_gemini') as mock_gemini:
            with patch.object(reviewer, '_review_with_azure_openai') as mock_azure:
                mock_gemini.side_effect = [mock_results[0], mock_results[1]]
                mock_azure.return_value = mock_results[2]
                
                result = await reviewer.review_with_all_models(
                    SAMPLE_TITLE,
                    SAMPLE_CONTENT,
                )
        
        # Average of 8.5 + 8.0 + 7.5 = 24 / 3 = 8.0
        assert result.average_score == 8.0
        assert len(result.individual_results) == 3
        assert len(result.combined_feedback) == 3
        assert result.can_apply_any_feedback is True  # At least one can apply
    
    @pytest.mark.asyncio
    async def test_review_with_all_models_some_failures(self, reviewer):
        """Test parallel review when some models fail."""
        mock_results = [
            ModelReviewResult("gemini-2.5-pro", 9.0, ["fb1"], True),
            ModelReviewResult("gemini-2.5-flash", 0.0, [], False, error="Failed"),
            ModelReviewResult("gpt-5-chat", 8.0, ["fb2"], True),
        ]
        
        with patch.object(reviewer, '_review_with_gemini') as mock_gemini:
            with patch.object(reviewer, '_review_with_azure_openai') as mock_azure:
                mock_gemini.side_effect = [mock_results[0], mock_results[1]]
                mock_azure.return_value = mock_results[2]
                
                result = await reviewer.review_with_all_models(
                    SAMPLE_TITLE,
                    SAMPLE_CONTENT,
                )
        
        # Average of valid scores only: (9.0 + 8.0) / 2 = 8.5
        assert result.average_score == 8.5
        assert len(result.individual_results) == 3
    
    @pytest.mark.asyncio
    async def test_review_and_improve_passes_first_iteration(self, reviewer):
        """Test review loop passes on first iteration."""
        mock_review = AggregatedReviewResult(
            average_score=9.5,
            individual_results=[],
            combined_feedback=[],
            can_apply_any_feedback=True,
        )
        
        with patch.object(reviewer, 'review_with_all_models') as mock_review_method:
            mock_review_method.return_value = mock_review
            
            final_content, final_review, history = await reviewer.review_and_improve(
                SAMPLE_TITLE,
                SAMPLE_CONTENT,
                max_iterations=5,
                score_threshold=9.0,
            )
        
        assert final_review.average_score == 9.5
        assert len(history) == 1
        assert final_content == SAMPLE_CONTENT
    
    @pytest.mark.asyncio
    async def test_review_and_improve_iterates(self, reviewer):
        """Test review loop iterates and improves."""
        # First iteration: score 7.0
        # Second iteration: score 9.5 (passes)
        mock_reviews = [
            AggregatedReviewResult(7.0, [], ["Improve intro"], True),
            AggregatedReviewResult(9.5, [], [], True),
        ]
        
        with patch.object(reviewer, 'review_with_all_models') as mock_review:
            with patch.object(reviewer, 'regenerate_with_feedback') as mock_regen:
                mock_review.side_effect = mock_reviews
                mock_regen.return_value = "Improved content"
                
                final_content, final_review, history = await reviewer.review_and_improve(
                    SAMPLE_TITLE,
                    SAMPLE_CONTENT,
                    max_iterations=5,
                    score_threshold=9.0,
                )
        
        assert final_review.average_score == 9.5
        assert len(history) == 2
        assert mock_regen.called
    
    @pytest.mark.asyncio
    async def test_review_and_improve_max_iterations(self, reviewer):
        """Test review loop respects max iterations."""
        # All iterations below threshold
        mock_review = AggregatedReviewResult(7.0, [], ["Improve"], True)
        
        with patch.object(reviewer, 'review_with_all_models') as mock_review_method:
            with patch.object(reviewer, 'regenerate_with_feedback') as mock_regen:
                mock_review_method.return_value = mock_review
                mock_regen.return_value = "Improved content"
                
                final_content, final_review, history = await reviewer.review_and_improve(
                    SAMPLE_TITLE,
                    SAMPLE_CONTENT,
                    max_iterations=3,
                    score_threshold=9.0,
                )
        
        assert len(history) == 3  # Stopped at max
        assert mock_regen.call_count == 2  # Called for iterations 1->2, 2->3
    
    @pytest.mark.asyncio
    async def test_review_and_improve_selects_best_version(self, reviewer):
        """Test that best version is selected when threshold not met."""
        # Iteration 1: 8.0, Iteration 2: 8.5, Iteration 3: 7.5
        mock_reviews = [
            AggregatedReviewResult(8.0, [], ["fb1"], True),
            AggregatedReviewResult(8.5, [], ["fb2"], True),
            AggregatedReviewResult(7.5, [], ["fb3"], True),
        ]
        
        with patch.object(reviewer, 'review_with_all_models') as mock_review:
            with patch.object(reviewer, 'regenerate_with_feedback') as mock_regen:
                mock_review.side_effect = mock_reviews
                mock_regen.side_effect = ["Content v2", "Content v3"]
                
                final_content, final_review, history = await reviewer.review_and_improve(
                    SAMPLE_TITLE,
                    SAMPLE_CONTENT,
                    max_iterations=3,
                    score_threshold=9.0,
                )
        
        # Should select version from iteration 2 (best score 8.5)
        assert final_content == "Content v2"


class TestReviewHistorySaving:
    """Tests for saving review history."""
    
    @pytest.fixture
    def reviewer(self):
        with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
            settings = Mock()
            settings.google_api_key = ""
            settings.google_api_key_1 = ""
            settings.google_api_key_2 = ""
            settings.azure_openai_api_key = ""
            settings.azure_openai_endpoint = ""
            settings.azure_api_version = ""
            settings.azure_deployment_name = ""
            mock_settings.return_value = settings
            return BlogReviewer()
    
    def test_save_review_history(self, reviewer, tmp_path):
        """Test saving review history to YAML."""
        history = [
            ReviewIterationHistory(
                iteration=1,
                content_version="Content preview...",
                review_result=AggregatedReviewResult(
                    average_score=7.5,
                    individual_results=[
                        ModelReviewResult("gemini-2.5-pro", 8.0, ["fb1"], True),
                        ModelReviewResult("gpt-5-chat", 7.0, ["fb2"], False),
                    ],
                    combined_feedback=["fb1", "fb2"],
                    can_apply_any_feedback=True,
                ),
            ),
        ]
        
        output_path = tmp_path / "review_history.yaml"
        reviewer.save_review_history(history, output_path)
        
        assert output_path.exists()
        
        import yaml
        with open(output_path) as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["total_iterations"] == 1
        assert len(saved_data["iterations"]) == 1
        assert saved_data["iterations"][0]["average_score"] == 7.5
        assert len(saved_data["iterations"][0]["models"]) == 2


class TestRegenerateWithFeedback:
    """Tests for content regeneration."""
    
    @pytest.fixture
    def reviewer(self):
        with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
            settings = Mock()
            settings.google_api_key = "test_key"
            settings.google_api_key_1 = ""
            settings.google_api_key_2 = ""
            settings.azure_openai_api_key = ""
            settings.azure_openai_endpoint = ""
            settings.azure_api_version = ""
            settings.azure_deployment_name = ""
            settings.max_retries = 1
            settings.retry_delay = 0
            mock_settings.return_value = settings
            return BlogReviewer()
    
    def test_regenerate_with_feedback_success(self, reviewer):
        """Test successful content regeneration."""
        # Patch where it's imported (inside the function)
        with patch.dict('sys.modules', {'src.utils.llm_helpers': Mock()}):
            import sys
            mock_module = Mock()
            mock_module.gemini_llm_call = Mock(return_value="# Improved Content\n\nBetter content here.")
            sys.modules['src.utils.llm_helpers'] = mock_module
            
            # Need to call with fresh import
            from src.refinement.blog_reviewer import BlogReviewer as FreshBlogReviewer
            
            with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
                settings = Mock()
                settings.google_api_key = "test_key"
                settings.google_api_key_1 = ""
                settings.google_api_key_2 = ""
                settings.azure_openai_api_key = ""
                settings.azure_openai_endpoint = ""
                settings.azure_api_version = ""
                settings.azure_deployment_name = ""
                mock_settings.return_value = settings
                
                fresh_reviewer = FreshBlogReviewer()
                result = fresh_reviewer.regenerate_with_feedback(
                    title=SAMPLE_TITLE,
                    content=SAMPLE_CONTENT,
                    feedback=["Add more examples", "Improve introduction"],
                )
                
            assert "Improved Content" in result
    
    def test_regenerate_with_feedback_failure_returns_original(self, reviewer):
        """Test regeneration returns original on failure."""
        with patch.dict('sys.modules', {'src.utils.llm_helpers': Mock()}):
            import sys
            mock_module = Mock()
            mock_module.gemini_llm_call = Mock(side_effect=Exception("API Error"))
            sys.modules['src.utils.llm_helpers'] = mock_module
            
            from src.refinement.blog_reviewer import BlogReviewer as FreshBlogReviewer
            
            with patch('src.refinement.blog_reviewer.get_settings') as mock_settings:
                settings = Mock()
                settings.google_api_key = "test_key"
                settings.google_api_key_1 = ""
                settings.google_api_key_2 = ""
                settings.azure_openai_api_key = ""
                settings.azure_openai_endpoint = ""
                settings.azure_api_version = ""
                settings.azure_deployment_name = ""
                mock_settings.return_value = settings
                
                fresh_reviewer = FreshBlogReviewer()
                result = fresh_reviewer.regenerate_with_feedback(
                    title=SAMPLE_TITLE,
                    content=SAMPLE_CONTENT,
                    feedback=["Some feedback"],
                )
            
            assert result == SAMPLE_CONTENT  # Returns original on failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
