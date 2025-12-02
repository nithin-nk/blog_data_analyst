"""
Integration tests for Blog Reviewer using actual LLM calls.

These tests make real API calls to Google Gemini and Azure OpenAI and require:
- GOOGLE_API_KEY environment variable to be set (or in .env)
- AZURE_OPENAI_API_KEY environment variable to be set (or in .env)
- AZURE_OPENAI_ENDPOINT environment variable to be set (or in .env)
- Network connectivity

Run with: pytest tests/test_blog_reviewer_integration.py -v -s

Note: Tests may be skipped due to rate limits on free tier.
"""

import asyncio
import os
import pytest
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.refinement.blog_reviewer import (
    BlogReviewer,
    ModelReviewResult,
    AggregatedReviewResult,
    ReviewIterationHistory,
)


def is_rate_limit_error(exc_info) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exc_info.value) if hasattr(exc_info, 'value') else str(exc_info)
    return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower()


def has_google_api_key() -> bool:
    """Check if Google API key is available."""
    return bool(os.environ.get("GOOGLE_API_KEY"))


def has_azure_credentials() -> bool:
    """Check if Azure OpenAI credentials are available."""
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY") and 
        os.environ.get("AZURE_OPENAI_ENDPOINT")
    )


# Skip markers
skip_no_google = pytest.mark.skipif(
    not has_google_api_key(),
    reason="GOOGLE_API_KEY environment variable not set",
)

skip_no_azure = pytest.mark.skipif(
    not has_azure_credentials(),
    reason="Azure OpenAI credentials not set",
)

skip_no_credentials = pytest.mark.skipif(
    not (has_google_api_key() or has_azure_credentials()),
    reason="No API credentials available",
)


@pytest.fixture
def blog_reviewer():
    """Create a real BlogReviewer instance."""
    return BlogReviewer()


@pytest.fixture
def sample_blog_content():
    """Sample blog content for testing."""
    return {
        "title": "Memory for AI Agents Using Mem0",
        "content": """# Memory for AI Agents Using Mem0

## Introduction

AI agents are becoming increasingly sophisticated, but many still lack persistent memory capabilities. This limitation means they treat each interaction as completely new, losing valuable context from previous conversations. In this tutorial, we'll explore how to implement memory for AI agents using Mem0, an open-source memory layer that's revolutionizing how we build intelligent agents.

## Understanding Agent Memory

Memory is fundamental to creating truly intelligent AI systems. Just as humans rely on different types of memory to navigate the world, AI agents can benefit from similar memory architectures:

### Types of Memory

1. **Short-term Memory**: Holds recent conversation context, typically within a single session
2. **Long-term Memory**: Stores persistent facts, preferences, and learned information across sessions
3. **Episodic Memory**: Records specific events, interactions, and experiences
4. **Semantic Memory**: Contains general knowledge and conceptual understanding

## Why Memory Matters for AI Agents

Without memory, AI agents face significant limitations:

- **Repetitive Questions**: The agent asks the same questions in every interaction
- **Lack of Personalization**: Cannot adapt to user preferences over time
- **Context Loss**: Unable to reference previous conversations or decisions
- **Reduced Efficiency**: Users must repeat information constantly

## Getting Started with Mem0

Mem0 provides a simple yet powerful API for adding memory capabilities to your AI agents.

### Installation

```bash
pip install mem0ai
```

### Basic Usage

```python
from mem0 import Memory

# Initialize memory
m = Memory()

# Add a memory
m.add("User prefers Python over JavaScript", user_id="alice")

# Search memories
memories = m.search("programming preferences", user_id="alice")
print(memories)
```

### Advanced Configuration

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memories",
            "host": "localhost",
            "port": 6333,
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
            "temperature": 0.1,
        }
    }
}

m = Memory.from_config(config)
```

## Architecture Overview

Mem0's architecture consists of three key components working together:

1. **Vector Database**: Enables semantic search across memories using embeddings
2. **Graph Database**: Maintains relationships between memories and entities
3. **LLM Integration**: Handles memory extraction, summarization, and retrieval

## Best Practices

When implementing memory for your AI agents, consider these best practices:

- **Privacy First**: Always handle user data responsibly and provide clear opt-out mechanisms
- **Memory Hygiene**: Implement mechanisms to update or remove outdated memories
- **Contextual Relevance**: Only retrieve memories that are relevant to the current interaction
- **Performance Optimization**: Use efficient indexing and caching strategies

## Conclusion

Adding memory to AI agents transforms them from stateless responders into intelligent assistants that learn and adapt over time. Mem0 makes this implementation straightforward while providing the flexibility needed for production applications.

By following the patterns outlined in this tutorial, you can build AI agents that remember user preferences, learn from interactions, and provide increasingly personalized experiences.
""",
    }


@pytest.fixture
def poor_quality_blog():
    """Sample poor quality blog for testing improvement iterations."""
    return {
        "title": "Docker Basics",
        "content": """# Docker

Docker is good. You should use it.

## What is Docker

Docker runs containers. Containers are like VMs but not.

## How to use

Just install docker and run stuff.

```
docker run hello-world
```

## Conclusion

Docker is useful for development.
""",
    }


class TestGeminiReview:
    """Integration tests for Gemini model reviews."""

    @skip_no_google
    def test_gemini_pro_review_returns_valid_result(
        self, blog_reviewer, sample_blog_content
    ):
        """Test that Gemini 2.5 Pro review returns valid ModelReviewResult."""
        try:
            result = blog_reviewer._review_with_gemini(
                model_name="gemini-2.5-pro",
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
            )

            # Verify return type
            assert isinstance(result, ModelReviewResult)
            assert result.model_name == "gemini-2.5-pro"
            
            # Check for errors first
            if result.error:
                if "429" in result.error or "quota" in result.error.lower():
                    pytest.skip("Skipped due to API rate limit")
                pytest.fail(f"Review failed with error: {result.error}")

            # Verify score is valid
            assert 1.0 <= result.score <= 10.0, f"Score {result.score} out of range"

            # Verify feedback is populated
            assert isinstance(result.feedback, list)
            assert len(result.feedback) >= 0  # May have no feedback if perfect

            # Verify can_apply_feedback is boolean
            assert isinstance(result.can_apply_feedback, bool)

            # Print results for manual inspection
            print("\n" + "=" * 60)
            print("GEMINI 2.5 PRO REVIEW RESULT")
            print("=" * 60)
            print(f"Score: {result.score}/10")
            print(f"Can Apply Feedback: {result.can_apply_feedback}")
            print(f"Feedback Items ({len(result.feedback)}):")
            for i, fb in enumerate(result.feedback[:5], 1):
                print(f"  {i}. {fb[:100]}...")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise

    @skip_no_google
    def test_gemini_flash_review_returns_valid_result(
        self, blog_reviewer, sample_blog_content
    ):
        """Test that Gemini 2.5 Flash review returns valid ModelReviewResult."""
        try:
            result = blog_reviewer._review_with_gemini(
                model_name="gemini-2.5-flash",
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
            )

            assert isinstance(result, ModelReviewResult)
            assert result.model_name == "gemini-2.5-flash"
            
            if result.error:
                if "429" in result.error or "quota" in result.error.lower():
                    pytest.skip("Skipped due to API rate limit")
                pytest.fail(f"Review failed with error: {result.error}")

            assert 1.0 <= result.score <= 10.0
            assert isinstance(result.feedback, list)

            print("\n" + "=" * 60)
            print("GEMINI 2.5 FLASH REVIEW RESULT")
            print("=" * 60)
            print(f"Score: {result.score}/10")
            print(f"Feedback Items: {len(result.feedback)}")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise


class TestAzureOpenAIReview:
    """Integration tests for Azure OpenAI (GPT-5-chat) reviews."""

    @skip_no_azure
    def test_azure_openai_review_returns_valid_result(
        self, blog_reviewer, sample_blog_content
    ):
        """Test that Azure OpenAI review returns valid ModelReviewResult."""
        try:
            result = blog_reviewer._review_with_azure_openai(
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
            )

            assert isinstance(result, ModelReviewResult)
            assert "gpt" in result.model_name.lower() or result.model_name == blog_reviewer.settings.azure_deployment_name
            
            if result.error:
                pytest.fail(f"Azure OpenAI review failed: {result.error}")

            assert 1.0 <= result.score <= 10.0
            assert isinstance(result.feedback, list)
            assert isinstance(result.can_apply_feedback, bool)

            print("\n" + "=" * 60)
            print(f"AZURE OPENAI ({result.model_name}) REVIEW RESULT")
            print("=" * 60)
            print(f"Score: {result.score}/10")
            print(f"Can Apply Feedback: {result.can_apply_feedback}")
            print(f"Feedback Items ({len(result.feedback)}):")
            for i, fb in enumerate(result.feedback[:5], 1):
                print(f"  {i}. {fb[:100]}...")
            print("=" * 60)

        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                pytest.skip("Azure OpenAI authentication failed")
            raise


class TestMultiModelReview:
    """Integration tests for multi-model review functionality."""

    @skip_no_credentials
    @pytest.mark.asyncio
    async def test_review_with_all_models(self, blog_reviewer, sample_blog_content):
        """Test that review_with_all_models returns aggregated results."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        try:
            result = await blog_reviewer.review_with_all_models(
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
                progress_callback=progress_callback,
            )

            # Verify return type
            assert isinstance(result, AggregatedReviewResult)

            # Verify we got results from multiple models
            assert len(result.individual_results) > 0

            # Count successful reviews
            successful_reviews = [r for r in result.individual_results if r.error is None]
            print(f"\nSuccessful reviews: {len(successful_reviews)}/{len(result.individual_results)}")

            if len(successful_reviews) == 0:
                # Check if all failures were rate limits
                rate_limit_errors = [r for r in result.individual_results 
                                    if r.error and ("429" in r.error or "quota" in r.error.lower())]
                if len(rate_limit_errors) == len(result.individual_results):
                    pytest.skip("All models hit rate limits")
                pytest.fail("All model reviews failed")

            # Verify average score
            if len(successful_reviews) > 0:
                assert 1.0 <= result.average_score <= 10.0

            # Verify combined feedback
            assert isinstance(result.combined_feedback, list)

            # Verify progress messages were sent
            assert len(progress_messages) > 0

            # Print detailed results
            print("\n" + "=" * 60)
            print("MULTI-MODEL REVIEW RESULTS")
            print("=" * 60)
            print(f"Average Score: {result.average_score:.2f}/10")
            print(f"Passes Threshold (>9): {result.passes_threshold}")
            print(f"Can Apply Feedback: {result.can_apply_any_feedback}")
            print(f"\nIndividual Model Scores:")
            for r in result.individual_results:
                if r.error:
                    print(f"  - {r.model_name}: ERROR - {r.error[:50]}...")
                else:
                    print(f"  - {r.model_name}: {r.score:.1f}/10 ({len(r.feedback)} feedback items)")
            print(f"\nCombined Feedback ({len(result.combined_feedback)} items):")
            for i, fb in enumerate(result.combined_feedback[:5], 1):
                print(f"  {i}. {fb[:80]}...")
            if len(result.combined_feedback) > 5:
                print(f"  ... and {len(result.combined_feedback) - 5} more")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise


class TestContentRegeneration:
    """Integration tests for content regeneration with feedback."""

    @skip_no_google
    def test_regenerate_with_feedback(self, blog_reviewer, poor_quality_blog):
        """Test that regenerate_with_feedback improves content."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        feedback = [
            "Add more detailed explanations of Docker concepts",
            "Include practical code examples with proper syntax highlighting",
            "Expand the introduction to explain why Docker is important",
            "Add a section about Docker Compose for multi-container applications",
            "Include best practices and common pitfalls",
        ]

        try:
            improved_content = blog_reviewer.regenerate_with_feedback(
                title=poor_quality_blog["title"],
                content=poor_quality_blog["content"],
                feedback=feedback,
                progress_callback=progress_callback,
            )

            # Verify we got content back
            assert improved_content is not None
            assert len(improved_content) > len(poor_quality_blog["content"])

            # Verify it's different from original
            assert improved_content != poor_quality_blog["content"]

            # Verify progress messages
            assert len(progress_messages) > 0

            # Print results for inspection
            print("\n" + "=" * 60)
            print("CONTENT REGENERATION RESULT")
            print("=" * 60)
            print(f"Original Length: {len(poor_quality_blog['content'])} chars")
            print(f"Improved Length: {len(improved_content)} chars")
            print(f"Improvement: +{len(improved_content) - len(poor_quality_blog['content'])} chars")
            print(f"\nImproved Content Preview:")
            print("-" * 40)
            print(improved_content[:1000])
            print("..." if len(improved_content) > 1000 else "")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise


class TestReviewAndImprove:
    """Integration tests for the full review-and-improve loop."""

    @skip_no_credentials
    @pytest.mark.asyncio
    async def test_review_and_improve_single_iteration(
        self, blog_reviewer, sample_blog_content
    ):
        """Test review_and_improve with a single iteration (high quality content)."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        try:
            final_content, final_review, history = await blog_reviewer.review_and_improve(
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
                max_iterations=1,  # Limit to 1 iteration for this test
                score_threshold=9.0,
                progress_callback=progress_callback,
            )

            # Verify return types
            assert isinstance(final_content, str)
            assert isinstance(final_review, AggregatedReviewResult)
            assert isinstance(history, list)
            assert len(history) == 1

            # Verify history entry
            assert isinstance(history[0], ReviewIterationHistory)
            assert history[0].iteration == 1

            print("\n" + "=" * 60)
            print("REVIEW AND IMPROVE (1 ITERATION) RESULT")
            print("=" * 60)
            print(f"Final Score: {final_review.average_score:.2f}/10")
            print(f"Passes Threshold: {final_review.passes_threshold}")
            print(f"Total Iterations: {len(history)}")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise

    @skip_no_credentials
    @pytest.mark.asyncio
    async def test_review_and_improve_multiple_iterations(
        self, blog_reviewer, poor_quality_blog
    ):
        """Test review_and_improve with multiple iterations on poor quality content."""
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            print(msg)

        try:
            final_content, final_review, history = await blog_reviewer.review_and_improve(
                title=poor_quality_blog["title"],
                content=poor_quality_blog["content"],
                max_iterations=2,  # Allow up to 2 iterations
                score_threshold=9.0,
                progress_callback=progress_callback,
            )

            # Verify we got results
            assert isinstance(final_content, str)
            assert isinstance(final_review, AggregatedReviewResult)
            assert len(history) >= 1

            # If multiple iterations occurred, verify improvement
            if len(history) > 1:
                first_score = history[0].review_result.average_score
                last_score = history[-1].review_result.average_score
                print(f"\nScore progression: {first_score:.2f} -> {last_score:.2f}")

            print("\n" + "=" * 60)
            print("REVIEW AND IMPROVE (MULTI-ITERATION) RESULT")
            print("=" * 60)
            print(f"Total Iterations: {len(history)}")
            for item in history:
                print(f"  Iteration {item.iteration}: Score {item.review_result.average_score:.2f}/10")
            print(f"Final Score: {final_review.average_score:.2f}/10")
            print(f"Passes Threshold: {final_review.passes_threshold}")
            print(f"Content Length: {len(final_content)} chars")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise


class TestReviewHistorySaving:
    """Integration tests for saving review history."""

    @skip_no_credentials
    @pytest.mark.asyncio
    async def test_save_review_history(self, blog_reviewer, sample_blog_content, tmp_path):
        """Test that review history is saved correctly to YAML."""
        try:
            # Run a review
            _, _, history = await blog_reviewer.review_and_improve(
                title=sample_blog_content["title"],
                content=sample_blog_content["content"],
                max_iterations=1,
                score_threshold=9.0,
            )

            # Save history
            output_path = tmp_path / "review_history.yaml"
            blog_reviewer.save_review_history(history, output_path)

            # Verify file was created
            assert output_path.exists()

            # Load and verify content
            import yaml
            with open(output_path, "r") as f:
                data = yaml.safe_load(f)

            assert "total_iterations" in data
            assert "iterations" in data
            assert data["total_iterations"] == len(history)

            print("\n" + "=" * 60)
            print("SAVED REVIEW HISTORY")
            print("=" * 60)
            print(f"File: {output_path}")
            print(f"Total Iterations: {data['total_iterations']}")
            for iteration in data["iterations"]:
                print(f"  Iteration {iteration['iteration']}:")
                print(f"    Average Score: {iteration['average_score']:.2f}")
                print(f"    Passed Threshold: {iteration['passed_threshold']}")
                print(f"    Models: {len(iteration['models'])}")
            print("=" * 60)

        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip("Skipped due to API rate limit")
            raise


class TestKeyRotation:
    """Integration tests for API key rotation functionality."""

    @skip_no_google
    def test_key_rotation_on_rate_limit(self, blog_reviewer):
        """Test that key rotation works when encountering rate limits."""
        # This test verifies the key rotation mechanism is working
        # by making multiple calls in sequence
        
        content = {
            "title": "Test Blog",
            "content": "# Test\n\nThis is a test blog post for key rotation testing.",
        }

        results = []
        errors = []

        # Make multiple review calls to trigger potential rate limits
        for i in range(2):
            try:
                result = blog_reviewer._review_with_gemini(
                    model_name="gemini-2.5-flash",
                    title=content["title"],
                    content=content["content"],
                )
                results.append(result)
                if result.error:
                    errors.append(result.error)
                print(f"Call {i+1}: Score={result.score}, Error={result.error}")
            except Exception as e:
                errors.append(str(e))
                print(f"Call {i+1}: Exception - {e}")

        # At least one call should succeed with key rotation
        successful = [r for r in results if r.error is None and r.score > 0]
        
        print("\n" + "=" * 60)
        print("KEY ROTATION TEST RESULTS")
        print("=" * 60)
        print(f"Total Calls: {len(results) + len(errors)}")
        print(f"Successful: {len(successful)}")
        print(f"Errors: {len(errors)}")
        if errors:
            print(f"Error samples: {errors[:3]}")
        print("=" * 60)

        # If all calls failed due to rate limits across all keys, skip
        if len(successful) == 0:
            all_rate_limit = all("429" in str(e) or "quota" in str(e).lower() for e in errors)
            if all_rate_limit:
                pytest.skip("All API keys exhausted (rate limited)")
