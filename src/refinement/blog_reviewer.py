"""
Blog reviewer agent for multi-model review.

Reviews blog content using multiple LLM models simultaneously and aggregates feedback.
Supports iterative improvement until quality threshold is met.
"""

import asyncio
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlogReview(BaseModel):
    """Structured output model for blog review."""
    score: float = Field(
        ge=1.0, le=10.0,
        description="Quality score from 1-10. Consider: Content Quality (30%), E-E-A-T (25%), Style & Readability (20%), Technical Accuracy (15%), SEO (10%)"
    )
    feedback: List[str] = Field(
        description="List of specific, actionable feedback items for improvement"
    )
    can_apply_feedback: bool = Field(
        description="True if the feedback can be applied using existing knowledge without external research. False if external data/research is needed."
    )


@dataclass
class ModelReviewResult:
    """Result from a single model's review."""
    model_name: str
    score: float
    feedback: List[str]
    can_apply_feedback: bool
    error: Optional[str] = None


@dataclass
class AggregatedReviewResult:
    """Aggregated review result from multiple models."""
    average_score: float
    individual_results: List[ModelReviewResult]
    combined_feedback: List[str]
    can_apply_any_feedback: bool
    
    @property
    def passes_threshold(self) -> bool:
        """Check if average score passes the threshold (>9)."""
        return self.average_score > 9.0


@dataclass
class ReviewIterationHistory:
    """Track history of review iterations."""
    iteration: int
    content_version: str
    review_result: AggregatedReviewResult
    

class BlogReviewer:
    """
    Multi-model blog reviewer agent.
    
    Reviews blog content using Gemini 2.5 Pro, Gemini 2.5 Flash, and Gemini Flash Latest
    sequentially, aggregates scores and feedback, and supports iterative improvement.
    """
    
    def __init__(self):
        self.settings = get_settings()
        logger.info("BlogReviewer initialized")
    
    def _get_review_prompt(self, title: str, content: str) -> str:
        """Generate the review prompt."""
        return f"""You are an expert blog content reviewer. Review the following blog post and provide detailed feedback.

BLOG TITLE: {title}

BLOG CONTENT:
{content}

EVALUATION CRITERIA:

1. **CONTENT QUALITY (Weight: 30%)**:
   - Is the content comprehensive and valuable?
   - Does it provide actionable insights?
   - Is it well-organized with clear flow?

2. **E-E-A-T (Weight: 25%)**:
   - Experience: Does it show first-hand experience?
   - Expertise: Is technical depth evident?
   - Authoritativeness: Would experts cite this?
   - Trustworthiness: Are claims accurate and sourced?

3. **STYLE & READABILITY (Weight: 20%)**:
   - Are sentences concise and clear?
   - Is formatting (headings, bullets, code) used well?
   - Is it engaging and easy to scan?

4. **TECHNICAL ACCURACY (Weight: 15%)**:
   - Are code examples correct and functional?
   - Are technical concepts explained accurately?
   - Are best practices followed?

5. **SEO & DISCOVERABILITY (Weight: 10%)**:
   - Does the title clearly convey the topic?
   - Are keywords naturally incorporated?
   - Is the content structured for search engines?

SCORING GUIDELINES:
* 9-10: Publication-ready, exceptional quality
* 7-8: Good quality, minor improvements needed
* 5-6: Acceptable, needs significant work
* 3-4: Poor quality, major rework required
* 1-2: Fundamentally flawed

IMPORTANT FOR FEEDBACK:
- Provide specific, actionable feedback items
- Focus on concrete improvements
- For each feedback item, consider if it can be fixed with existing knowledge (no external research needed)

Provide your review in this exact JSON format:
{{
    "score": <float between 1-10>,
    "feedback": ["feedback item 1", "feedback item 2", ...],
    "can_apply_feedback": <true if ALL feedback can be applied without external research, false otherwise>
}}
"""
    
    def _review_with_gemini(
        self,
        model_name: str,
        title: str,
        content: str,
    ) -> ModelReviewResult:
        """Review content with a Gemini model."""
        import json
        import time
        from src.utils.llm_helpers import gemini_llm_call
        
        prompt = self._get_review_prompt(title, content)
        messages = [HumanMessage(content=prompt)]
        
        try:
            response_text = gemini_llm_call(
                messages,
                model_name=model_name,
                settings=self.settings,
            )
            
            # Parse JSON response
            # Extract JSON from response (handle markdown code blocks)
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            review_data = json.loads(json_str)
            
            return ModelReviewResult(
                model_name=model_name,
                score=float(review_data.get("score", 5.0)),
                feedback=review_data.get("feedback", []),
                can_apply_feedback=review_data.get("can_apply_feedback", True),
            )
            
        except Exception as e:
            logger.warning(f"Gemini review failed for {model_name}: {e}")
            return ModelReviewResult(
                model_name=model_name,
                score=0.0,
                feedback=[],
                can_apply_feedback=False,
                error=str(e),
            )
    
    def _review_with_gemini_flash_preview(
        self,
        title: str,
        content: str,
    ) -> ModelReviewResult:
        """Review content with Gemini Flash Latest model."""
        return self._review_with_gemini(
            model_name="gemini-flash-latest",
            title=title,
            content=content,
        )
    
    def _is_rate_limit_error(self, error: str) -> bool:
        """Check if error is a rate limit error."""
        error_lower = error.lower()
        return "429" in error or "quota" in error_lower or "rate" in error_lower or "exhausted" in error_lower

    def _review_with_retry(
        self,
        review_func: Callable,
        model_name: str,
        title: str,
        content: str,
        max_retries: int = 3,
        retry_delay: int = 30,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ModelReviewResult:
        """
        Execute a review function with per-model retry on rate limit errors.
        
        Args:
            review_func: The review function to call (_review_with_gemini or _review_with_azure_openai)
            model_name: Name of the model for logging
            title: Blog title
            content: Blog content
            max_retries: Maximum retry attempts for this specific model
            retry_delay: Delay in seconds between retries
            progress_callback: Optional progress callback
            
        Returns:
            ModelReviewResult from the review
        """
        import time
        
        for attempt in range(1, max_retries + 1):
            if review_func == self._review_with_gemini_flash_preview:
                result = review_func(title, content)
            else:
                result = review_func(model_name, title, content)
            
            # If no error, return the result
            if result.error is None:
                return result
            
            # Check if it's a rate limit error
            if self._is_rate_limit_error(result.error):
                if attempt < max_retries:
                    if progress_callback:
                        progress_callback(f"   ⏳ {model_name}: Rate limit hit, waiting {retry_delay}s (retry {attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if progress_callback:
                        progress_callback(f"   ❌ {model_name}: Rate limit - max retries exceeded")
            else:
                # Non-rate-limit error, don't retry
                if progress_callback:
                    progress_callback(f"   ❌ {model_name}: Error - {result.error[:50]}...")
                break
        
        return result

    async def review_with_all_models(
        self,
        title: str,
        content: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> AggregatedReviewResult:
        """
        Review content with 3 different Gemini models sequentially.
        
        Models: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini Flash Latest
        Each model has its own retry logic for rate limit errors.
        
        Args:
            title: Blog title
            content: Blog content (markdown)
            progress_callback: Optional callback for progress updates
            
        Returns:
            AggregatedReviewResult with average score and combined feedback
        """
        if progress_callback:
            progress_callback("🔍 Starting multi-model review (sequential)...")
        
        model_results: List[ModelReviewResult] = []
        
        # Review 1: Gemini 2.5 Pro
        if progress_callback:
            progress_callback("   ├─ Gemini 2.5 Pro: reviewing...")
        
        gemini_pro_result = self._review_with_retry(
            review_func=self._review_with_gemini,
            model_name="gemini-2.5-pro",
            title=title,
            content=content,
            max_retries=3,
            retry_delay=30,
            progress_callback=progress_callback,
        )
        model_results.append(gemini_pro_result)
        
        if gemini_pro_result.error is None:
            if progress_callback:
                progress_callback(f"   ✓ Gemini 2.5 Pro: Score {gemini_pro_result.score:.1f}/10")
        
        # Review 2: Gemini 2.5 Flash
        if progress_callback:
            progress_callback("   ├─ Gemini 2.5 Flash: reviewing...")
        
        gemini_flash_result = self._review_with_retry(
            review_func=self._review_with_gemini,
            model_name="gemini-2.5-flash",
            title=title,
            content=content,
            max_retries=3,
            retry_delay=30,
            progress_callback=progress_callback,
        )
        model_results.append(gemini_flash_result)
        
        if gemini_flash_result.error is None:
            if progress_callback:
                progress_callback(f"   ✓ Gemini 2.5 Flash: Score {gemini_flash_result.score:.1f}/10")
        
        # Review 3: Gemini Flash Latest
        if progress_callback:
            progress_callback("   └─ Gemini Flash Latest: reviewing...")
        
        gemini_preview_result = self._review_with_retry(
            review_func=self._review_with_gemini_flash_preview,
            model_name="gemini-flash-latest",
            title=title,
            content=content,
            max_retries=3,
            retry_delay=30,
            progress_callback=progress_callback,
        )
        model_results.append(gemini_preview_result)
        
        if gemini_preview_result.error is None:
            if progress_callback:
                progress_callback(f"   ✓ Gemini Flash Latest: Score {gemini_preview_result.score:.1f}/10")
        
        # Calculate aggregate metrics
        valid_results = [r for r in model_results if r.error is None and r.score > 0]
        
        if not valid_results:
            logger.error("All model reviews failed!")
            return AggregatedReviewResult(
                average_score=0.0,
                individual_results=model_results,
                combined_feedback=["All review models failed. Please check API keys and connectivity."],
                can_apply_any_feedback=False,
            )
        
        # Calculate average score
        average_score = sum(r.score for r in valid_results) / len(valid_results)
        
        # Combine feedback (deduplicate similar items)
        combined_feedback: List[str] = []
        seen_feedback: set = set()
        for result in valid_results:
            for fb in result.feedback:
                fb_lower = fb.lower().strip()
                if fb_lower not in seen_feedback:
                    combined_feedback.append(fb)
                    seen_feedback.add(fb_lower)
        
        # Check if any feedback can be applied
        can_apply_any = any(r.can_apply_feedback for r in valid_results)
        
        if progress_callback:
            progress_callback(f"\n📊 Review Results:")
            for r in model_results:
                if r.error:
                    progress_callback(f"   ├─ {r.model_name}: ❌ Error - {r.error[:50]}...")
                else:
                    emoji = "✅" if r.score > 9 else "⚠️" if r.score >= 7 else "❌"
                    progress_callback(f"   ├─ {r.model_name}: {emoji} Score {r.score:.1f}/10")
            progress_callback(f"   └─ Average Score: {'✅' if average_score > 9 else '⚠️'} {average_score:.2f}/10")
        
        return AggregatedReviewResult(
            average_score=average_score,
            individual_results=model_results,
            combined_feedback=combined_feedback,
            can_apply_any_feedback=can_apply_any,
        )
    
    def regenerate_with_feedback(
        self,
        title: str,
        content: str,
        feedback: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Regenerate blog content incorporating the feedback.
        
        Args:
            title: Blog title
            content: Current blog content
            feedback: List of feedback items to address
            progress_callback: Optional callback for progress updates
            
        Returns:
            Improved blog content
        """
        import time
        from src.utils.llm_helpers import gemini_llm_call
        
        if progress_callback:
            progress_callback("🔄 Regenerating content with feedback...")
        
        feedback_text = "\n".join(f"- {fb}" for fb in feedback)
        
        prompt = f"""You are an expert blog writer. Revise the following blog post to address the feedback provided.

BLOG TITLE: {title}

CURRENT CONTENT:
{content}

FEEDBACK TO ADDRESS:
{feedback_text}

REQUIREMENTS:
1. Address ALL feedback items specifically
2. Maintain the overall structure and topic coverage
3. Keep the same markdown formatting
4. Improve clarity and conciseness
5. Ensure technical accuracy
6. Keep code examples functional and correct
7. Do NOT add placeholder comments like "existing content..." - write the full content

OUTPUT: Provide the complete revised blog post in markdown format. Start directly with the content (no preamble).
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            revised_content = gemini_llm_call(
                messages,
                model_name="gemini-2.5-flash",
                settings=self.settings,
            )
            
            if progress_callback:
                progress_callback("✅ Content regenerated successfully")
            
            return revised_content
            
        except Exception as e:
            logger.error(f"Content regeneration failed: {e}")
            if progress_callback:
                progress_callback(f"❌ Regeneration failed: {e}")
            return content  # Return original on failure
    
    async def review_and_improve(
        self,
        title: str,
        content: str,
        max_iterations: int = 5,
        score_threshold: float = 9.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, AggregatedReviewResult, List[ReviewIterationHistory]]:
        """
        Review and iteratively improve blog content until threshold is met.
        
        Args:
            title: Blog title
            content: Initial blog content
            max_iterations: Maximum improvement iterations (default: 5)
            score_threshold: Target score threshold (default: 9.0)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (final_content, final_review, iteration_history)
        """
        current_content = content
        iteration_history: List[ReviewIterationHistory] = []
        best_score = 0.0
        best_content = content
        best_review: Optional[AggregatedReviewResult] = None
        
        for iteration in range(1, max_iterations + 1):
            if progress_callback:
                progress_callback(f"\n{'='*60}")
                progress_callback(f"📝 REVIEW ITERATION {iteration}/{max_iterations}")
                progress_callback(f"{'='*60}")
            
            # Review with all models
            review_result = await self.review_with_all_models(
                title,
                current_content,
                progress_callback,
            )
            
            # Store in history
            iteration_history.append(ReviewIterationHistory(
                iteration=iteration,
                content_version=current_content[:500] + "...",  # Store preview
                review_result=review_result,
            ))
            
            # Track best version
            if review_result.average_score > best_score:
                best_score = review_result.average_score
                best_content = current_content
                best_review = review_result
            
            # Check if threshold met
            if review_result.average_score > score_threshold:
                if progress_callback:
                    progress_callback(f"\n🎉 SUCCESS! Score {review_result.average_score:.2f} > {score_threshold}")
                    progress_callback(f"   Blog approved after {iteration} iteration(s)")
                return current_content, review_result, iteration_history
            
            # Check if max iterations reached
            if iteration >= max_iterations:
                if progress_callback:
                    progress_callback(f"\n⚠️  Max iterations ({max_iterations}) reached")
                    progress_callback(f"   Final score: {review_result.average_score:.2f}")
                    progress_callback(f"   Best score achieved: {best_score:.2f}")
                break
            
            # Check if feedback can be applied
            if not review_result.can_apply_any_feedback:
                if progress_callback:
                    progress_callback("\n⚠️  Feedback requires external research - cannot auto-apply")
                    progress_callback("   Proceeding with current content")
                break
            
            # Regenerate with feedback
            if progress_callback:
                progress_callback(f"\n📋 Applying {len(review_result.combined_feedback)} feedback items...")
            
            current_content = self.regenerate_with_feedback(
                title,
                current_content,
                review_result.combined_feedback,
                progress_callback,
            )
        
        # Return best version if threshold not met
        if best_review is None:
            best_review = review_result
        
        if progress_callback:
            progress_callback(f"\n📌 Using best version (Score: {best_score:.2f})")
        
        return best_content, best_review, iteration_history
    
    def save_review_history(
        self,
        history: List[ReviewIterationHistory],
        output_path: Path,
    ) -> None:
        """Save review iteration history to YAML file."""
        history_data = {
            "total_iterations": len(history),
            "iterations": [],
        }
        
        for item in history:
            iteration_data = {
                "iteration": item.iteration,
                "average_score": item.review_result.average_score,
                "passed_threshold": item.review_result.passes_threshold,
                "models": [],
                "combined_feedback": item.review_result.combined_feedback,
                "can_apply_feedback": item.review_result.can_apply_any_feedback,
            }
            
            for model_result in item.review_result.individual_results:
                iteration_data["models"].append({
                    "name": model_result.model_name,
                    "score": model_result.score,
                    "feedback_count": len(model_result.feedback),
                    "can_apply_feedback": model_result.can_apply_feedback,
                    "error": model_result.error,
                })
            
            history_data["iterations"].append(iteration_data)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(history_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Review history saved to: {output_path}")
