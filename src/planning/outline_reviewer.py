"""
Outline Reviewer module for evaluating and improving blog outlines.

Uses LLM to review generated outlines and iteratively improve them until
they meet quality standards.
"""

import asyncio
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.planning.outline_generator import BlogOutline, OutlineGenerator
from src.research.content_extractor import AggregatedExtractedContent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlineReview(BaseModel):
    """Review result for a blog outline."""

    score: float = Field(description="Overall quality score from 0.0-10.0", ge=0.0, le=10.0)
    completeness_score: float = Field(
        description="Does it cover the topic thoroughly? (0.0-10.0)", ge=0.0, le=10.0
    )
    logical_flow_score: float = Field(
        description="Is the section order sensible? (0.0-10.0)", ge=0.0, le=10.0
    )
    depth_score: float = Field(
        description="Are subtopics specific enough? (0.0-10.0)", ge=0.0, le=10.0
    )
    balance_score: float = Field(
        description="Good mix of theory, code, visuals? (0.0-10.0)", ge=0.0, le=10.0
    )
    audience_fit_score: float = Field(
        description="Appropriate for target audience? (0.0-10.0)", ge=0.0, le=10.0
    )
    strengths: list[str] = Field(
        description="Key strengths of the outline (2-3 points)"
    )
    weaknesses: list[str] = Field(
        description="Areas that need improvement (2-3 points)"
    )
    specific_feedback: str = Field(
        description="Detailed, actionable feedback for improving the outline"
    )


class OutlineReviewer:
    """
    Reviews blog outlines using LLM and iteratively improves them.
    """

    SYSTEM_PROMPT = """
You are an expert technical content reviewer and editor.

Your task is to CRITICALLY EVALUATE a blog post outline and provide HONEST, ACTIONABLE feedback.

EVALUATION CRITERIA (Score each 1-10):

1. **Completeness** (1-10):
   - Does the outline cover all essential aspects of the topic?
   - Are there obvious gaps or missing sections?
   - Does it address common questions readers would have?

2. **Logical Flow** (1-10):
   - Does the section order make sense?
   - Does each section build on previous ones?
   - Is there a clear progression from introduction to conclusion?

3. **Depth** (1-10):
   - Are section descriptions specific and detailed?
   - Do summaries clearly explain WHAT and WHY?
   - Are code/diagram instructions concrete?

4. **Balance** (1-10):
   - Good mix of conceptual explanations and practical examples?
   - Appropriate amount of code examples and diagrams?
   - Sections are roughly balanced in size/scope?

5. **Audience Fit** (1-10):
   - Is the difficulty level appropriate for the target audience?
   - Is the technical depth suitable?
   - Will the target audience find this useful?

SCORING GUIDELINES (Use decimal precision for nuanced evaluation):
- 9.0-10.0: Exceptional, publication-ready
- 7.5-8.9: Good, minor improvements needed
- 6.0-7.4: Acceptable, but significant improvements needed
- 4.0-5.9: Poor, major rework required
- 0.0-3.9: Fundamentally flawed

BE CRITICAL: If the outline has issues, score it lower. Don't inflate scores.
Use decimal scores (e.g., 8.5, 7.2) to provide nuanced feedback.

FEEDBACK REQUIREMENTS:
- Identify 2-3 key strengths
- Identify 2-3 key weaknesses
- Provide specific, actionable improvement suggestions
- If score < 8, your feedback will be used to regenerate the outline

OUTPUT:
Generate a structured JSON response matching the `OutlineReview` schema.
"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the outline reviewer.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        self.settings = get_settings()
        self.model_name = model_name
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy initialization of LLM with structured output."""
        if self._llm is None:
            if not self.settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for outline review")

            base_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.settings.google_api_key,
                temperature=0.3,  # Lower temp for more consistent scoring
            )
            self._llm = base_llm.with_structured_output(OutlineReview)
            logger.debug(f"Initialized reviewer LLM with model: {self.model_name}")
        return self._llm

    async def review(self, outline: BlogOutline) -> OutlineReview:
        """
        Review a blog outline and provide scores and feedback.

        Args:
            outline: The blog outline to review

        Returns:
            OutlineReview with scores and feedback
        """
        logger.info(f"Reviewing outline for: {outline.topic}")

        # Format outline for review
        outline_text = f"Topic: {outline.topic}\n\n"
        outline_text += f"Target Audience: {outline.metadata.target_audience}\n"
        outline_text += f"Difficulty: {outline.metadata.difficulty}\n"
        outline_text += f"Estimated Reading Time: {outline.metadata.estimated_reading_time}\n\n"
        outline_text += "Sections:\n"

        for i, section in enumerate(outline.sections, 1):
            outline_text += f"\n{i}. {section.heading}\n"
            outline_text += f"   Summary: {section.summary}\n"
            outline_text += f"   References: {len(section.references)} URLs\n"

        prompt = f"Review the following blog post outline:\n\n{outline_text}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result: OutlineReview = await self.llm.ainvoke(messages)
            logger.info(
                f"Review complete - Score: {result.score}/10 "
                f"(C:{result.completeness_score} F:{result.logical_flow_score} "
                f"D:{result.depth_score} B:{result.balance_score} A:{result.audience_fit_score})"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to review outline: {e}")
            raise

    def display_review(self, review: OutlineReview, iteration: int, console) -> None:
        """
        Display review feedback in console.

        Args:
            review: The review to display
            iteration: Current iteration number
            console: Rich console instance
        """
        from rich.panel import Panel
        from rich.table import Table

        # Create feedback panel
        feedback_text = f"[bold cyan]Iteration {iteration} Review[/bold cyan]\n\n"
        feedback_text += f"[bold]Overall Score:[/bold] {review.score:.1f}/10.0\n\n"
        
        # Score breakdown table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Criteria", style="cyan")
        table.add_column("Score", justify="right", style="yellow")
        
        table.add_row("Completeness", f"{review.completeness_score:.1f}/10")
        table.add_row("Logical Flow", f"{review.logical_flow_score:.1f}/10")
        table.add_row("Depth", f"{review.depth_score:.1f}/10")
        table.add_row("Balance", f"{review.balance_score:.1f}/10")
        table.add_row("Audience Fit", f"{review.audience_fit_score:.1f}/10")
        
        console.print(table)
        
        # Strengths and weaknesses
        console.print(f"\n[bold green]Strengths:[/bold green]")
        for strength in review.strengths:
            console.print(f"  ✓ {strength}")
        
        console.print(f"\n[bold red]Weaknesses:[/bold red]")
        for weakness in review.weaknesses:
            console.print(f"  ✗ {weakness}")
        
        console.print(f"\n[bold]Improvement Feedback:[/bold]")
        console.print(f"{review.specific_feedback}\n")

    async def save_reviews(
        self,
        reviews: list[OutlineReview],
        file_path,
    ) -> None:
        """
        Save all reviews to a YAML file.

        Args:
            reviews: List of reviews from all iterations
            file_path: Path to save reviews YAML
        """
        from datetime import datetime
        import yaml

        reviews_data = {
            "reviewed_at": datetime.now().isoformat(),
            "total_iterations": len(reviews),
            "final_score": reviews[-1].score,
            "iterations": []
        }

        for i, review in enumerate(reviews, 1):
            iteration_data = {
                "iteration": i,
                "overall_score": float(review.score),
                "scores": {
                    "completeness": float(review.completeness_score),
                    "logical_flow": float(review.logical_flow_score),
                    "depth": float(review.depth_score),
                    "balance": float(review.balance_score),
                    "audience_fit": float(review.audience_fit_score),
                },
                "strengths": review.strengths,
                "weaknesses": review.weaknesses,
                "specific_feedback": review.specific_feedback,
            }
            reviews_data["iterations"].append(iteration_data)

        # Save to YAML
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(reviews_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Saved {len(reviews)} review(s) to {file_path}")

    async def review_and_iterate(
        self,
        topic: str,
        research_data: AggregatedExtractedContent,
        generator: OutlineGenerator,
        max_iterations: Optional[int] = None,
        quality_threshold: Optional[int] = None,
    ) -> tuple[BlogOutline, list[OutlineReview]]:
        """
        Review outline and iteratively improve it until quality threshold is met.

        Args:
            topic: Blog topic
            research_data: Research data for generating outline
            generator: OutlineGenerator instance
            max_iterations: Max iterations (default: from settings)
            quality_threshold: Quality score threshold (default: from settings)

        Returns:
            Tuple of (final outline, list of all reviews)
        """
        max_iterations = max_iterations or self.settings.max_outline_iterations
        quality_threshold = quality_threshold or self.settings.outline_quality_threshold

        logger.info(
            f"Starting outline review iteration (max: {max_iterations}, threshold: {quality_threshold})"
        )

        reviews: list[OutlineReview] = []
        current_outline = await generator.generate(topic, research_data)

        for iteration in range(1, max_iterations + 1):
            logger.info(f"Iteration {iteration}/{max_iterations}")

            # Review current outline
            review = await self.review(current_outline)
            reviews.append(review)

            if review.score >= quality_threshold:
                logger.info(
                    f"✓ Outline approved! Score: {review.score}/{quality_threshold}"
                )
                return current_outline, reviews

            # If not last iteration, regenerate with feedback
            if iteration < max_iterations:
                logger.info(
                    f"Score {review.score} < {quality_threshold}. Regenerating with feedback..."
                )
                logger.debug(f"Feedback: {review.specific_feedback}")

                # Regenerate with feedback
                current_outline = await self._regenerate_with_feedback(
                    topic, research_data, generator, review
                )
            else:
                logger.warning(
                    f"Max iterations reached. Final score: {review.score}/{quality_threshold}"
                )

        return current_outline, reviews

    async def _regenerate_with_feedback(
        self,
        topic: str,
        research_data: AggregatedExtractedContent,
        generator: OutlineGenerator,
        review: OutlineReview,
    ) -> BlogOutline:
        """
        Regenerate outline with feedback from review.

        Args:
            topic: Blog topic
            research_data: Research data
            generator: OutlineGenerator instance
            review: Review with feedback

        Returns:
            Improved outline
        """
        # Convert review to dict for the generator
        feedback_dict = {
            "score": review.score,
            "strengths": review.strengths,
            "weaknesses": review.weaknesses,
            "specific_feedback": review.specific_feedback,
        }
        
        logger.debug("Regenerating outline with feedback")
        return await generator.generate(topic, research_data, feedback=feedback_dict)

    def review_sync(self, outline: BlogOutline) -> OutlineReview:
        """Synchronous wrapper for review()."""
        return asyncio.run(self.review(outline))

    def review_and_iterate_sync(
        self,
        topic: str,
        research_data: AggregatedExtractedContent,
        generator: OutlineGenerator,
        max_iterations: Optional[int] = None,
        quality_threshold: Optional[int] = None,
    ) -> tuple[BlogOutline, list[OutlineReview]]:
        """Synchronous wrapper for review_and_iterate()."""
        return asyncio.run(
            self.review_and_iterate(
                topic, research_data, generator, max_iterations, quality_threshold
            )
        )
