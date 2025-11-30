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
    """Review result for a blog outline based on E-E-A-T principles."""

    score: float = Field(description="Overall quality score from 0.0-10.0", ge=0.0, le=10.0)
    completeness_score: float = Field(
        description="Does it cover the topic thoroughly with E-E-A-T elements (experience, expertise)? (0.0-10.0)", ge=0.0, le=10.0
    )
    logical_flow_score: float = Field(
        description="Is the section order sensible with clear structure? (0.0-10.0)", ge=0.0, le=10.0
    )
    depth_score: float = Field(
        description="Are subtopics specific with original insights beyond obvious information? (0.0-10.0)", ge=0.0, le=10.0
    )
    balance_score: float = Field(
        description="Good mix of theory, code, visuals, and actionable content? (0.0-10.0)", ge=0.0, le=10.0
    )
    audience_fit_score: float = Field(
        description="Appropriate for target audience with people-first approach? (0.0-10.0)", ge=0.0, le=10.0
    )
    strengths: list[str] = Field(
        description="Key strengths of the outline including E-E-A-T elements (2-3 points)"
    )
    weaknesses: list[str] = Field(
        description="Areas that need improvement including E-E-A-T gaps (2-3 points)"
    )
    specific_feedback: str = Field(
        description="Detailed, actionable feedback for improving the outline with E-E-A-T focus"
    )


class OutlineReviewer:
    """
    Reviews blog outlines using LLM and iteratively improves them.
    """

    SYSTEM_PROMPT = """
You are an expert technical content reviewer and editor evaluating blog outlines based on Google's Search Quality Guidelines and E-E-A-T principles.

Your task is to CRITICALLY EVALUATE a blog post outline and provide HONEST, ACTIONABLE feedback.

EVALUATION CRITERIA (Score each 0.0-10.0):

1. **Completeness & E-E-A-T Alignment** (0.0-10.0):
   - Does the outline cover all essential aspects of the topic?
   - Are there obvious gaps or missing sections?
   - Does it address common questions readers would have?
   - **E-E-A-T**: Does the outline plan demonstrate:
     * First-hand experience (practical examples, case studies)?
     * Deep expertise (technical depth, evidence-based claims)?
     * Authoritative sources and references?
     * Trustworthy, factual information?
   - Will readers feel they've learned enough to achieve their goals?
   - Is this outline for people-first content, not search-engine-first?

2. **Logical Flow & Structure** (0.0-10.0):
   - Does the section order make sense?
   - Does each section build on previous ones?
   - Is there a clear progression from introduction to conclusion?
   - Are headings descriptive and not clickbait?
   - Does the structure facilitate easy scanning and readability?

3. **Depth & Originality** (0.0-10.0):
   - Are section descriptions specific and detailed?
   - Do summaries clearly explain WHAT and WHY?
   - Are code/diagram instructions concrete?
   - Does the outline promise original insights beyond obvious information?
   - Will content add substantial value vs. just rehashing existing sources?
   - Does it avoid promising answers to unanswerable questions?

4. **Balance & Content Mix** (0.0-10.0):
   - Good mix of conceptual explanations and practical examples?
   - Appropriate amount of code examples and diagrams?
   - Sections are roughly balanced in size/scope?
   - Does it include actionable, useful information for readers?
   - Balance between theory and hands-on guidance?

5. **Audience Fit & Value** (0.0-10.0):
   - Is the difficulty level appropriate for the target audience?
   - Is the technical depth suitable?
   - Will the target audience find this useful if they came directly to the site?
   - Does it appear designed for an existing audience vs. just search traffic?
   - Would readers bookmark, share, or recommend this content?
   - Is this the kind of outline you'd expect in a printed magazine or encyclopedia?

SCORING GUIDELINES (Use decimal precision for nuanced evaluation):
- 9.0-10.0: Exceptional, demonstrates strong E-E-A-T planning, publication-ready
- 7.5-8.9: Good with minor improvements needed
- 6.0-7.4: Acceptable but needs significant improvements (lacking E-E-A-T or depth)
- 4.0-5.9: Poor, major rework required
- 0.0-3.9: Fundamentally flawed, appears search-engine-first

**BE CRITICAL**: If the outline has issues, score it lower. Don't inflate scores.
Use decimal scores (e.g., 8.5, 7.2) to provide nuanced feedback.

**CRITICAL E-E-A-T REQUIREMENTS**:
- If outline lacks evidence of first-hand experience/expertise, score Completeness < 7.0
- If outline appears designed for search engines rather than people, score < 6.0
- If sections are too generic or lack specificity, score Depth < 7.0
- If outline doesn't plan for substantial original value, score Depth < 7.0

FEEDBACK REQUIREMENTS:
- Identify 2-3 key strengths (especially E-E-A-T elements)
- Identify 2-3 key weaknesses (especially E-E-A-T gaps)
- Provide specific, actionable improvement suggestions
- If score < 8, your feedback will be used to regenerate the outline
- Explicitly mention E-E-A-T aspects in your feedback

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
