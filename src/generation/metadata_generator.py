"""
Metadata generator for blog posts.

Generates SEO-optimized titles, tags, and search descriptions.
Includes review loop to ensure quality.
"""

import yaml
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call

logger = get_logger(__name__)


class BlogMetadata(BaseModel):
    """Structured output for blog metadata."""
    titles: List[str] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 unique, SEO-optimized blog titles"
    )
    tags: List[str] = Field(
        min_length=3,
        max_length=4,
        description="3-4 relevant tags for the blog post"
    )
    search_description: str = Field(
        max_length=160,
        description="Short SEO meta description (max 160 characters)"
    )


class MetadataReview(BaseModel):
    """Structured output for metadata review."""
    score: float = Field(
        ge=1.0, le=10.0,
        description="Quality score from 1-10 for the metadata"
    )
    feedback: List[str] = Field(
        description="List of specific improvements needed"
    )


@dataclass
class MetadataResult:
    """Final metadata result after review."""
    titles: List[str]
    tags: List[str]
    search_description: str
    selected_title: Optional[str] = None
    review_score: float = 0.0
    iterations: int = 0


class MetadataGenerator:
    """
    Generates and reviews blog metadata (titles, tags, description).
    
    Uses Gemini 2.5 Flash for generation and review.
    Implements iterative improvement with max 3 attempts.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model_name = "gemini-2.5-flash"
        logger.info("MetadataGenerator initialized")
    
    def _generate_metadata(
        self,
        topic: str,
        content: str,
        feedback: Optional[List[str]] = None,
    ) -> BlogMetadata:
        """Generate blog metadata using LLM."""
        
        feedback_section = ""
        if feedback:
            feedback_text = "\n".join(f"- {fb}" for fb in feedback)
            feedback_section = f"""
PREVIOUS FEEDBACK TO ADDRESS:
{feedback_text}

Please improve the metadata based on this feedback.
"""
        
        prompt = f"""You are an SEO expert. Generate metadata for the following blog post.

BLOG TOPIC: {topic}

BLOG CONTENT (first 2000 chars):
{content[:2000]}...

{feedback_section}

REQUIREMENTS:

1. **TITLES** (exactly 5):
   - Each title should be unique and compelling
   - SEO-optimized with relevant keywords
   - 50-60 characters ideal length
   - Clear, descriptive, not clickbait
   - Different angles/approaches for each title

2. **TAGS** (3-4):
   - Relevant keywords for categorization
   - Single words or short phrases
   - Help with discoverability

3. **SEARCH DESCRIPTION**:
   - Maximum 155 characters
   - Compelling summary that encourages clicks
   - Include primary keyword naturally
   - Action-oriented when possible

Return JSON format:
{{
    "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
    "tags": ["tag1", "tag2", "tag3"],
    "search_description": "Short compelling description under 155 chars"
}}
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            response = gemini_llm_call(
                messages,
                model_name=self.model_name,
                settings=self.settings,
            )
            
            # Parse JSON response
            import json
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Ensure exactly 5 titles
            titles = data.get("titles", [])[:5]
            while len(titles) < 5:
                titles.append(f"{topic} - Variation {len(titles) + 1}")
            
            # Ensure 3-4 tags
            tags = data.get("tags", [])[:4]
            if len(tags) < 3:
                tags.extend(["technology", "guide", "tutorial"][:3 - len(tags)])
            
            # Ensure description is under 160 chars
            description = data.get("search_description", f"Learn about {topic}")[:155]
            
            return BlogMetadata(
                titles=titles,
                tags=tags,
                search_description=description,
            )
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            # Return fallback metadata
            return BlogMetadata(
                titles=[
                    f"{topic}",
                    f"Guide to {topic}",
                    f"Understanding {topic}",
                    f"Complete {topic} Tutorial",
                    f"Getting Started with {topic}",
                ],
                tags=["technology", "guide", "tutorial"],
                search_description=f"Learn everything about {topic} in this comprehensive guide."[:155],
            )
    
    def _review_metadata(
        self,
        metadata: BlogMetadata,
        topic: str,
    ) -> MetadataReview:
        """Review metadata quality using LLM."""
        
        titles_text = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(metadata.titles))
        tags_text = ", ".join(metadata.tags)
        
        prompt = f"""You are an SEO expert reviewing blog metadata quality.

BLOG TOPIC: {topic}

GENERATED METADATA:

Titles:
{titles_text}

Tags: {tags_text}

Search Description: {metadata.search_description}

EVALUATION CRITERIA:

1. **TITLES (40%)**:
   - Are they unique and compelling?
   - SEO-optimized with keywords?
   - Appropriate length (50-60 chars)?
   - Clear and descriptive?
   - Different angles covered?

2. **TAGS (30%)**:
   - Relevant to the content?
   - Good for categorization?
   - Right number (3-4)?

3. **SEARCH DESCRIPTION (30%)**:
   - Under 155 characters?
   - Compelling and click-worthy?
   - Contains primary keyword?
   - Accurately summarizes content?

SCORING:
- 9-10: Excellent, publication-ready
- 7-8: Good, minor improvements
- 5-6: Acceptable, needs work
- 1-4: Poor, major rework needed

Return JSON:
{{
    "score": <float 1-10>,
    "feedback": ["improvement 1", "improvement 2", ...]
}}
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            response = gemini_llm_call(
                messages,
                model_name=self.model_name,
                settings=self.settings,
            )
            
            import json
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            return MetadataReview(
                score=float(data.get("score", 5.0)),
                feedback=data.get("feedback", []),
            )
            
        except Exception as e:
            logger.warning(f"Metadata review failed: {e}")
            return MetadataReview(score=5.0, feedback=[str(e)])
    
    def generate_with_review(
        self,
        topic: str,
        content: str,
        max_iterations: int = 3,
        score_threshold: float = 9.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> MetadataResult:
        """
        Generate metadata with iterative review and improvement.
        
        Args:
            topic: Blog topic
            content: Blog content
            max_iterations: Max improvement attempts (default: 3)
            score_threshold: Target score (default: 9.0)
            progress_callback: Optional progress callback
            
        Returns:
            MetadataResult with titles, tags, description
        """
        if progress_callback:
            progress_callback("\n📝 Generating blog metadata (titles, tags, description)...")
        
        best_metadata: Optional[BlogMetadata] = None
        best_score = 0.0
        feedback: Optional[List[str]] = None
        
        for iteration in range(1, max_iterations + 1):
            if progress_callback:
                progress_callback(f"\n   [Iteration {iteration}/{max_iterations}] Generating metadata...")
            
            # Generate metadata
            metadata = self._generate_metadata(topic, content, feedback)
            
            if progress_callback:
                progress_callback(f"   [Iteration {iteration}/{max_iterations}] Reviewing metadata...")
            
            # Review metadata
            review = self._review_metadata(metadata, topic)
            
            # Display results
            score_emoji = "✅" if review.score >= score_threshold else "⚠️"
            if progress_callback:
                progress_callback(f"   {score_emoji} [Iteration {iteration}/{max_iterations}] Score: {review.score:.1f}/10")
            
            # Track best version
            if review.score > best_score:
                best_score = review.score
                best_metadata = metadata
            
            # Check if threshold met
            if review.score >= score_threshold:
                if progress_callback:
                    progress_callback(f"   ✓ Metadata quality threshold met!")
                break
            
            # Update feedback for next iteration
            feedback = review.feedback
            
            if iteration < max_iterations and progress_callback:
                progress_callback(f"   Feedback: {', '.join(feedback[:2])}...")
        
        if best_metadata is None:
            best_metadata = metadata
        
        if progress_callback:
            progress_callback(f"\n   Generated {len(best_metadata.titles)} titles, {len(best_metadata.tags)} tags")
            progress_callback(f"   Description: {best_metadata.search_description[:50]}...")
        
        return MetadataResult(
            titles=best_metadata.titles,
            tags=best_metadata.tags,
            search_description=best_metadata.search_description,
            review_score=best_score,
            iterations=iteration,
        )
    
    def prompt_title_selection(
        self,
        titles: List[str],
        console=None,
    ) -> str:
        """
        Prompt user to select a title from the list.
        
        Args:
            titles: List of 5 titles to choose from
            console: Rich console for formatted output (optional)
            
        Returns:
            Selected title string
        """
        if console:
            from rich.table import Table
            from rich.prompt import Prompt
            
            console.print("\n[bold cyan]📋 Select a title for your blog:[/bold cyan]\n")
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("#", style="dim", width=3)
            table.add_column("Title", style="cyan")
            table.add_column("Length", style="dim", width=8)
            
            for i, title in enumerate(titles, 1):
                table.add_row(str(i), title, f"{len(title)} chars")
            
            console.print(table)
            console.print()
            
            while True:
                choice = Prompt.ask(
                    "Enter title number (1-5)",
                    choices=["1", "2", "3", "4", "5"],
                    default="1",
                )
                return titles[int(choice) - 1]
        else:
            # Fallback to simple input
            print("\nSelect a title for your blog:")
            for i, title in enumerate(titles, 1):
                print(f"  {i}. {title}")
            
            while True:
                try:
                    choice = input("\nEnter title number (1-5): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(titles):
                        return titles[idx]
                    print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
    
    def save_metadata(
        self,
        result: MetadataResult,
        output_path: Path,
    ) -> None:
        """
        Save metadata to YAML file.
        
        Args:
            result: MetadataResult with all metadata
            output_path: Path to metadata.yaml
        """
        metadata_dict = {
            "title": result.selected_title or result.titles[0],
            "all_titles": result.titles,
            "tags": result.tags,
            "search_description": result.search_description,
            "review_score": result.review_score,
            "iterations": result.iterations,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Metadata saved to: {output_path}")
