import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call

logger = get_logger(__name__)


class SectionReview(BaseModel):
    """Structured output model for section content review."""
    score: float = Field(
        description="Quality score from 1-10, where 10 is perfect. Consider: Quality (35%), Conciseness (35%), Structure (10%), Readability (10%), SEO (10%)"
    )
    feedback: str = Field(
        description="Detailed, actionable feedback on what needs improvement, especially regarding quality and conciseness. Use bullet points."
    )


class ContentGenerator:
    def __init__(self):
        self.settings = get_settings()

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_research_content(self, references: List[str], research_data: Dict[str, Any]) -> str:
        """
        Retrieves content for the top 3 references from the research data.
        """
        content_parts = []
        found_count = 0
        
        # Create a lookup map for faster access
        url_to_content = {item['url']: item for item in research_data.get('contents', [])}
        
        for url in references:
            if found_count >= 3:
                break
            
            if url in url_to_content:
                item = url_to_content[url]
                # Use markdown content if available, otherwise snippet
                text = item.get('markdown', item.get('snippet', ''))
                # Truncate if too long to avoid context window issues (though Gemini has large context)
                # Let's limit to ~10k chars per source to be safe and efficient
                if len(text) > 10000:
                    text = text[:10000] + "...(truncated)"
                
                content_parts.append(f"Source: {url}\nContent:\n{text}\n")
                found_count += 1
                
        return "\n---\n".join(content_parts)

    def _generate_section_content(self, 
                                section: Dict[str, Any], 
                                research_content: str, 
                                previous_context: str,
                                topic: str,
                                feedback: Optional[str] = None,
                                existing_content: Optional[str] = None) -> str:
        heading = section.get('heading', '')
        summary = section.get('summary', '')
        
        if feedback and existing_content:
            # Regeneration mode with feedback
            prompt = f"""
You are an expert technical blog writer revising a section for a blog post titled "{topic}".

Current Subtopic: {heading}
Subtopic Summary: {summary}

EXISTING CONTENT TO REVISE:
{existing_content}

REVIEW FEEDBACK:
{feedback}

Research Content (from top references):
{research_content}

Context from previous sections:
{previous_context if previous_context else "This is the first section."}

CRITICAL REQUIREMENTS:

1. REVISE the existing content based on the feedback above, focusing especially on QUALITY and CONCISENESS.

2. MAINTAIN the structure and flow while addressing all feedback points.

3. Keep all the other requirements below:
"""
        else:
            # Initial generation mode
            prompt = f"""
You are an expert technical blog writer. You are writing a section for a blog post titled "{topic}".

Current Subtopic: {heading}
Subtopic Summary: {summary}

Research Content (from top references):
{research_content}

Context from previous sections:
{previous_context if previous_context else "This is the first section."}

CRITICAL REQUIREMENTS:

1. CONCISENESS: Write concise, to-the-point content for this subtopic only. This is ONE subsection of the blog, not the entire article.

2. FORMATTING: Use bullet points liberally to improve readability. Avoid long, dense paragraphs. Break complex ideas into digestible points.

3. REFERENCES: Include relevant links to the provided references (MAXIMUM 2 links per section). Format: [descriptive text](URL). Only link to URLs from the research content provided above.

4. FLOW & COHESION: Ensure smooth transitions from the previous sections. The content must feel like a natural continuation of the blog, maintaining a cohesive narrative throughout.

5. SEO OPTIMIZATION: Use relevant keywords naturally. Include the main topic and subtopic keywords where appropriate without keyword stuffing.

6. CODE SNIPPETS: If the subtopic involves technical implementation or examples, generate well-commented code snippets using triple backticks with language specification (e.g., ```python, ```javascript, etc.).

7. DIAGRAMS: If the subtopic involves processes, workflows, or system architecture, create Mermaid diagrams using triple backticks (e.g., ```mermaid). Use flowcharts, sequence diagrams, or other appropriate diagram types.

8. ACCESSIBILITY: Write in simple, clear language that anyone can understand, regardless of their technical background. Explain jargon when necessary.

9. CONTEXT ACCURACY: Base ALL content strictly on the provided research context above. Do not invent facts or add information not supported by the research.

10. SCOPE CONTROL: Remember this is just ONE subsection. Keep it focused and appropriately sized - not too long, not too short. Do not write conclusions for the entire blog post.

Generate the content for this section now.
"""
        messages = [HumanMessage(content=prompt)]
        return gemini_llm_call(messages, model_name="gemini-2.5-flash", settings=self.settings)
    
    def _review_section_content(self,
                              topic: str,
                              heading: str,
                              content: str) -> Dict[str, Any]:
        """
        Reviews section content and provides a quality score and feedback.
        Prioritizes quality and conciseness using structured output.
        
        Returns:
            Dict with 'score' (float) and 'feedback' (str)
        """
        prompt = f"""
You are an expert content reviewer evaluating a blog section for quality.

Blog Title: {topic}
Section Heading: {heading}

CONTENT TO REVIEW:
{content}

EVALUATION CRITERIA (prioritize quality and conciseness):

1. **QUALITY (Weight: 35%)**: Is the content accurate, insightful, and valuable? Does it provide actionable information?

2. **CONCISENESS (Weight: 35%)**: Is the content concise and to-the-point? No unnecessary verbosity or repetition?

3. **STRUCTURE (Weight: 10%)**: Is the content well-organized with clear flow and logical progression?

4. **READABILITY (Weight: 10%)**: Is it easy to read with appropriate use of bullet points, short paragraphs, and clear language?

5. **SEO (Weight: 10%)**: Does it naturally incorporate relevant keywords without stuffing?

Provide a score (1-10) and specific, actionable feedback focusing on quality and conciseness improvements.
"""
        
        # Use gemini_llm_call with structured output
        review_model = self.settings.section_reviewer_model
        messages = [HumanMessage(content=prompt)]
        
        try:
            review = gemini_llm_call(
                messages,
                model_name=review_model,
                settings=self.settings,
                structured_output=SectionReview
            )
            return {
                'score': review.score,
                'feedback': review.feedback
            }
        except Exception as e:
            logger.warning(f"Structured output failed: {e}. Using fallback.")
            return {
                'score': 5.0,
                'feedback': f"Review failed: {str(e)}"
            }
    
    def _generate_section_with_review(self,
                                     section: Dict[str, Any],
                                     research_content: str,
                                     previous_context: str,
                                     topic: str,
                                     progress_callback: Optional[Any] = None) -> str:
        """
        Generates section content with iterative review and improvement.
        Continues up to max_section_review_iterations until quality threshold is met.
        Returns the best version if threshold is never reached.
        """
        max_iterations = self.settings.max_section_review_iterations
        threshold = self.settings.section_quality_threshold
        heading = section.get('heading', '')
        
        # Log section heading prominently
        msg = f"\n{'='*80}\n📝 SECTION: {heading}\n{'='*80}"
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        
        best_content = None
        best_score = 0.0
        
        for iteration in range(1, max_iterations + 1):
            msg = f"  [Iteration {iteration}/{max_iterations}] Generating content..."
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
            
            # Generate content (with feedback if not first iteration)
            if iteration == 1:
                content = self._generate_section_content(
                    section,
                    research_content,
                    previous_context,
                    topic
                )
            else:
                content = self._generate_section_content(
                    section,
                    research_content,
                    previous_context,
                    topic,
                    feedback=feedback,
                    existing_content=best_content
                )
            
            # Review the content
            msg = f"  [Iteration {iteration}/{max_iterations}] Reviewing content..."
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
            
            review_result = self._review_section_content(topic, heading, content)
            score = review_result['score']
            feedback = review_result['feedback']
            
            # Log score prominently with visual indicator
            score_emoji = "✅" if score > threshold else "⚠️"
            msg = f"  {score_emoji} [Iteration {iteration}/{max_iterations}] SCORE: {score:.1f}/10 (threshold: {threshold})"
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
            
            # Track best version
            if score > best_score:
                best_score = score
                best_content = content
            
            # Check if threshold met
            if score > threshold:
                msg = f"  ✓ Quality threshold met (score: {score:.1f} > {threshold})!"
                logger.info(msg)
                if progress_callback:
                    progress_callback(msg)
                return content
        
        # Threshold not met after all iterations, return best version
        msg = f"  Using best version (score: {best_score:.1f} after {max_iterations} iterations)"
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        
        return best_content

    def generate_blog_post(self, outline_path: Path, research_path: Path, progress_callback: Optional[Any] = None) -> str:
        """
        Generates the full blog post by iterating through the outline.
        """
        outline_data = self._load_yaml(outline_path)
        research_data = self._load_yaml(research_path)
        
        topic = outline_data.get('topic', 'Untitled Blog')
        sections = outline_data.get('sections', [])
        
        full_content = f"# {topic}\n\n"
        previous_context = ""
        
        total_sections = len(sections)
        logger.info(f"Starting content generation for topic: {topic} ({total_sections} sections)")
        
        for i, section in enumerate(sections):
            heading = section.get('heading', '')
            msg = f"Generating section {i+1}/{total_sections}: {heading}"
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
            
            references = section.get('references', [])
            research_content = self._get_research_content(references, research_data)
            
            # Generate section with iterative review
            section_content = self._generate_section_with_review(
                section,
                research_content,
                previous_context,
                topic,
                progress_callback
            )
            
            # Append to full content
            full_content += f"## {heading}\n\n{section_content}\n\n"
            
            # Update context
            previous_context += f"\nSummary of {heading}:\n{section.get('summary', '')}\nGenerated Content:\n{section_content}\n"
            
            # Rate limit wait (except for the last one)
            if i < total_sections - 1:
                msg = "Waiting 20 seconds to respect rate limits..."
                logger.info(msg)
                if progress_callback:
                    progress_callback(msg)
                time.sleep(20)
                
        return full_content

    def save_content(self, content: str, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved generated content to {output_path}")
