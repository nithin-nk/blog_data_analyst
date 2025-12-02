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
        description="Quality score from 1-10, where 10 is perfect. Consider: Style Compliance (25%), E-E-A-T & People-First Content (35%), Content Quality & Completeness (25%), Structure & Readability (10%), SEO & Discoverability (5%)"
    )
    feedback: str = Field(
        description="Detailed, actionable feedback on what needs improvement. Focus on E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness), people-first content, conciseness, and style. Use bullet points."
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

CRITICAL REQUIREMENTS FOR REVISION:

1.  **ADDRESS FEEDBACK**: Incorporate the review feedback.
2.  **STRICT STYLE ADHERENCE**:
    *   **Short Sentences**: Use less words. Be specific.
    *   **Bullet Points**: Use them liberally.
    *   **Example Style**:
        ```
        Why Deploying AI Agents Is Hard
        Agent frameworks differ widely in required infrastructure.

        Latency varies from milliseconds to minutes.

        Real-time streaming is needed for modern AI UX.
        ```
3.  **NO CONVERSATIONAL FILLER**: Output ONLY the blog section content.
4.  **NO MERMAID DIAGRAMS**: Do NOT include any mermaid code blocks or diagram syntax. Diagrams are generated separately in another step.
"""
        else:
            # Initial generation mode
            prompt = f"""
You are an expert technical blog writer. Write a section for a blog post titled "{topic}".

Current Subtopic: {heading}
Subtopic Summary: {summary}

Research Content (from top references):
{research_content}

Context from previous sections:
{previous_context if previous_context else "This is the first section."}

WRITING STYLE - STRICTLY FOLLOW THIS FORMAT:

1.  **Short Sentences**: Use less words. Be specific. Avoid long, complex sentences.
2.  **Bullet Points**: Use bullet points liberally for lists, steps, and key facts.
3.  **Directness**: Get straight to the point. No fluff.

EXAMPLE STYLE (Follow this pattern):
```
This blog post explains how to design a production-ready, open-source architecture for AI agents using FastAPI, Celery, Redis, Kubernetes, KEDA, Prometheus, Grafana, LangFuse, and LangGraph.

Why Deploying AI Agents Is Hard
Agent frameworks differ widely in required infrastructure.

Latency varies from milliseconds to minutes depending on workflow complexity.

Real-time streaming is needed for modern AI UX.

REST-only patterns can’t handle long execution, retries, or async scheduling.

Scaling compute-heavy agents is fundamentally different from scaling API servers.
```

CRITICAL REQUIREMENTS:

1.  **NO CONVERSATIONAL FILLER**: Output ONLY the blog section content.
2.  **CONCISENESS**:
    *   Short paragraphs (1-2 sentences).
    *   Specific details.
3.  **FORMATTING**:
    *   Use `###` for sub-sections.
    *   Use bullet points.
4.  **CODE**:
    *   Include relevant code snippets (```python, etc.).
5.  **LINKS**:
    *   Max 2 relevant links per section from provided research.
6.  **SCOPE**:
    *   Focus ONLY on this subtopic.
7.  **NO MERMAID DIAGRAMS**: Do NOT include any mermaid code blocks or diagram syntax. Diagrams are generated separately in a dedicated step.

Generate the content now.
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
You are an expert content reviewer evaluating a blog section for quality and style based on Google's Search Quality Guidelines and E-E-A-T principles.

Blog Title: {topic}
Section Heading: {heading}

CONTENT TO REVIEW:
{content}

EVALUATION CRITERIA:

1. **STYLE COMPLIANCE (Weight: 25%)**:
   * Are sentences SHORT and SPECIFIC?
   * Are BULLET POINTS used effectively?
   * Does it avoid long, complex paragraphs?
   * Does it match the concise style:
     ```
     Why Deploying AI Agents Is Hard
     Agent frameworks differ widely in required infrastructure.
     Latency varies from milliseconds to minutes.
     ```

2. **E-E-A-T & PEOPLE-FIRST CONTENT (Weight: 35%)**:
   * **Experience**: Does it demonstrate first-hand expertise or practical knowledge?
   * **Expertise**: Is the depth of knowledge evident? Are claims backed by evidence?
   * **Authoritativeness**: Would this be referenced in a magazine or encyclopedia?
   * **Trustworthiness**: Are sources clear? No easily-verified factual errors?
   * **People-First**: Is this useful for readers coming directly, not just for search engines?
   * Does it provide original information, analysis, or insights beyond the obvious?
   * Would readers leave feeling they've learned enough to achieve their goal?

3. **CONTENT QUALITY & COMPLETENESS (Weight: 25%)**:
   * Does it provide substantial, complete description of the topic?
   * Is the content accurate, valuable, and actionable?
   * Does the heading provide a descriptive, helpful summary (not clickbait)?
   * Does it avoid simply copying other sources? Does it add substantial value?
   * Is this content you'd bookmark, share, or recommend?

4. **STRUCTURE & READABILITY (Weight: 10%)**:
   * Clear logical flow and easy to scan?
   * No spelling or stylistic issues?
   * Well-produced, not sloppy or hastily made?

5. **SEO & DISCOVERABILITY (Weight: 5%)**:
   * Natural keyword usage without keyword stuffing?
   * Does it answer the topic clearly and completely?

SCORING GUIDELINES:
* 9-10: Exceptional, demonstrates strong E-E-A-T, publication-ready
* 7-8: Good quality with minor improvements needed
* 5-6: Acceptable but needs significant improvements (lacking E-E-A-T or conciseness)
* 3-4: Poor, major rework required
* 1-2: Fundamentally flawed

Provide a score (1-10) and specific, actionable feedback.
**CRITICAL**: 
- If content lacks first-hand experience/expertise evidence, score below 7
- If content uses long sentences or lacks bullet points, score below 7
- If content appears to be search-engine-first rather than people-first, score below 6
- Explicitly mention what needs improvement in your feedback
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
                msg = "Waiting 5 seconds to respect rate limits..."
                logger.info(msg)
                if progress_callback:
                    progress_callback(msg)
                time.sleep(5)
                
        return full_content

    def save_content(self, content: str, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved generated content to {output_path}")
