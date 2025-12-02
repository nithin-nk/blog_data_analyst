"""
Mermaid Diagram Generator module.

Identifies diagram opportunities in blog content, generates mermaid diagrams with
quality review feedback loops, converts to images, and outputs structured YAML.
"""

import base64
import re
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call

logger = get_logger(__name__)


class DiagramOpportunity(BaseModel):
    """Structured output for diagram opportunity identification."""
    heading: str = Field(description="H2 or H3 heading where diagram should be placed")
    diagram_type: str = Field(
        description="Type of mermaid diagram (flowchart, sequenceDiagram, classDiagram, stateDiagram, erDiagram, gantt, pie, gitGraph)"
    )
    description: str = Field(description="What the diagram should illustrate")
    reasoning: str = Field(description="Why this diagram would add value")


class DiagramOpportunities(BaseModel):
    """Container for multiple diagram opportunities."""
    opportunities: List[DiagramOpportunity] = Field(
        description="List of diagram opportunities identified in the content"
    )


class DiagramReview(BaseModel):
    """Structured output for diagram quality review."""
    score: float = Field(
        description="Quality score from 0-10, where 10 is perfect. Evaluate: clarity, accuracy, completeness, proper mermaid syntax, and visual effectiveness"
    )
    strengths: List[str] = Field(description="What works well in this diagram (2-3 points)")
    weaknesses: List[str] = Field(description="What needs improvement (2-3 points)")
    feedback: str = Field(
        description="Detailed, actionable feedback for improving the diagram. Be specific about syntax issues, missing elements, or clarity improvements."
    )


class GeneratedDiagram(BaseModel):
    """Final diagram output with image."""
    heading: str
    diagram_type: str
    mermaid_code: str
    image_base64: str
    score: float
    description: str


class DiagramGenerator:
    """Generates mermaid diagrams from blog content with quality review feedback loops."""
    
    def __init__(self):
        self.settings = get_settings()
        
    def _extract_headings(self, content: str) -> List[str]:
        """Extract H2 and H3 headings from markdown content."""
        headings = []
        for line in content.split('\n'):
            if line.startswith('## ') or line.startswith('### '):
                heading = line.lstrip('#').strip()
                headings.append(heading)
        return headings
    
    def identify_diagram_opportunities(self, title: str, content: str) -> List[DiagramOpportunity]:
        """
        Identify potential diagram opportunities in blog content.
        
        Args:
            title: Blog title
            content: Blog content in markdown
            
        Returns:
            List of DiagramOpportunity objects
        """
        logger.info(f"Identifying diagram opportunities for: {title}")
        
        headings = self._extract_headings(content)
        headings_text = "\n".join([f"- {h}" for h in headings])
        
        prompt = f"""
You are an expert technical content visualizer. Analyze this blog post and identify opportunities for mermaid diagrams.

Blog Title: {title}

Available Headings (H2/H3):
{headings_text}

Blog Content:
{content[:5000]}... (truncated for analysis)

DIAGRAM TYPES:
- flowchart: Process flows, decision trees, system flows
- sequenceDiagram: Interactions between components/actors over time
- classDiagram: Object-oriented structures, relationships
- stateDiagram: State transitions, finite state machines
- erDiagram: Entity relationships, database schemas
- gantt: Project timelines, schedules
- pie: Data proportions, distributions
- gitGraph: Version control workflows

IDENTIFICATION CRITERIA:
1. **Add Value**: Diagram should clarify complex concepts, not simple ones
2. **Match Heading**: Diagram must match an existing H2/H3 heading
3. **Specificity**: Be specific about what the diagram shows
4. **Relevance**: Diagram should directly support the content

CONSTRAINTS:
- Only suggest diagrams for headings that exist in the content
- Maximum 5 diagrams per blog post
- Focus on technical concepts that benefit from visualization
- Avoid diagrams for simple lists or concepts

Identify diagram opportunities that would significantly enhance reader understanding.
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            result = gemini_llm_call(
                messages,
                model_name=self.settings.diagram_identifier_model,
                settings=self.settings,
                structured_output=DiagramOpportunities
            )
            
            logger.info(f"Identified {len(result.opportunities)} diagram opportunities")
            return result.opportunities
            
        except Exception as e:
            logger.error(f"Failed to identify diagram opportunities: {e}")
            return []
    
    def generate_mermaid_diagram(
        self,
        opportunity: DiagramOpportunity,
        content: str,
        feedback: Optional[str] = None,
        previous_code: Optional[str] = None
    ) -> str:
        """
        Generate mermaid diagram code.
        
        Args:
            opportunity: DiagramOpportunity defining what to generate
            content: Full blog content for context
            feedback: Optional feedback from previous iteration
            previous_code: Optional previous mermaid code to improve
            
        Returns:
            Mermaid diagram code as string
        """
        if feedback and previous_code:
            # Regeneration with feedback
            prompt = f"""
You are an expert mermaid diagram creator. Improve the existing mermaid diagram based on review feedback.

DIAGRAM REQUIREMENTS:
- Heading: {opportunity.heading}
- Type: {opportunity.diagram_type}
- Description: {opportunity.description}

PREVIOUS MERMAID CODE:
```mermaid
{previous_code}
```

REVIEW FEEDBACK:
{feedback}

IMPROVEMENT GUIDELINES:
1. Address all points in the feedback
2. Ensure proper mermaid syntax
3. Make it clear and visually effective
4. Keep it concise but complete
5. Use appropriate styling and labels

Generate ONLY the improved mermaid code. Do NOT include markdown code fences, just the raw mermaid syntax starting with the diagram type.
"""
        else:
            # Initial generation
            prompt = f"""
You are an expert mermaid diagram creator. Generate a mermaid diagram for this blog section.

DIAGRAM REQUIREMENTS:
- Heading: {opportunity.heading}
- Type: {opportunity.diagram_type}
- Description: {opportunity.description}
- Reasoning: {opportunity.reasoning}

CONTENT CONTEXT (section around this heading):
{self._extract_section_content(content, opportunity.heading)}

MERMAID DIAGRAM BEST PRACTICES:
1. Start with diagram type ({opportunity.diagram_type})
2. Use clear, concise labels
3. Maintain logical flow
4. Use appropriate styling (colors, shapes)
5. Keep it visually balanced
6. Include all key elements mentioned in description

SYNTAX EXAMPLES:
- flowchart TD
- sequenceDiagram
- classDiagram
- stateDiagram-v2
- erDiagram
- gantt
- pie

Generate ONLY the mermaid code. Do NOT include markdown code fences, just the raw mermaid syntax starting with the diagram type.
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            result = gemini_llm_call(
                messages,
                model_name=self.settings.diagram_generator_model,
                settings=self.settings
            )
            
            # Clean the result - remove any markdown code fences if present
            code = result.strip()
            if code.startswith('```'):
                code = re.sub(r'^```(?:mermaid)?\n', '', code)
                code = re.sub(r'\n```$', '', code)
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise
    
    def _extract_section_content(self, content: str, heading: str) -> str:
        """Extract content around a specific heading."""
        lines = content.split('\n')
        start_idx = None
        
        # Find the heading
        for i, line in enumerate(lines):
            if heading in line and (line.startswith('## ') or line.startswith('### ')):
                start_idx = i
                break
        
        if start_idx is None:
            return content[:2000]  # Return beginning if heading not found
        
        # Extract next ~1000 chars from heading
        section_lines = []
        for i in range(start_idx, min(start_idx + 50, len(lines))):
            section_lines.append(lines[i])
            if len('\n'.join(section_lines)) > 1000:
                break
        
        return '\n'.join(section_lines)
    
    def review_diagram(
        self,
        mermaid_code: str,
        opportunity: DiagramOpportunity,
        content: str
    ) -> Dict[str, Any]:
        """
        Review mermaid diagram quality.
        
        Args:
            mermaid_code: Generated mermaid code
            opportunity: Original diagram opportunity
            content: Blog content for context
            
        Returns:
            Dict with 'score', 'strengths', 'weaknesses', 'feedback'
        """
        logger.info(f"Reviewing diagram for: {opportunity.heading}")
        
        prompt = f"""
You are an expert mermaid diagram reviewer. Evaluate this diagram's quality.

DIAGRAM REQUIREMENTS:
- Heading: {opportunity.heading}
- Type: {opportunity.diagram_type}
- Description: {opportunity.description}

GENERATED MERMAID CODE:
```mermaid
{mermaid_code}
```

EVALUATION CRITERIA:

1. **Syntax Correctness (Weight: 25%)**:
   - Proper mermaid syntax
   - Valid diagram type
   - No syntax errors
   - Follows mermaid best practices

2. **Clarity & Visual Design (Weight: 25%)**:
   - Easy to understand
   - Logical flow/structure
   - Appropriate use of labels
   - Good visual balance

3. **Accuracy & Completeness (Weight: 25%)**:
   - Matches the description
   - Includes all key elements
   - Accurate representation of concepts
   - No misleading information

4. **Value & Effectiveness (Weight: 25%)**:
   - Adds value to the content
   - Enhances understanding
   - Appropriate complexity level
   - Professional quality

SCORING GUIDELINES:
- 9-10: Exceptional, publication-ready
- 7-8: Good with minor improvements needed
- 5-6: Acceptable but needs significant improvements
- 3-4: Poor, major rework required
- 1-2: Fundamentally flawed

Provide a score, strengths, weaknesses, and specific actionable feedback.
"""
        
        messages = [HumanMessage(content=prompt)]
        
        try:
            result = gemini_llm_call(
                messages,
                model_name=self.settings.diagram_reviewer_model,
                settings=self.settings,
                structured_output=DiagramReview
            )
            
            return {
                'score': result.score,
                'strengths': result.strengths,
                'weaknesses': result.weaknesses,
                'feedback': result.feedback
            }
            
        except Exception as e:
            logger.warning(f"Diagram review failed: {e}. Using fallback.")
            return {
                'score': 5.0,
                'strengths': [],
                'weaknesses': [],
                'feedback': f"Review failed: {str(e)}"
            }
    
    def generate_with_feedback_loop(
        self,
        opportunity: DiagramOpportunity,
        content: str
    ) -> Dict[str, Any]:
        """
        Generate diagram with iterative feedback loop.
        
        Args:
            opportunity: DiagramOpportunity defining what to generate
            content: Full blog content
            
        Returns:
            Dict with 'mermaid_code', 'score', 'review'
        """
        max_attempts = self.settings.max_diagram_generation_attempts
        threshold = self.settings.diagram_quality_threshold
        
        best_code = None
        best_score = 0.0
        best_review = None
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"  [Attempt {attempt}/{max_attempts}] Generating diagram for: {opportunity.heading}")
            
            # Generate diagram
            if attempt == 1:
                mermaid_code = self.generate_mermaid_diagram(opportunity, content)
            else:
                mermaid_code = self.generate_mermaid_diagram(
                    opportunity,
                    content,
                    feedback=best_review['feedback'],
                    previous_code=best_code
                )
            
            # Review diagram
            review = self.review_diagram(mermaid_code, opportunity, content)
            score = review['score']
            
            logger.info(f"  [Attempt {attempt}/{max_attempts}] Score: {score:.1f}/{threshold}")
            
            # Track best version
            if score > best_score:
                best_score = score
                best_code = mermaid_code
                best_review = review
            
            # Check if threshold met
            if score >= threshold:
                logger.info(f"  ✓ Quality threshold met (score: {score:.1f} >= {threshold})!")
                return {
                    'mermaid_code': mermaid_code,
                    'score': score,
                    'review': review
                }
        
        # Return best version
        logger.info(f"  Using best version (score: {best_score:.1f} after {max_attempts} attempts)")
        return {
            'mermaid_code': best_code,
            'score': best_score,
            'review': best_review
        }
    
    def convert_mermaid_to_image(self, mermaid_code: str) -> str:
        """
        Convert mermaid code to base64 encoded PNG image using Kroki.io API.
        
        Args:
            mermaid_code: Mermaid diagram code
            
        Returns:
            Base64 encoded PNG image
        """
        logger.info("Converting mermaid to image using Kroki.io")
        
        import requests
        import time
        
        url = "https://kroki.io/mermaid/png"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Sending diagram to: {url} (Attempt {attempt+1}/{max_retries})")
                
                # Make request with timeout
                response = requests.post(
                    url, 
                    data=mermaid_code.encode('utf-8'),
                    headers={'Content-Type': 'text/plain'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Get PNG bytes
                    png_bytes = response.content
                    
                    # Convert to base64
                    base64_image = base64.b64encode(png_bytes).decode('utf-8')
                    
                    logger.info(f"Successfully converted mermaid to image ({len(base64_image)} chars base64)")
                    return base64_image
                
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limited by Kroki.io. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"Kroki.io error {response.status_code}: {response.text[:200]}")
                    # Don't retry for 400 errors (syntax error)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
            
            except Exception as e:
                logger.error(f"Failed to convert mermaid to image: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return ""
    
    def generate_all_diagrams(
        self,
        title: str,
        content: str,
        progress_callback: Optional[Any] = None
    ) -> List[GeneratedDiagram]:
        """
        Main workflow: identify opportunities, generate diagrams, convert to images.
        
        Args:
            title: Blog title
            content: Blog content
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of GeneratedDiagram objects
        """
        logger.info(f"Starting diagram generation for: {title}")
        
        if progress_callback:
            progress_callback(f"\n{'='*80}\n📊 DIAGRAM GENERATION\n{'='*80}")
        
        # Step 1: Identify opportunities
        if progress_callback:
            progress_callback("Identifying diagram opportunities...")
        
        opportunities = self.identify_diagram_opportunities(title, content)
        
        if not opportunities:
            logger.info("No diagram opportunities identified")
            if progress_callback:
                progress_callback("No diagram opportunities identified")
            return []
        
        if progress_callback:
            progress_callback(f"Found {len(opportunities)} diagram opportunities")
        
        # Step 2: Generate diagrams with feedback loops
        generated_diagrams = []
        
        for i, opportunity in enumerate(opportunities, 1):
            if progress_callback:
                progress_callback(f"\n[Diagram {i}/{len(opportunities)}] {opportunity.heading}")
            
            try:
                # Generate with feedback loop
                result = self.generate_with_feedback_loop(opportunity, content)
                
                # Convert to image
                if progress_callback:
                    progress_callback(f"  Converting to image...")
                
                image_base64 = self.convert_mermaid_to_image(result['mermaid_code'])
                
                # Create final diagram object
                diagram = GeneratedDiagram(
                    heading=opportunity.heading,
                    diagram_type=opportunity.diagram_type,
                    mermaid_code=result['mermaid_code'],
                    image_base64=image_base64,
                    score=result['score'],
                    description=opportunity.description
                )
                
                generated_diagrams.append(diagram)
                
                if progress_callback:
                    progress_callback(f"  ✓ Diagram generated (score: {result['score']:.1f})")
                
            except Exception as e:
                logger.error(f"Failed to generate diagram for '{opportunity.heading}': {e}")
                if progress_callback:
                    progress_callback(f"  ✗ Failed to generate diagram: {e}")
        
        logger.info(f"Generated {len(generated_diagrams)} diagrams")
        return generated_diagrams
    
    def save_diagrams(self, diagrams: List[GeneratedDiagram], output_path: Path):
        """
        Save diagrams to YAML file and also save individual image files.
        
        Args:
            diagrams: List of GeneratedDiagram objects
            output_path: Path to save YAML file
        """
        output_data = {
            'diagrams': [
                {
                    'heading': d.heading,
                    'diagram_type': d.diagram_type,
                    'description': d.description,
                    'mermaid_code': d.mermaid_code,
                    'image_base64': d.image_base64,
                    'score': float(d.score)
                }
                for d in diagrams
            ],
            'total_diagrams': len(diagrams)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Saved {len(diagrams)} diagrams to {output_path}")
        
        # Also save individual diagram images to images folder
        self.save_diagram_images(diagrams, output_path.parent)
    
    def save_diagram_images(self, diagrams: List[GeneratedDiagram], blog_dir: Path):
        """
        Save diagram images as individual PNG files in the images folder.
        
        Args:
            diagrams: List of GeneratedDiagram objects
            blog_dir: Blog directory path
        """
        # Create images directory
        images_dir = blog_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i, diagram in enumerate(diagrams, 1):
            if not diagram.image_base64:
                logger.warning(f"Diagram '{diagram.heading}' has no image data, skipping file save")
                continue
            
            # Generate safe filename from heading
            safe_heading = re.sub(r"[^\w\s-]", "", diagram.heading.lower())
            safe_heading = re.sub(r"[-\s]+", "_", safe_heading).strip("_")[:40]
            filename = f"diagram_{i:02d}_{safe_heading}.png"
            
            image_path = images_dir / filename
            
            # Decode base64 and save
            try:
                image_bytes = base64.b64decode(diagram.image_base64)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Diagram image saved to: {image_path}")
            except Exception as e:
                logger.error(f"Failed to save diagram image '{diagram.heading}': {e}")
