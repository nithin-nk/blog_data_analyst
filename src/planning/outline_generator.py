"""
Outline Generator module for creating structured blog outlines.

Generates a detailed blog outline using Gemini LLM based on research data.
"""

import asyncio
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.research.content_extractor import AggregatedExtractedContent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlineSection(BaseModel):
    """A single section in the blog outline."""

    heading: str = Field(description="Section heading")
    summary: str = Field(
        description="Detailed summary of what this section should cover. DO NOT include URLs or links here."
    )
    references: list[str] = Field(
        default_factory=list,
        description="List of source URLs relevant to this section (from research data).",
    )


class OutlineMetadata(BaseModel):
    """Metadata for the blog post."""

    target_audience: str = Field(description="Primary audience for the blog post")
    difficulty: str = Field(
        description="Difficulty level (e.g., Beginner, Intermediate, Advanced)"
    )
    estimated_reading_time: str = Field(description="Estimated reading time (e.g., '10 minutes')")


class BlogOutline(BaseModel):
    """Structured blog outline."""

    topic: str = Field(description="Blog post topic/title")
    sections: list[OutlineSection] = Field(
        description="Ordered list of blog sections (Introduction -> Body -> Conclusion)"
    )
    metadata: OutlineMetadata = Field(description="Blog post metadata")


class OutlineGenerator:
    """
    Generates a structured blog outline from research data using Gemini LLM.
    """

    SYSTEM_PROMPT = """
You are an expert technical content strategist and editor specializing in creating high-quality, SEO-optimized technical blog posts.

Your task is to create a COMPREHENSIVE yet CONCISE blog post outline that is optimized for both reader engagement and search visibility.

CONTENT PHILOSOPHY:
- Create outlines that enable COMPREHENSIVE content while maintaining QUICK READABILITY (aim for 12-18 min estimated reading time)
- Balance theory, practical implementation, and real-world context
- Mix conceptual explanations with hands-on code examples and visual diagrams
- Maintain a tutorial-like flow that guides readers from understanding to implementation

STRUCTURE GUIDELINES:
1.  **Opening (Introduction)**:
    - Hook readers with a problem statement or use case
    - Briefly explain why this topic matters (business value, technical benefits)
    - Set clear expectations for what readers will learn
    - Target keyword should appear naturally in the intro summary

2.  **Core Concepts (Foundation)**:
    - Break down complex topics into digestible sections
    - Use analogies or comparisons when helpful
    - Include architecture diagrams for system-level concepts
    - Mention "Include Mermaid diagram for [architecture/flow/concept]" when visualizing helps understanding

3.  **Implementation (Hands-On)**:
    - Provide step-by-step guides with code examples
    - For setup/installation sections: "Include code example for installation and basic configuration"
    - For integration sections: "Include code example demonstrating [specific integration/feature]"
    - Code examples should be MINIMAL but COMPLETE (enough to run/test, not overly verbose)
    - Prefer practical, real-world snippets over toy examples

4.  **Advanced Topics (When Applicable)**:
    - Include ONLY if research data supports it: Performance optimization, Best practices, Common pitfalls, Troubleshooting
    - Keep advanced sections focused and actionable

5.  **Comparisons (When Relevant)**:
    - Compare with alternatives only if it adds value
    - Focus on strengths/weaknesses, not just feature lists
    - Use comparison tables (suggest in summary when appropriate)

6.  **Practical Applications**:
    - Showcase real-world use cases or examples
    - Connect theory to practice

7.  **Closing (Conclusion)**:
    - Summarize key takeaways (3-5 bullet points worth)
    - Suggest next steps or further learning paths
    - End with a forward-looking statement about the technology's future

SEO OPTIMIZATION:
- Primary keyword should appear in 2-3 section headings naturally
- Each section summary should hint at secondary keywords from research
- Section headings should be descriptive and search-friendly (not clever/vague)
- Include a "Prerequisites" or "What You'll Need" section IF the topic requires specific setup/knowledge
- Suggest H2/H3 hierarchy that follows SEO best practices

CODE & DIAGRAM SPECIFICATIONS:
- **Code Examples**: Specify WHEN and WHAT
  - "Include code example for: [specific task, e.g., 'initializing the client and making the first API call']"
  - Indicate language/framework if multiple options exist
  - Mention if code should show error handling or be basic
  
- **Mermaid Diagrams**: Specify PURPOSE
  - "Include Mermaid diagram showing: [architecture/flow/relationship/sequence]"
  - Examples: "system architecture diagram", "data flow diagram", "authentication sequence diagram"
  - Only suggest diagrams when they CLARIFY complexity (not for simple concepts)

REFERENCES:
- Map 2-5 most relevant URLs to each section
- Prioritize official docs, well-known blogs, and research papers
- **CRITICAL**: The `summary` field must NOT contain any URLs or links. All links must go in the `references` list.

QUALITY CRITERIA:
- Sections should be balanced (avoid one massive section and several tiny ones)
- Aim for 6-10 main sections for comprehensive coverage
- Each section summary should be 2-4 sentences, clearly stating WHAT will be covered and WHY it matters
- Avoid redundancy between sections
- Ensure logical progression: each section should build on previous ones

METADATA:
- target_audience: Be specific (e.g., "Backend developers familiar with Python", "DevOps engineers", "AI/ML practitioners")
- difficulty: Choose Beginner/Intermediate/Advanced based on prerequisites and complexity
- estimated_reading_time: Calculate based on content depth (6-10 sections = 12-18 minutes typically)

INPUT DATA:
You will be given:
-   Topic (the blog post subject)
-   Summarized research data (titles, snippets, headings from authoritative sources)

OUTPUT:
Generate a structured JSON/YAML response matching the `BlogOutline` schema. Ensure all guidelines above are reflected in your outline.
"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the outline generator.

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash)
        """
        self.settings = get_settings()
        self.model_name = model_name
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy initialization of LLM with structured output."""
        if self._llm is None:
            if not self.settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for outline generation")

            base_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.settings.google_api_key,
                temperature=0.7,
            )
            self._llm = base_llm.with_structured_output(BlogOutline)
            logger.debug(f"Initialized LLM with model: {self.model_name}")
        return self._llm

    def _format_research_data(self, research_data: AggregatedExtractedContent) -> str:
        """
        Format research data for the prompt.

        Args:
            research_data: Aggregated content from research phase

        Returns:
            Formatted string summary of research
        """
        summary = f"Research Summary for '{research_data.topic}':\n\n"
        
        # Add stats
        stats = research_data.statistics
        summary += f"Analyzed {stats.get('successful', 0)} URLs.\n\n"

        # Add content summaries
        for i, content in enumerate(research_data.contents, 1):
            if not content.success:
                continue
                
            summary += f"Source {i}: {content.title}\n"
            summary += f"URL: {content.url}\n"
            summary += f"Snippet: {content.snippet}\n"
            
            # Add top headings to give context of content structure
            if content.headings:
                top_headings = content.headings[:5]
                summary += "Key Sections:\n" + "\n".join([f"  - {h}" for h in top_headings]) + "\n"
            
            summary += "\n" + "-" * 40 + "\n\n"

        return summary

    async def generate(
        self,
        topic: str,
        research_data: AggregatedExtractedContent
    ) -> BlogOutline:
        """
        Generate a blog outline based on research data.

        Args:
            topic: Blog topic
            research_data: Extracted content from research phase

        Returns:
            BlogOutline structured object
        """
        logger.info(f"Generating outline for topic: {topic}")
        
        research_summary = self._format_research_data(research_data)
        
        prompt = f"Generate a detailed blog post outline for the topic: '{topic}'\n\n"
        prompt += "Based on the following research data:\n\n"
        prompt += research_summary
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result: BlogOutline = await self.llm.ainvoke(messages)
            logger.info(f"Generated outline with {len(result.sections)} sections")
            return result
        except Exception as e:
            logger.error(f"Failed to generate outline: {e}")
            raise

    def generate_sync(
        self,
        topic: str,
        research_data: AggregatedExtractedContent
    ) -> BlogOutline:
        """
        Synchronous wrapper for generate().

        Args:
            topic: Blog topic
            research_data: Extracted content from research phase

        Returns:
            BlogOutline structured object
        """
        return asyncio.run(self.generate(topic, research_data))
