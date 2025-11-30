"""
Content generator using LangChain and Google Gemini.

Generates blog content sections with citations from research data.
"""

from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from src.config.settings import get_settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class ContentGenerator:
    """Generator for blog content using LLM."""
    
    def __init__(self) -> None:
        """Initialize the content generator."""
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.settings.google_api_key,
            temperature=0.7,
        )
        logger.info("ContentGenerator initialized with Gemini")
    
    async def generate_section(
        self,
        question: str,
        research_data: List[Dict[str, Any]],
        topic: str,
    ) -> str:
        """
        Generate a blog section for a specific question using research data.
        
        Args:
            question: The outline question to answer
            research_data: List of extracted content from web sources
            topic: Main blog topic for context
            
        Returns:
            Generated markdown content with citations
        """
        logger.info(f"Generating section for: {question}")
        
        # Prepare research context
        context = self._prepare_context(research_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert blog content writer. Generate a comprehensive, 
            engaging section (300-500 words) that answers the given question.
            
            Requirements:
            - Use the provided research data as factual sources
            - Include inline citations in format: [Source Name](URL)
            - Write in a clear, professional tone
            - Use markdown formatting (headers, lists, bold, italic)
            - Make content SEO-friendly with relevant keywords
            - Ensure accuracy and depth
            """),
            ("user", """Topic: {topic}
            
Question to answer: {question}

Research Data:
{context}

Generate a comprehensive section answering this question."""),
        ])
        
        # TODO: Implement actual LLM call
        result = "TODO: Generated content will appear here"
        
        logger.info(f"Generated {len(result)} characters for section")
        return result
    
    def _prepare_context(self, research_data: List[Dict[str, Any]]) -> str:
        """
        Prepare research data as context for LLM.
        
        Args:
            research_data: List of extracted content
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, data in enumerate(research_data, 1):
            if data.get("success"):
                context_parts.append(
                    f"Source {i}: {data.get('title', 'Unknown')}\n"
                    f"URL: {data.get('url', '')}\n"
                    f"Content: {data.get('text', '')[:500]}...\n"
                )
        
        return "\n\n".join(context_parts)
    
    async def combine_sections(
        self,
        sections: List[str],
        topic: str,
    ) -> str:
        """
        Combine multiple sections into a cohesive blog post.
        
        Args:
            sections: List of generated sections
            topic: Main blog topic
            
        Returns:
            Combined blog post with smooth transitions
        """
        logger.info(f"Combining {len(sections)} sections")
        
        # TODO: Implement section combining with transitions
        combined = "\n\n".join(sections)
        
        return combined
