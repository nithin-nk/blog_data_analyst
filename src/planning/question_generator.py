"""
Question Generator module for creating research queries.

Generates optimized Google search queries from a blog topic and context
using Gemini LLM with structured output.
"""

import asyncio
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchQuestions(BaseModel):
    """Structured output model for generated research questions."""
    
    questions: list[str] = Field(
        description="Up to 12 Google-optimized search queries covering all relevant categories",
        min_length=1,
        max_length=12,
    )
    categories_covered: list[str] = Field(
        description="Categories covered by the questions (e.g., 'fundamentals', 'implementation', 'ecosystem', etc.)",
        default_factory=list,
    )


class QuestionGenerator:
    """
    Generates research questions for a blog topic using Gemini LLM.
    
    Uses structured output to ensure consistent question format optimized
    for Google search queries.
    """
    
    SYSTEM_PROMPT = """
You are a research assistant helping to generate Google search queries for technical research.

Your task is to generate a COMPREHENSIVE set of Google-optimized search queries that will help gather all relevant information about a technical topic.

GUIDELINES:
1. Generate as many queries as needed to cover all important aspects. Do NOT limit to a fixed number.
2. Queries must be OPTIMIZED FOR GOOGLE SEARCH (not full questions)
    - Good: "LangChain AI agent framework overview"
    - Bad: "What is LangChain?"
3. Dynamically select categories based on the topic. Example categories:
    - Fundamentals / Overview
    - Implementation / Getting Started / Tutorials
    - Architecture / Design Patterns
    - Integration / Ecosystem / Tools
    - Best Practices / Production
    - Common Pitfalls / Troubleshooting
    - Community Support / Forums
    - Recent Developments / Releases
    - Open Issues / GitHub
    - Scalability / Performance
    - Real-world Use Cases
    - Future Roadmap / Trends
4. If context is provided, incorporate it into relevant queries and category selection.
5. Include queries about recent developments, community support, and open issues when relevant.
6. Overlapping or similar queries are acceptable if they increase coverage.
7. Output must be in YAML format with the following structure:

topic: "<topic>"
questions:
  - "<query 1>"
  - "<query 2>"
  ...
categories_covered:
  - "<category 1>"
  - "<category 2>"
  ...
"""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the question generator.
        
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
                raise ValueError("GOOGLE_API_KEY is required for question generation")
            
            base_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.settings.google_api_key,
                temperature=0.7,
            )
            self._llm = base_llm.with_structured_output(ResearchQuestions)
            logger.debug(f"Initialized LLM with model: {self.model_name}")
        return self._llm
    
    def _build_prompt(self, topic: str, context: Optional[str] = None) -> str:
        """
        Build the user prompt for comprehensive question generation.
        
        Args:
            topic: Technical topic/title
            context: Optional additional context or constraints
        Returns:
            Formatted prompt string
        """
        prompt = f"Generate a comprehensive set of Google-optimized search queries for researching this technical topic. Output YAML as specified.\n\n"
        prompt += f"TOPIC: {topic}\n"
        if context:
            prompt += f"\nADDITIONAL CONTEXT/CONSTRAINTS:\n{context}\n"
            prompt += "\nIncorporate the context into relevant queries and category selection."
        prompt += "\n\nRemember: Generate as many search-optimized queries as needed, not full questions."
        return prompt
    
    async def generate(
        self,
        topic: str,
        context: Optional[str] = None
    ) -> ResearchQuestions:
        """
        Generate up to 12 research questions for a technical topic.
        Args:
            topic: Technical topic/title to research
            context: Optional additional context or constraints
        Returns:
            ResearchQuestions with up to 12 Google-optimized search queries
        """
        logger.info(f"Generating up to 12 research questions for topic: {topic}")
        if context:
            logger.debug(f"Context provided: {context[:100]}...")
        prompt = self._build_prompt(topic, context)
        prompt += "\n\nLimit the output to a maximum of 12 search queries."
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            result: ResearchQuestions = await self.llm.ainvoke(messages)
            # Truncate to 12 if LLM returns more
            if len(result.questions) > 12:
                logger.warning(f"LLM returned {len(result.questions)} questions, truncating to 12.")
                result.questions = result.questions[:12]
            logger.info(f"Generated {len(result.questions)} research questions")
            logger.debug(f"Categories covered: {result.categories_covered}")
            for i, question in enumerate(result.questions, 1):
                logger.debug(f"  Q{i}: {question}")
            return result
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            raise
    
    def generate_sync(
        self, 
        topic: str, 
        context: Optional[str] = None
    ) -> ResearchQuestions:
        """
        Synchronous wrapper for generate().
        
        Args:
            topic: Blog topic/title to research
            context: Optional additional context or constraints
            
        Returns:
            ResearchQuestions with list of Google-optimized search queries
        """
        return asyncio.run(self.generate(topic, context))


async def main():
    """Demo usage of QuestionGenerator."""
    generator = QuestionGenerator()
    
    topic = "How to build memory for AI agents using mem0"
    context = "Focus on the open source Python framework, include practical implementation examples"
    
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Context: {context}")
    print(f"{'='*60}\n")
    
    result = await generator.generate(topic, context)
    
    print(f"Generated {len(result.questions)} questions:\n")
    for i, q in enumerate(result.questions, 1):
        print(f"  {i:2}. {q}")
    
    print(f"\nCategories covered: {', '.join(result.categories_covered)}")


if __name__ == "__main__":
    asyncio.run(main())
