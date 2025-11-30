"""
Code and Mermaid diagram generator.

Generates code snippets and Mermaid diagrams based on outline markers.
"""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.llm_helpers import gemini_llm_call
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import get_settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class CodeGenerator:
    """Generator for code snippets and Mermaid diagrams."""
    
    def __init__(self) -> None:
        """Initialize the code generator."""
        self.settings = get_settings()
        # LLM handled by helper
        logger.info("CodeGenerator initialized")
    
    async def generate_code(
        self,
        question: str,
        topic: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Generate code snippet for a specific question.
        
        Args:
            question: The question requiring code
            topic: Main blog topic
            language: Programming language (auto-detected if None)
            
        Returns:
            Generated code snippet with explanation
        """
        logger.info(f"Generating code for: {question}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert programmer. Generate clean, well-commented 
            code that demonstrates the concept clearly.
            
            Requirements:
            - Include inline comments
            - Follow best practices
            - Make it production-ready
            - Add a brief explanation before the code
            """),
            ("user", """Topic: {topic}
Question: {question}
Language: {language}

Generate a code example with explanation."""),
        ])
        
        messages = prompt.format(topic=topic, question=question, language=language or "python")
        result = gemini_llm_call([
            {"role": "system", "content": "You are an expert programmer. Generate clean, well-commented code that demonstrates the concept clearly."},
            {"role": "user", "content": f"Topic: {topic}\nQuestion: {question}\nLanguage: {language or 'python'}\nGenerate a code example with explanation."}
        ], model_name="gemini-pro", settings=self.settings)
        logger.info("Code generated successfully")
        return result
    
    async def generate_mermaid(
        self,
        question: str,
        topic: str,
        diagram_type: Optional[str] = None,
    ) -> str:
        """
        Generate Mermaid diagram for visualization.
        
        Args:
            question: The question requiring a diagram
            topic: Main blog topic
            diagram_type: Type of diagram (flowchart, sequence, etc.)
            
        Returns:
            Mermaid diagram syntax
        """
        logger.info(f"Generating Mermaid diagram for: {question}")
        
        result = gemini_llm_call([
            {"role": "system", "content": "You are an expert programmer. Generate a mermaid diagram for visualization."},
            {"role": "user", "content": f"Topic: {topic}\nQuestion: {question}\nDiagram Type: {diagram_type or 'flowchart'}\nGenerate a mermaid diagram."}
        ], model_name="gemini-pro", settings=self.settings)
        logger.info("Mermaid diagram generated successfully")
        return result
    
    @staticmethod
    def validate_mermaid(mermaid_syntax: str) -> bool:
        """
        Validate Mermaid diagram syntax.
        
        Args:
            mermaid_syntax: The Mermaid code to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - check for required elements
        has_mermaid_marker = "```mermaid" in mermaid_syntax
        has_graph_type = any(
            keyword in mermaid_syntax
            for keyword in ["graph", "sequenceDiagram", "classDiagram", "flowchart"]
        )
        
        return has_mermaid_marker and has_graph_type
