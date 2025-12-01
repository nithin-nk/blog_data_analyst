"""
Blog refiner for iterative improvement.

Applies quality check feedback to improve blog content.
"""

from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import get_settings
from src.optimization.quality_checker import QualityChecker
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class BlogRefiner:
    """Iterative blog content refiner."""
    
    def __init__(self) -> None:
        """Initialize the blog refiner."""
        self.settings = get_settings()
        self.quality_checker = QualityChecker()
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.settings.google_api_key,
            temperature=0.5,
        )
        
        logger.info("BlogRefiner initialized")
    
    async def refine(
        self,
        content: str,
        topic: str,
        max_iterations: int = 3,
        target_score: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Iteratively refine blog content based on quality feedback.
        
        Args:
            content: Initial blog content
            topic: Main blog topic
            max_iterations: Maximum refinement iterations
            target_score: Target quality score (1-10)
            
        Returns:
            Dict containing final content, scores, and iteration history
        """
        logger.info(f"Starting refinement (max {max_iterations} iterations, target: {target_score})")
        
        current_content = content
        iteration_history = []
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"Refinement iteration {iteration}/{max_iterations}")
            
            # Check quality
            quality_result = await self.quality_checker.check_quality(
                current_content,
                topic,
            )
            
            overall_score = quality_result["overall_score"]
            logger.info(f"Current quality score: {overall_score:.1f}/10")
            
            iteration_history.append({
                "iteration": iteration,
                "score": overall_score,
                "content_length": len(current_content),
            })
            
            # Check if target achieved
            if overall_score >= target_score:
                logger.info(f"Target score {target_score} achieved!")
                break
            
            # Check if max iterations reached
            if iteration >= max_iterations:
                logger.info("Max iterations reached")
                break
            
            # Apply improvements
            current_content = await self._apply_improvements(
                current_content,
                topic,
                quality_result["combined_suggestions"],
            )
        
        final_quality = await self.quality_checker.check_quality(current_content, topic)
        
        return {
            "final_content": current_content,
            "final_score": final_quality["overall_score"],
            "final_quality_details": final_quality,
            "iterations_used": len(iteration_history),
            "iteration_history": iteration_history,
            "target_achieved": final_quality["overall_score"] >= target_score,
        }
    
    async def _apply_improvements(
        self,
        content: str,
        topic: str,
        suggestions: list[str],
    ) -> str:
        """
        Apply improvement suggestions to content.
        
        Args:
            content: Current blog content
            topic: Main blog topic
            suggestions: List of improvement suggestions
            
        Returns:
            Improved content
        """
        logger.info(f"Applying {len(suggestions)} improvement suggestions")
        
        suggestions_text = "\n".join(f"- {s}" for s in suggestions)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert blog editor. Improve the given blog content 
            by applying the specific suggestions provided. Maintain the original structure 
            and key points while making targeted improvements.
            
            Requirements:
            - Address each suggestion specifically
            - Preserve citations and formatting
            - Maintain overall length (don't make it significantly longer/shorter)
            - Keep the markdown formatting intact
            """),
            ("user", """Topic: {topic}

Current Content:
{content}

Improvement Suggestions:
{suggestions}

Please provide the improved version of the content."""),
        ])
        
        # TODO: Implement actual improvement application
        improved_content = content  # Placeholder
        
        logger.info("Improvements applied")
        return improved_content
