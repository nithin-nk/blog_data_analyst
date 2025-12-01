"""
Quality checker using dual LLM review.

Uses two different LLMs (Gemini + GPT-4) to evaluate blog quality.
"""

from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import get_settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class QualityChecker:
    """Dual LLM quality checker for blog content."""
    
    def __init__(self) -> None:
        """Initialize both LLMs for quality checking."""
        self.settings = get_settings()
        
        # Primary LLM: Google Gemini
        self.llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.settings.google_api_key,
            temperature=0.3,
        )
        
        # Secondary LLM: OpenAI GPT-4
        self.llm_openai = ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.settings.openai_api_key,
            temperature=0.3,
        )
        
        logger.info("QualityChecker initialized with Gemini and GPT-4")
    
    async def check_quality(self, content: str, topic: str) -> Dict[str, Any]:
        """
        Perform dual LLM quality check.
        
        Args:
            content: Blog content to evaluate
            topic: Main blog topic
            
        Returns:
            Dict containing scores and suggestions from both LLMs
        """
        logger.info("Starting dual LLM quality check")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content quality reviewer. Evaluate this blog post 
            on a scale of 1-10 for each criterion and provide specific improvement suggestions.
            
            Criteria to evaluate:
            1. Accuracy: Factual correctness and reliability
            2. Clarity: Clear writing and easy to understand
            3. Engagement: Interesting and engaging content
            4. Structure: Logical flow and organization
            5. Completeness: Thorough coverage of the topic
            
            Provide your response in this exact format:
            SCORES:
            Accuracy: X/10
            Clarity: X/10
            Engagement: X/10
            Structure: X/10
            Completeness: X/10
            
            SUGGESTIONS:
            - Specific improvement 1
            - Specific improvement 2
            - ...
            """),
            ("user", """Topic: {topic}

Content:
{content}

Please evaluate this blog post."""),
        ])
        
        # TODO: Implement actual LLM calls
        gemini_result = self._parse_review("TODO: Gemini review")
        openai_result = self._parse_review("TODO: OpenAI review")
        
        # Calculate average scores
        averaged_scores = self._average_scores(gemini_result, openai_result)
        
        result = {
            "gemini_review": gemini_result,
            "openai_review": openai_result,
            "averaged_scores": averaged_scores,
            "overall_score": averaged_scores["overall"],
            "combined_suggestions": self._combine_suggestions(
                gemini_result["suggestions"],
                openai_result["suggestions"],
            ),
        }
        
        logger.info(f"Quality check complete. Overall score: {result['overall_score']:.1f}/10")
        return result
    
    def _parse_review(self, review_text: str) -> Dict[str, Any]:
        """
        Parse LLM review response into structured format.
        
        Args:
            review_text: Raw LLM response
            
        Returns:
            Dict with scores and suggestions
        """
        # TODO: Implement actual parsing
        scores = {
            "accuracy": 0,
            "clarity": 0,
            "engagement": 0,
            "structure": 0,
            "completeness": 0,
        }
        
        suggestions = []
        
        overall = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "scores": scores,
            "overall": overall,
            "suggestions": suggestions,
        }
    
    @staticmethod
    def _average_scores(
        gemini_result: Dict[str, Any],
        openai_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Average scores from both LLMs."""
        gemini_scores = gemini_result["scores"]
        openai_scores = openai_result["scores"]
        
        averaged = {}
        for key in gemini_scores:
            averaged[key] = (gemini_scores[key] + openai_scores[key]) / 2
        
        averaged["overall"] = sum(averaged.values()) / len(averaged) if averaged else 0
        
        return averaged
    
    @staticmethod
    def _combine_suggestions(
        gemini_suggestions: List[str],
        openai_suggestions: List[str],
    ) -> List[str]:
        """Combine and deduplicate suggestions from both LLMs."""
        # TODO: Implement smart deduplication
        all_suggestions = gemini_suggestions + openai_suggestions
        
        # Simple deduplication for now
        seen = set()
        unique = []
        for suggestion in all_suggestions:
            suggestion_lower = suggestion.lower().strip()
            if suggestion_lower not in seen:
                seen.add(suggestion_lower)
                unique.append(suggestion)
        
        return unique
