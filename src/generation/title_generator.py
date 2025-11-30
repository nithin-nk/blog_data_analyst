"""
Title, tags, and metadata generator.

Generates catchy titles, relevant tags, and SEO meta descriptions.
"""

from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from src.config.settings import get_settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class TitleGenerator:
    """Generator for blog titles, tags, and metadata."""
    
    def __init__(self) -> None:
        """Initialize the title generator."""
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.settings.google_api_key,
            temperature=0.8,  # Higher temperature for creativity
        )
        logger.info("TitleGenerator initialized")
    
    async def generate_titles(
        self,
        topic: str,
        content: str,
        num_options: int = 5,
    ) -> List[str]:
        """
        Generate multiple title options.
        
        Args:
            topic: Main blog topic
            content: Full blog content
            num_options: Number of title variations to generate
            
        Returns:
            List of title options
        """
        logger.info(f"Generating {num_options} title options")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at writing catchy, SEO-optimized blog titles.
            
            Requirements:
            - Titles should be 50-60 characters
            - Include power words and numbers when appropriate
            - Make them click-worthy but not clickbait
            - Include relevant keywords
            """),
            ("user", """Topic: {topic}

Generate {num_options} different title options for this blog post."""),
        ])
        
        # TODO: Implement actual title generation
        titles = [f"TODO: Title {i+1}" for i in range(num_options)]
        
        logger.info(f"Generated {len(titles)} titles")
        return titles
    
    async def generate_tags(
        self,
        topic: str,
        content: str,
        num_tags: int = 10,
    ) -> List[str]:
        """
        Generate relevant tags/keywords.
        
        Args:
            topic: Main blog topic
            content: Full blog content
            num_tags: Number of tags to generate
            
        Returns:
            List of tags
        """
        logger.info(f"Generating {num_tags} tags")
        
        # TODO: Implement tag generation
        tags = []
        
        logger.info(f"Generated {len(tags)} tags")
        return tags
    
    async def generate_meta_description(
        self,
        topic: str,
        content: str,
    ) -> str:
        """
        Generate SEO meta description.
        
        Args:
            topic: Main blog topic
            content: Full blog content
            
        Returns:
            Meta description (150-160 characters)
        """
        logger.info("Generating meta description")
        
        # TODO: Implement meta description generation
        description = "TODO: Meta description will be generated here"
        
        # Ensure it's within character limit
        if len(description) > 160:
            description = description[:157] + "..."
        
        logger.info(f"Generated meta description: {len(description)} characters")
        return description
    
    async def generate_all_metadata(
        self,
        topic: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Generate all metadata at once.
        
        Args:
            topic: Main blog topic
            content: Full blog content
            
        Returns:
            Dict containing titles, tags, and meta description
        """
        logger.info("Generating all metadata")
        
        titles = await self.generate_titles(topic, content)
        tags = await self.generate_tags(topic, content)
        description = await self.generate_meta_description(topic, content)
        
        return {
            "titles": titles,
            "tags": tags,
            "meta_description": description,
        }
