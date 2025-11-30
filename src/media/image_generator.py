"""
Image generator using Banana/Nano API.

Generates blog post images based on topic and themes.
"""

from pathlib import Path
from typing import Optional
import httpx

from src.config.settings import get_settings
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class ImageGenerator:
    """Generator for blog post images using Banana/Nano API."""
    
    def __init__(self) -> None:
        """Initialize the image generator."""
        self.settings = get_settings()
        self.api_key = self.settings.banana_api_key
        self.model_key = self.settings.banana_model_key
        self.base_url = "https://api.banana.dev"  # Update with actual Banana API URL
        
        logger.info("ImageGenerator initialized")
    
    async def generate(
        self,
        topic: str,
        themes: list[str],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an image for a blog post.
        
        Args:
            topic: Main blog topic
            themes: Key themes from the blog
            output_path: Where to save the image (auto-generated if None)
            
        Returns:
            Path to the saved image
        """
        logger.info(f"Generating image for topic: {topic}")
        
        # Generate image prompt
        prompt = self._create_prompt(topic, themes)
        logger.debug(f"Image prompt: {prompt}")
        
        # Call Banana API
        image_url = await self._call_banana_api(prompt)
        
        # Download and save image
        if output_path is None:
            output_path = self._get_default_path(topic)
        
        await self._download_image(image_url, output_path)
        
        logger.info(f"Image saved to: {output_path}")
        return output_path
    
    def _create_prompt(self, topic: str, themes: list[str]) -> str:
        """
        Create an image generation prompt from topic and themes.
        
        Args:
            topic: Main blog topic
            themes: Key themes
            
        Returns:
            Image generation prompt
        """
        themes_text = ", ".join(themes[:3])  # Use top 3 themes
        
        prompt = (
            f"Professional blog header image for '{topic}'. "
            f"Themes: {themes_text}. "
            f"Style: modern, clean, professional, tech-oriented. "
            f"High quality, 16:9 aspect ratio."
        )
        
        return prompt
    
    async def _call_banana_api(self, prompt: str) -> str:
        """
        Call Banana API to generate image.
        
        Args:
            prompt: Image generation prompt
            
        Returns:
            URL of the generated image
        """
        logger.debug("Calling Banana API")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "modelKey": self.model_key,
            "modelInputs": {
                "prompt": prompt,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            },
        }
        
        async with httpx.AsyncClient() as client:
            # TODO: Implement actual Banana API call
            # response = await client.post(
            #     f"{self.base_url}/start/v4",
            #     headers=headers,
            #     json=payload,
            # )
            # response.raise_for_status()
            # result = response.json()
            # image_url = result["modelOutputs"]["image_url"]
            
            # Placeholder
            image_url = "https://placeholder.com/image.png"
        
        logger.debug(f"Received image URL: {image_url}")
        return image_url
    
    async def _download_image(self, url: str, output_path: Path) -> None:
        """
        Download image from URL and save to file.
        
        Args:
            url: Image URL
            output_path: Where to save the image
        """
        logger.debug(f"Downloading image from: {url}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        logger.debug(f"Image downloaded to: {output_path}")
    
    def _get_default_path(self, topic: str) -> Path:
        """
        Generate default output path for image.
        
        Args:
            topic: Blog topic
            
        Returns:
            Path for the image file
        """
        # Create safe filename from topic
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic.lower())
        safe_topic = safe_topic[:50]  # Limit length
        
        filename = f"{safe_topic}_header.png"
        return self.settings.output_dir / "images" / filename
