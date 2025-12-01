"""
Blog Image Generator module.

Generates a blog post cover/social sharing image using Gemini 2.5 Flash.
Takes the blog title and content, generates a creative image description,
then generates an image using Gemini's image generation model.
"""

import base64
from pathlib import Path
from typing import Any, Optional

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call


logger = get_logger(__name__)


class ImageDescription(BaseModel):
    """Structured output for image description generation."""

    description: str = Field(
        description="A vivid, creative visual description for the blog post image. Should be 2-3 sentences describing the scene, style, and key visual elements."
    )
    alt_text: str = Field(
        description="Concise alt text for accessibility (1 sentence, max 125 characters)"
    )
    style: str = Field(
        description="Image style: illustration, abstract, photorealistic, minimal, etc."
    )


class GeneratedBlogImage(BaseModel):
    """Final blog image output."""

    title: str
    description: str
    alt_text: str
    style: str
    image_base64: str
    format: str = "png"


class BlogImageGenerator:
    """
    Generates blog cover images using Gemini.

    Workflow:
    1. Analyze blog title and content
    2. Generate creative image description
    3. Use Gemini 2.5 Flash to generate image
    4. Return base64 encoded image with metadata
    """

    def __init__(self):
        self.settings = get_settings()
        self._init_genai_client()

    def _init_genai_client(self):
        """Initialize the Google GenAI client with paid API key for image generation."""
        # Use paid key specifically for image generation
        api_key = self.settings.google_api_paid_key
        if not api_key:
            # Fallback to regular key if paid key not set
            api_key = self.settings.google_api_key
            logger.warning("GOOGLE_API_PAID_KEY not set, falling back to GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_PAID_KEY or GOOGLE_API_KEY is required for image generation")
        self.client = genai.Client(api_key=api_key)
        logger.info("GenAI client initialized with paid API key for image generation")

    def generate_image_description(
        self, title: str, content: str
    ) -> ImageDescription:
        """
        Generate a creative image description from blog title and content.

        Args:
            title: Blog post title
            content: Blog post content (markdown)

        Returns:
            ImageDescription with description, alt_text, and style
        """
        logger.info(f"Generating image description for: {title}")

        # Truncate content to avoid token limits
        content_preview = content[:3000] if len(content) > 3000 else content

        prompt = f"""
You are an expert visual designer creating cover images for technical blog posts.
Analyze this blog post and generate a compelling image description that:
1. Captures the main theme/concept visually
2. Is suitable for a 1200x630 social sharing image
3. Uses modern, professional design aesthetics
4. Incorporates relevant visual metaphors for technical concepts

Blog Title: {title}

Blog Content Preview:
{content_preview}

GUIDELINES:
- Description should be vivid but achievable by an AI image generator
- Focus on visual storytelling that represents the blog's key concept
- Use colors and elements that convey professionalism and tech themes
- Avoid text in images (text renders poorly in AI-generated images)
- Alt text should be concise and accessible (max 125 chars)

Generate a creative, visually compelling description for this blog's cover image.
"""

        messages = [HumanMessage(content=prompt)]

        try:
            result = gemini_llm_call(
                messages,
                model_name=self.settings.blog_image_description_model,
                settings=self.settings,
                structured_output=ImageDescription,
            )

            logger.info(f"Generated image description: {result.description[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Failed to generate image description: {e}")
            # Return a fallback description
            return ImageDescription(
                description=f"A modern, professional tech illustration representing the concept of {title}. Clean design with abstract geometric shapes and a blue-purple color palette.",
                alt_text=f"Blog cover image for {title[:100]}",
                style="illustration",
            )

    def generate_image(self, description: str) -> tuple[bytes, str]:
        """
        Generate an image using Gemini 2.5 Flash image generation.

        Args:
            description: Visual description/prompt for the image

        Returns:
            Tuple of (image_bytes, format)
        """
        logger.info("Generating image with Gemini")

        # Enhance prompt for social sharing dimensions
        enhanced_prompt = (
            f"{description} "
            "Style: Professional blog cover image, 1200x630 aspect ratio suitable for social sharing. "
            "High quality, modern design, no text overlays."
        )

        max_retries = 3
        retry_delay = 25  # seconds, based on API retry suggestion

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=[enhanced_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    ),
                )

                # Extract image from response
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type
                        fmt = mime_type.split("/")[-1] if mime_type else "png"
                        logger.info(
                            f"Image generated successfully ({len(image_data)} bytes)"
                        )
                        return image_data, fmt

                raise ValueError("No image data in response")

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if is_rate_limit and attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay)
                    continue
                
                logger.error(f"Failed to generate image: {e}")
                raise

    def generate_blog_image(
        self, title: str, content: str, progress_callback: Optional[Any] = None
    ) -> GeneratedBlogImage:
        """
        Main workflow: generate description and image for a blog post.

        Args:
            title: Blog post title
            content: Blog post content (markdown)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratedBlogImage with all metadata and base64 image
        """
        logger.info(f"Starting blog image generation for: {title}")

        if progress_callback:
            progress_callback(
                f"\n{'='*80}\n🖼️  BLOG IMAGE GENERATION\n{'='*80}"
            )

        # Step 1: Generate description
        if progress_callback:
            progress_callback("Generating image description from blog content...")

        description_result = self.generate_image_description(title, content)

        if progress_callback:
            progress_callback(f"  Style: {description_result.style}")
            progress_callback(
                f"  Description: {description_result.description[:80]}..."
            )

        # Step 2: Generate image
        if progress_callback:
            progress_callback("Generating image with Gemini...")

        try:
            image_bytes, fmt = self.generate_image(description_result.description)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            if progress_callback:
                progress_callback(
                    f"  ✓ Image generated ({len(image_base64)} chars base64)"
                )

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            if progress_callback:
                progress_callback(f"  ✗ Image generation failed: {e}")
            raise

        # Create final result
        result = GeneratedBlogImage(
            title=title,
            description=description_result.description,
            alt_text=description_result.alt_text,
            style=description_result.style,
            image_base64=image_base64,
            format=fmt,
        )

        logger.info("Blog image generation complete")
        return result

    def save_to_diagrams_yaml(
        self, blog_image: GeneratedBlogImage, diagrams_path: Path
    ) -> None:
        """
        Save the blog image to the existing diagrams.yaml file or create new one.

        Args:
            blog_image: GeneratedBlogImage object
            diagrams_path: Path to diagrams.yaml file
        """
        logger.info(f"Saving blog image to: {diagrams_path}")

        # Load existing diagrams if file exists
        if diagrams_path.exists():
            with open(diagrams_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {"diagrams": [], "total_diagrams": 0}

        # Add blog image data
        data["blog_image"] = {
            "title": blog_image.title,
            "description": blog_image.description,
            "alt_text": blog_image.alt_text,
            "style": blog_image.style,
            "image_base64": blog_image.image_base64,
            "format": blog_image.format,
        }

        # Ensure parent directory exists
        diagrams_path.parent.mkdir(parents=True, exist_ok=True)

        # Save updated data
        with open(diagrams_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"Blog image saved to {diagrams_path}")
