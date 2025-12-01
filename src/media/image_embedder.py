"""
Image Embedder module.

Embeds generated images (blog cover and mermaid diagrams) into markdown content
at appropriate locations using LLM-guided placement.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call


logger = get_logger(__name__)


class ImagePlacement(BaseModel):
    """Structured output for image placement decision."""

    heading: str = Field(
        description="The exact heading text after which the image should be placed"
    )
    placement: str = Field(
        description="Where to place relative to heading: 'after_heading' or 'end_of_section'"
    )
    reasoning: str = Field(description="Brief explanation of why this location was chosen")


class DiagramPlacements(BaseModel):
    """Container for multiple diagram placements."""

    placements: List[ImagePlacement] = Field(
        description="List of placement decisions for each diagram"
    )


class ImageEmbedder:
    """
    Embeds images into markdown content at appropriate locations.

    Handles:
    - Blog cover image (placed after the title/H1)
    - Mermaid diagrams (placed under their corresponding headings)
    """

    def __init__(self):
        self.settings = get_settings()

    def load_diagrams_yaml(self, diagrams_path: Path) -> Dict[str, Any]:
        """
        Load diagrams and blog image data from YAML file.

        Args:
            diagrams_path: Path to diagrams.yaml

        Returns:
            Dict containing diagrams and blog_image data
        """
        if not diagrams_path.exists():
            logger.warning(f"Diagrams file not found: {diagrams_path}")
            return {"diagrams": [], "blog_image": None}

        with open(diagrams_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return data

    def _create_base64_image_markdown(
        self, base64_data: str, alt_text: str, fmt: str = "png"
    ) -> str:
        """
        Create markdown image tag with base64 data URI.

        Args:
            base64_data: Base64 encoded image data
            alt_text: Alt text for the image
            fmt: Image format (png, jpeg, etc.)

        Returns:
            Markdown image string
        """
        return f"![{alt_text}](data:image/{fmt};base64,{base64_data})"

    def _find_heading_line(self, content: str, heading: str) -> Optional[int]:
        """
        Find the line number of a heading in the content.

        Args:
            content: Markdown content
            heading: Heading text to find

        Returns:
            Line number (0-indexed) or None if not found
        """
        lines = content.split("\n")
        heading_lower = heading.lower().strip()

        for i, line in enumerate(lines):
            # Check for H2 or H3 headings
            if line.startswith("## ") or line.startswith("### "):
                line_heading = line.lstrip("#").strip().lower()
                if heading_lower in line_heading or line_heading in heading_lower:
                    return i

        return None

    def _find_title_line(self, content: str) -> int:
        """
        Find the line number of the blog title (H1).

        Args:
            content: Markdown content

        Returns:
            Line number (0-indexed) of the H1, or 0 if not found
        """
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if line.startswith("# ") and not line.startswith("## "):
                return i

        return 0

    def _find_section_end(self, content: str, heading_line: int) -> int:
        """
        Find the end of a section (line before next heading or end of content).

        Args:
            content: Markdown content
            heading_line: Line number of the section heading

        Returns:
            Line number where section ends
        """
        lines = content.split("\n")

        for i in range(heading_line + 1, len(lines)):
            if lines[i].startswith("## ") or lines[i].startswith("# "):
                return i - 1

        return len(lines) - 1

    def embed_cover_image(self, content: str, blog_image: Dict[str, Any]) -> str:
        """
        Embed the blog cover image after the title.

        Args:
            content: Markdown content
            blog_image: Blog image data from diagrams.yaml

        Returns:
            Updated markdown content with embedded cover image
        """
        if not blog_image or not blog_image.get("image_base64"):
            logger.info("No blog cover image to embed")
            return content

        logger.info("Embedding blog cover image after title")

        # Create image markdown
        image_md = self._create_base64_image_markdown(
            base64_data=blog_image["image_base64"],
            alt_text=blog_image.get("alt_text", "Blog cover image"),
            fmt=blog_image.get("format", "png"),
        )

        # Find title line and insert after it
        lines = content.split("\n")
        title_line = self._find_title_line(content)

        # Insert image after title with blank lines for formatting
        lines.insert(title_line + 1, "")
        lines.insert(title_line + 2, image_md)
        lines.insert(title_line + 3, "")

        return "\n".join(lines)

    def _get_diagram_placements_from_llm(
        self, content: str, diagrams: List[Dict[str, Any]]
    ) -> List[ImagePlacement]:
        """
        Use LLM to determine optimal placement for diagrams.

        Args:
            content: Markdown content
            diagrams: List of diagram data

        Returns:
            List of ImagePlacement decisions
        """
        if not diagrams:
            return []

        # Build diagram info for prompt
        diagram_info = "\n".join(
            [
                f"- Diagram {i+1}: Heading='{d['heading']}', Type={d['diagram_type']}, Description='{d['description']}'"
                for i, d in enumerate(diagrams)
            ]
        )

        # Extract headings from content
        lines = content.split("\n")
        headings = [
            line.lstrip("#").strip()
            for line in lines
            if line.startswith("## ") or line.startswith("### ")
        ]
        headings_text = "\n".join([f"- {h}" for h in headings])

        prompt = f"""
You are an expert content editor. Analyze this blog post and determine the best placement for each diagram.

BLOG CONTENT HEADINGS:
{headings_text}

DIAGRAMS TO PLACE:
{diagram_info}

PLACEMENT RULES:
1. Each diagram should be placed directly after its corresponding heading
2. Match the diagram's 'heading' field to the closest matching heading in the content
3. If exact match not found, find the most semantically similar heading
4. Place diagrams to enhance understanding of the section content

For each diagram, specify:
- The exact heading text it should be placed after
- Whether to place 'after_heading' (immediately after) or 'end_of_section' (before next heading)
- Brief reasoning for the placement

Return placements for all {len(diagrams)} diagrams.
"""

        messages = [HumanMessage(content=prompt)]

        try:
            result = gemini_llm_call(
                messages,
                model_name=self.settings.image_embedder_model,
                settings=self.settings,
                structured_output=DiagramPlacements,
            )

            logger.info(f"LLM determined placements for {len(result.placements)} diagrams")
            return result.placements

        except Exception as e:
            logger.warning(f"LLM placement failed: {e}. Using default placements.")
            # Fallback: use diagram's own heading field
            return [
                ImagePlacement(
                    heading=d["heading"],
                    placement="after_heading",
                    reasoning="Default placement based on diagram heading",
                )
                for d in diagrams
            ]

    def embed_diagrams(
        self, content: str, diagrams: List[Dict[str, Any]], use_llm: bool = True
    ) -> str:
        """
        Embed mermaid diagrams into the content.

        Args:
            content: Markdown content
            diagrams: List of diagram data from diagrams.yaml
            use_llm: Whether to use LLM for placement decisions

        Returns:
            Updated markdown content with embedded diagrams
        """
        if not diagrams:
            logger.info("No diagrams to embed")
            return content

        logger.info(f"Embedding {len(diagrams)} diagrams into content")

        # Get placements (from LLM or default)
        if use_llm:
            placements = self._get_diagram_placements_from_llm(content, diagrams)
        else:
            placements = [
                ImagePlacement(
                    heading=d["heading"],
                    placement="after_heading",
                    reasoning="Default placement",
                )
                for d in diagrams
            ]

        # Match placements to diagrams
        placement_map = {p.heading.lower(): p for p in placements}

        # Sort diagrams by their position in the document (reverse order for insertion)
        diagram_positions = []
        for diagram in diagrams:
            heading = diagram["heading"]
            placement = placement_map.get(
                heading.lower(),
                ImagePlacement(
                    heading=heading,
                    placement="after_heading",
                    reasoning="Fallback",
                ),
            )

            # Find the actual heading in content
            line_num = self._find_heading_line(content, placement.heading)
            if line_num is None:
                # Try with original diagram heading
                line_num = self._find_heading_line(content, heading)

            if line_num is not None:
                if placement.placement == "end_of_section":
                    insert_line = self._find_section_end(content, line_num)
                else:
                    insert_line = line_num + 1  # After heading

                diagram_positions.append((insert_line, diagram))
                logger.debug(
                    f"Diagram '{heading}' will be placed at line {insert_line}"
                )
            else:
                logger.warning(
                    f"Could not find heading for diagram: {heading}. Skipping."
                )

        # Sort by position (descending) to insert from bottom to top
        diagram_positions.sort(key=lambda x: x[0], reverse=True)

        # Insert diagrams
        lines = content.split("\n")
        for insert_line, diagram in diagram_positions:
            if not diagram.get("image_base64"):
                logger.warning(
                    f"Diagram '{diagram['heading']}' has no image_base64. Skipping."
                )
                continue

            # Create image markdown
            image_md = self._create_base64_image_markdown(
                base64_data=diagram["image_base64"],
                alt_text=f"{diagram['diagram_type']} diagram: {diagram['description']}",
                fmt="png",
            )

            # Insert with formatting
            lines.insert(insert_line + 1, "")
            lines.insert(insert_line + 2, image_md)
            lines.insert(insert_line + 3, "")

        return "\n".join(lines)

    def embed_all_images(
        self,
        content: str,
        diagrams_path: Path,
        progress_callback: Optional[Any] = None,
    ) -> str:
        """
        Main workflow: embed all available images into content.

        Args:
            content: Markdown content
            diagrams_path: Path to diagrams.yaml
            progress_callback: Optional callback for progress updates

        Returns:
            Updated markdown content with all images embedded
        """
        logger.info("Starting image embedding process")

        if progress_callback:
            progress_callback(
                f"\n{'='*80}\n🖼️  IMAGE EMBEDDING\n{'='*80}"
            )

        # Load diagrams data
        data = self.load_diagrams_yaml(diagrams_path)
        diagrams = data.get("diagrams", [])
        blog_image = data.get("blog_image")

        images_embedded = 0

        # Embed blog cover image first (after title)
        if blog_image and blog_image.get("image_base64"):
            if progress_callback:
                progress_callback("Embedding blog cover image after title...")

            content = self.embed_cover_image(content, blog_image)
            images_embedded += 1

            if progress_callback:
                progress_callback("  ✓ Blog cover image embedded")

        # Embed diagrams
        diagrams_with_images = [d for d in diagrams if d.get("image_base64")]
        if diagrams_with_images:
            if progress_callback:
                progress_callback(
                    f"Embedding {len(diagrams_with_images)} diagram(s)..."
                )

            content = self.embed_diagrams(content, diagrams_with_images, use_llm=True)
            images_embedded += len(diagrams_with_images)

            if progress_callback:
                progress_callback(
                    f"  ✓ {len(diagrams_with_images)} diagram(s) embedded"
                )

        if progress_callback:
            progress_callback(f"\n✓ Total images embedded: {images_embedded}")

        logger.info(f"Image embedding complete. {images_embedded} images embedded.")
        return content
