"""
File handler for I/O operations.

Handles reading, writing, and managing blog files safely.
"""

from pathlib import Path
from typing import Any, Dict
import json

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class FileHandler:
    """Handler for file I/O operations."""
    
    @staticmethod
    def read_file(file_path: Path) -> str:
        """
        Read file contents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        logger.debug(f"Reading file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content
    
    @staticmethod
    def write_file(file_path: Path, content: str) -> None:
        """
        Write content to file.
        
        Args:
            file_path: Path to the file
            content: Content to write
        """
        logger.debug(f"Writing to file: {file_path}")
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.debug(f"Wrote {len(content)} characters to {file_path}")
    
    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        logger.debug(f"Reading JSON file: {file_path}")
        
        content = FileHandler.read_file(file_path)
        data = json.loads(content)
        
        return data
    
    @staticmethod
    def write_json(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Write data to JSON file.
        
        Args:
            file_path: Path to JSON file
            data: Data to write
            indent: JSON indentation level
        """
        logger.debug(f"Writing JSON file: {file_path}")
        
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        FileHandler.write_file(file_path, content)
    
    @staticmethod
    def create_blog_structure(blog_name: str, output_dir: Path) -> Dict[str, Path]:
        """
        Create directory structure for a blog post.
        
        Args:
            blog_name: Name/slug of the blog
            output_dir: Base output directory
            
        Returns:
            Dict with paths for drafts, final, images, metadata
        """
        logger.info(f"Creating blog structure for: {blog_name}")
        
        # Create safe directory name
        safe_name = "".join(c if c.isalnum() else "_" for c in blog_name.lower())
        safe_name = safe_name[:50]
        
        blog_dir = output_dir / safe_name
        
        paths = {
            "blog_dir": blog_dir,
            "draft": blog_dir / "drafts" / f"{safe_name}.md",
            "final_md": blog_dir / "final" / f"{safe_name}.md",
            "final_html": blog_dir / "final" / f"{safe_name}.html",
            "image": blog_dir / "images" / f"{safe_name}_header.png",
            "metadata": blog_dir / f"{safe_name}_metadata.json",
        }
        
        # Create directories
        for key, path in paths.items():
            if key != "blog_dir":  # Don't try to create parent for blog_dir itself
                path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Blog structure created at: {blog_dir}")
        return paths
    
    @staticmethod
    def save_blog_output(
        paths: Dict[str, Path],
        draft_content: str,
        final_content: str,
        html_content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Save all blog outputs.
        
        Args:
            paths: Dict of file paths
            draft_content: Draft markdown content
            final_content: Final markdown content
            html_content: Final HTML content
            metadata: Blog metadata (scores, tags, etc.)
        """
        logger.info("Saving blog outputs")
        
        FileHandler.write_file(paths["draft"], draft_content)
        FileHandler.write_file(paths["final_md"], final_content)
        FileHandler.write_file(paths["final_html"], html_content)
        FileHandler.write_json(paths["metadata"], metadata)
        
        logger.info("All blog outputs saved successfully")
