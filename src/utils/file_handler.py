"""
File handler for I/O operations.

Handles reading, writing, and managing blog files safely.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

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
    def read_json(file_path: Path) -> dict[str, Any]:
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
    def write_json(file_path: Path, data: dict[str, Any], indent: int = 2) -> None:
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
    def read_yaml(file_path: Path) -> dict[str, Any]:
        """
        Read YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed YAML data
        """
        logger.debug(f"Reading YAML file: {file_path}")
        
        content = FileHandler.read_file(file_path)
        data = yaml.safe_load(content)
        
        return data if data else {}
    
    @staticmethod
    def write_yaml(file_path: Path, data: dict[str, Any]) -> None:
        """
        Write data to YAML file.
        
        Args:
            file_path: Path to YAML file
            data: Data to write
        """
        logger.debug(f"Writing YAML file: {file_path}")
        
        content = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
        FileHandler.write_file(file_path, content)
    
    @staticmethod
    def slugify(text: str, max_length: int = 50) -> str:
        """
        Convert text to a URL/filesystem-safe slug.
        
        Args:
            text: Text to slugify
            max_length: Maximum length of the slug
            
        Returns:
            Slugified string
        """
        # Convert to lowercase and replace spaces/special chars with underscores
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "_", slug)
        slug = slug.strip("_")
        return slug[:max_length]

    @staticmethod
    def create_blog_structure(blog_name: str, output_dir: Path) -> dict[str, Path]:
        """
        Create directory structure for a blog post.
        
        Args:
            blog_name: Name/slug of the blog
            output_dir: Base output directory
            
        Returns:
            Dict with paths for drafts, final, images, research, metadata
        """
        logger.info(f"Creating blog structure for: {blog_name}")
        
        safe_name = FileHandler.slugify(blog_name)
        blog_dir = output_dir / safe_name
        
        paths = {
            "blog_dir": blog_dir,
            "research_dir": blog_dir / "research",
            "research_questions": blog_dir / "research" / "research_questions.yaml",
            "search_results": blog_dir / "research" / "search_results.yaml",
            "extracted_content": blog_dir / "research" / "extracted_content.yaml",
            "draft": blog_dir / "drafts" / f"{safe_name}.md",
            "final_md": blog_dir / "final" / f"{safe_name}.md",
            "final_html": blog_dir / "final" / f"{safe_name}.html",
            "image": blog_dir / "images" / f"{safe_name}_header.png",
            "metadata": blog_dir / f"{safe_name}_metadata.yaml",
        }
        
        # Create directories
        for key, path in paths.items():
            if key.endswith("_dir"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Blog structure created at: {blog_dir}")
        return paths

    @staticmethod
    def save_research_questions(
        file_path: Path,
        topic: str,
        context: str | None,
        questions: list[str],
        categories: list[str],
    ) -> None:
        """
        Save research questions to YAML file.
        
        Args:
            file_path: Path to save the YAML file
            topic: Research topic
            context: Optional context provided
            questions: List of generated questions
            categories: Categories covered
        """
        data = {
            "topic": topic,
            "context": context,
            "generated_at": datetime.now().isoformat(),
            "questions": questions,
            "categories_covered": categories,
            "total_questions": len(questions),
        }
        
        FileHandler.write_yaml(file_path, data)
        logger.info(f"Saved {len(questions)} research questions to {file_path}")

    @staticmethod
    def save_search_results(
        file_path: Path,
        topic: str,
        query_results: list[dict[str, Any]],
        all_urls: list[str],
        url_to_queries: dict[str, list[str]],
        stats: dict[str, int],
    ) -> None:
        """
        Save search results to YAML file.
        
        Args:
            file_path: Path to save the YAML file
            topic: Research topic
            query_results: List of results per query
            all_urls: Deduplicated list of all URLs
            url_to_queries: Mapping of URLs to queries
            stats: Search statistics
        """
        data = {
            "topic": topic,
            "searched_at": datetime.now().isoformat(),
            "statistics": {
                "total_queries": stats.get("total_queries", 0),
                "successful_queries": stats.get("successful_queries", 0),
                "failed_queries": stats.get("failed_queries", 0),
                "total_results": stats.get("total_results", 0),
                "unique_urls": len(all_urls),
            },
            "unique_urls": all_urls,
            "url_sources": url_to_queries,
            "results_by_query": query_results,
        }
        
        FileHandler.write_yaml(file_path, data)
        logger.info(f"Saved search results ({len(all_urls)} unique URLs) to {file_path}")

    @staticmethod
    def save_extracted_content(
        file_path: Path,
        topic: str,
        contents: list[dict[str, Any]],
        statistics: dict[str, int],
    ) -> None:
        """
        Save extracted content to YAML file.

        Args:
            file_path: Path to save the YAML file
            topic: Research topic
            contents: List of extracted content dicts
            statistics: Extraction statistics
        """
        data = {
            "topic": topic,
            "extracted_at": datetime.now().isoformat(),
            "statistics": statistics,
            "contents": contents,
        }

        FileHandler.write_yaml(file_path, data)
        successful = statistics.get("successful", 0)
        total = statistics.get("total_urls", 0)
        logger.info(
            f"Saved extracted content ({successful}/{total} successful) to {file_path}"
        )

    @staticmethod
    def save_blog_output(
        paths: dict[str, Path],
        draft_content: str,
        final_content: str,
        html_content: str,
        metadata: dict[str, Any],
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
        FileHandler.write_yaml(paths["metadata"], metadata)
        
        logger.info("All blog outputs saved successfully")
