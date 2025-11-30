"""
Content extractor using crawl4ai for web scraping.

Extracts clean text, headings, and code blocks from web pages.
Supports concurrent extraction with rate limiting.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ExtractedContent(BaseModel):
    """Extracted content from a single URL."""

    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    snippet: str = Field(default="", description="Search result snippet (preview text)")
    markdown: str = Field(default="", description="Content in Markdown format")
    headings: list[str] = Field(default_factory=list, description="List of headings")
    code_blocks: list[dict[str, str]] = Field(
        default_factory=list, description="List of code snippets with language"
    )
    success: bool = Field(default=False, description="Whether extraction succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Extraction timestamp",
    )
    source_queries: list[str] = Field(
        default_factory=list, description="Search queries that led to this URL"
    )


class AggregatedExtractedContent(BaseModel):
    """Aggregated extracted content from multiple URLs."""

    topic: str = Field(description="Research topic")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Extraction timestamp",
    )
    statistics: dict[str, int] = Field(
        default_factory=dict, description="Extraction statistics"
    )
    contents: list[ExtractedContent] = Field(
        default_factory=list, description="Extracted content from each URL"
    )


class ContentExtractor:
    """Extractor for web page content using crawl4ai."""

    def __init__(
        self,
        concurrency_limit: int = 5,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the content extractor.

        Args:
            concurrency_limit: Maximum concurrent extractions (default: 5)
            timeout: Timeout in seconds for each extraction (default: 30)
        """
        self.settings = get_settings()
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self._crawler = None
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        logger.info(
            f"ContentExtractor initialized (concurrency={concurrency_limit}, "
            f"timeout={timeout}s)"
        )

    async def _get_crawler(self):
        """Lazy initialization of the crawler."""
        if self._crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler, BrowserConfig

                browser_config = BrowserConfig(
                    headless=True,
                    verbose=False,
                )
                self._crawler = AsyncWebCrawler(config=browser_config)
                await self._crawler.__aenter__()
                logger.debug("AsyncWebCrawler initialized for content extraction")
            except ImportError as e:
                logger.error(f"crawl4ai not installed: {e}")
                raise ImportError(
                    "crawl4ai is required. Install with: pip install crawl4ai"
                ) from e
        return self._crawler

    async def _close_crawler(self) -> None:
        """Close the crawler if it's open."""
        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing crawler: {e}")
            finally:
                self._crawler = None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try <title> tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try <h1> as fallback
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        # Try og:title meta tag
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        return ""

    def _extract_headings(self, soup: BeautifulSoup) -> list[str]:
        """Extract all headings (h1-h6) from HTML."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text and len(text) > 2:  # Skip very short headings
                    headings.append(f"H{level}: {text}")
        return headings

    def _extract_code_blocks(self, soup: BeautifulSoup) -> list[dict[str, str]]:
        """Extract code blocks from HTML."""
        code_blocks = []

        # Find <pre><code> blocks (most common pattern)
        for pre in soup.find_all("pre"):
            code = pre.find("code")
            if code:
                code_text = code.get_text()
            else:
                code_text = pre.get_text()

            if code_text and len(code_text.strip()) > 10:  # Skip very short snippets
                # Try to detect language from class
                language = "unknown"
                code_elem = code if code else pre
                classes = code_elem.get("class", [])
                for cls in classes:
                    if isinstance(cls, str):
                        # Common patterns: language-python, lang-python, python
                        match = re.match(r"(?:language-|lang-)?(\w+)", cls.lower())
                        if match:
                            lang = match.group(1)
                            if lang not in ("hljs", "highlight", "code", "pre"):
                                language = lang
                                break

                code_blocks.append({
                    "language": language,
                    "code": code_text.strip()[:2000],  # Limit code length
                })

        # Also find standalone <code> blocks (inline code usually)
        for code in soup.find_all("code"):
            # Skip if already processed inside <pre>
            if code.parent and code.parent.name == "pre":
                continue

            code_text = code.get_text()
            # Only include substantial inline code blocks
            if code_text and len(code_text.strip()) > 50 and "\n" in code_text:
                code_blocks.append({
                    "language": "unknown",
                    "code": code_text.strip()[:2000],
                })

        return code_blocks[:20]  # Limit to 20 code blocks per page

    def _clean_markdown(self, markdown: str) -> str:
        """Clean and normalize markdown content."""
        if not markdown:
            return ""

        # Remove excessive blank lines (more than 2 consecutive)
        lines = markdown.split("\n")
        cleaned_lines = []
        blank_count = 0

        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)

        markdown = "\n".join(cleaned_lines).strip()

        # Limit content length
        if len(markdown) > 20000:
            markdown = markdown[:20000] + "\n\n[Content truncated...]"

        return markdown

    async def extract(
        self,
        url: str,
        source_queries: Optional[list[str]] = None,
        snippet: str = "",
    ) -> ExtractedContent:
        """
        Extract content from a single URL.

        Args:
            url: The URL to extract content from
            source_queries: List of search queries that led to this URL
            snippet: Search result snippet (preview text)

        Returns:
            ExtractedContent with extracted data
        """
        logger.info(f"Extracting content from: {url}")

        try:
            from crawl4ai import CrawlerRunConfig, CacheMode

            async with self._semaphore:
                crawler = await self._get_crawler()

                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                )

                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_config),
                    timeout=self.timeout,
                )

                if not result.success:
                    logger.warning(f"Crawl failed for URL: {url}")
                    return ExtractedContent(
                        url=url,
                        snippet=snippet,
                        success=False,
                        error="Crawl request failed",
                        source_queries=source_queries or [],
                    )

                # Parse HTML for structured data
                soup = BeautifulSoup(result.html, "html.parser")

                # Extract structured components
                title = self._extract_title(soup)
                headings = self._extract_headings(soup)
                code_blocks = self._extract_code_blocks(soup)

                # Use crawl4ai's built-in markdown conversion
                markdown_content = self._clean_markdown(result.markdown or "")

                logger.info(
                    f"Extracted from {url}: {len(markdown_content)} chars markdown, "
                    f"{len(headings)} headings, {len(code_blocks)} code blocks"
                )

                return ExtractedContent(
                    url=url,
                    title=title,
                    snippet=snippet,
                    markdown=markdown_content,
                    headings=headings,
                    code_blocks=code_blocks,
                    success=True,
                    source_queries=source_queries or [],
                )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting content from: {url}")
            return ExtractedContent(
                url=url,
                snippet=snippet,
                success=False,
                error=f"Timeout after {self.timeout}s",
                source_queries=source_queries or [],
            )
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return ExtractedContent(
                url=url,
                snippet=snippet,
                success=False,
                error=str(e),
                source_queries=source_queries or [],
            )

    async def extract_multiple(
        self,
        urls: list[str],
        url_to_queries: Optional[dict[str, list[str]]] = None,
        url_to_snippet: Optional[dict[str, str]] = None,
        topic: str = "",
    ) -> AggregatedExtractedContent:
        """
        Extract content from multiple URLs concurrently.

        Args:
            urls: List of URLs to extract content from
            url_to_queries: Optional mapping of URLs to search queries
            url_to_snippet: Optional mapping of URLs to longest snippet
            topic: Research topic for metadata

        Returns:
            AggregatedExtractedContent with all results
        """
        logger.info(f"Extracting content from {len(urls)} URLs (concurrency={self.concurrency_limit})")

        url_to_queries = url_to_queries or {}
        url_to_snippet = url_to_snippet or {}

        try:
            # Create extraction tasks
            tasks = [
                self.extract(
                    url,
                    source_queries=url_to_queries.get(url, []),
                    snippet=url_to_snippet.get(url, ""),
                )
                for url in urls
            ]

            # Run concurrently (semaphore limits actual concurrency)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            contents: list[ExtractedContent] = []
            successful = 0
            failed = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Extraction task failed for {urls[i]}: {result}")
                    contents.append(ExtractedContent(
                        url=urls[i],
                        snippet=url_to_snippet.get(urls[i], ""),
                        success=False,
                        error=str(result),
                        source_queries=url_to_queries.get(urls[i], []),
                    ))
                    failed += 1
                else:
                    contents.append(result)
                    if result.success:
                        successful += 1
                    else:
                        failed += 1

            logger.info(
                f"Extraction complete: {successful}/{len(urls)} successful, "
                f"{failed} failed"
            )

            return AggregatedExtractedContent(
                topic=topic,
                statistics={
                    "total_urls": len(urls),
                    "successful": successful,
                    "failed": failed,
                },
                contents=contents,
            )

        finally:
            await self._close_crawler()

    def extract_sync(
        self,
        url: str,
        source_queries: Optional[list[str]] = None,
    ) -> ExtractedContent:
        """
        Synchronous wrapper for extract().

        Args:
            url: The URL to extract content from
            source_queries: List of search queries that led to this URL

        Returns:
            ExtractedContent with extracted data
        """
        return asyncio.run(self._extract_and_close(url, source_queries))

    async def _extract_and_close(
        self,
        url: str,
        source_queries: Optional[list[str]] = None,
    ) -> ExtractedContent:
        """Execute extraction and close crawler."""
        try:
            return await self.extract(url, source_queries)
        finally:
            await self._close_crawler()

    def extract_multiple_sync(
        self,
        urls: list[str],
        url_to_queries: Optional[dict[str, list[str]]] = None,
        url_to_snippet: Optional[dict[str, str]] = None,
        topic: str = "",
    ) -> AggregatedExtractedContent:
        """
        Synchronous wrapper for extract_multiple().

        Args:
            urls: List of URLs to extract content from
            url_to_queries: Optional mapping of URLs to search queries
            url_to_snippet: Optional mapping of URLs to longest snippet
            topic: Research topic for metadata

        Returns:
            AggregatedExtractedContent with all results
        """
        return asyncio.run(self.extract_multiple(urls, url_to_queries, url_to_snippet, topic))

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()


async def main():
    """Demo usage of ContentExtractor."""
    extractor = ContentExtractor(concurrency_limit=3, timeout=30)

    # Single extraction demo
    print("\n" + "=" * 60)
    print("Single Extraction Demo")
    print("=" * 60)

    url = "https://docs.mem0.ai/open-source/python-quickstart"
    result = await extractor._extract_and_close(url)

    print(f"\nURL: {result.url}")
    print(f"Success: {result.success}")
    print(f"Title: {result.title}")
    print(f"Markdown length: {len(result.markdown)} chars")
    print(f"Headings: {len(result.headings)}")
    for h in result.headings[:5]:
        print(f"  - {h}")
    print(f"Code blocks: {len(result.code_blocks)}")
    for cb in result.code_blocks[:2]:
        print(f"  - {cb['language']}: {cb['code'][:100]}...")
    print(f"\nMarkdown preview:\n{result.markdown[:500]}...")

    # Multiple extraction demo
    print("\n" + "=" * 60)
    print("Multiple Extraction Demo")
    print("=" * 60)

    urls = [
        "https://github.com/mem0ai/mem0",
        "https://docs.mem0.ai/quickstart",
        "https://pypi.org/project/mem0ai/",
    ]

    url_to_queries = {
        "https://github.com/mem0ai/mem0": ["mem0 AI agent memory"],
        "https://docs.mem0.ai/quickstart": ["mem0 python tutorial"],
        "https://pypi.org/project/mem0ai/": ["mem0 python package"],
    }

    extractor2 = ContentExtractor(concurrency_limit=3)
    results = await extractor2.extract_multiple(
        urls, url_to_queries=url_to_queries, topic="mem0 AI memory"
    )

    print(f"\nTopic: {results.topic}")
    print(f"Statistics: {results.statistics}")
    for content in results.contents:
        print(f"\n  URL: {content.url}")
        print(f"  Success: {content.success}")
        print(f"  Title: {content.title[:60]}..." if len(content.title) > 60 else f"  Title: {content.title}")
        print(f"  Markdown: {len(content.markdown)} chars")
        print(f"  Source queries: {content.source_queries}")


if __name__ == "__main__":
    asyncio.run(main())
