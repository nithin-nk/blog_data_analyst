"""
Tools module - Utility functions for web search, content extraction, and processing.

This module provides async functions for:
- Web search via DuckDuckGo
- URL content fetching via trafilatura
- Content chunking for LLM context
- Originality checking
- Mermaid diagram rendering
"""

import asyncio
import base64
import logging
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any

import httpx
import trafilatura
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Thread pool for running sync operations
_executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# Search Functions
# =============================================================================


async def search_duckduckgo(
    query: str,
    max_results: int = 5,
    retry_count: int = 2,
) -> list[dict[str, str]]:
    """
    Search DuckDuckGo and return results.

    Args:
        query: Search query string
        max_results: Maximum results to return (default 5)
        retry_count: Number of retries on failure (default 2)

    Returns:
        List of search results with keys: title, url, snippet

    Example:
        results = await search_duckduckgo("Python asyncio tutorial")
        for r in results:
            print(f"{r['title']}: {r['url']}")
    """

    def _search_sync() -> list[dict[str, str]]:
        """Synchronous search function to run in thread pool."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            raise

    backoff_times = [2, 5]  # seconds

    for attempt in range(retry_count + 1):
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(_executor, _search_sync)
            return results
        except Exception as e:
            if attempt < retry_count:
                wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                logger.info(
                    f"Search attempt {attempt + 1} failed, "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Search failed after {retry_count + 1} attempts: {e}")
                return []  # Return empty list instead of raising

    return []


# =============================================================================
# Content Fetching
# =============================================================================


async def fetch_url_content(
    url: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Fetch and extract content from URL using trafilatura.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (default 30)

    Returns:
        Dictionary with:
            - url: Original URL
            - title: Page title
            - content: Extracted text content
            - success: Whether extraction succeeded
            - error: Error message if failed
            - tokens_estimate: Approximate token count

    Example:
        result = await fetch_url_content("https://example.com/article")
        if result["success"]:
            print(f"Content: {result['content'][:100]}...")
    """
    result = {
        "url": url,
        "title": "",
        "content": "",
        "success": False,
        "error": None,
        "tokens_estimate": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        # Extract title from HTML
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # Extract content using trafilatura (run in thread pool as it's sync)
        def _extract() -> str | None:
            return trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
            )

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(_executor, _extract)

        if content:
            result["content"] = content
            result["success"] = True
            result["tokens_estimate"] = len(content) // 4
        else:
            result["error"] = "No content extracted"

    except httpx.TimeoutException:
        result["error"] = f"Timeout after {timeout}s"
        logger.warning(f"Timeout fetching {url}")
    except httpx.HTTPStatusError as e:
        result["error"] = f"HTTP {e.response.status_code}"
        logger.warning(f"HTTP error fetching {url}: {e}")
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Error fetching {url}: {e}")

    return result


# =============================================================================
# Content Processing
# =============================================================================


def chunk_content(
    content: str,
    max_tokens: int = 4000,
) -> list[str]:
    """
    Split long content into chunks for LLM context.

    Uses approximately 4 characters per token estimation.

    Args:
        content: Text content to chunk
        max_tokens: Maximum tokens per chunk (default 4000)

    Returns:
        List of content chunks

    Example:
        chunks = chunk_content(long_article, max_tokens=2000)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {len(chunk)} chars")
    """
    if not content or not content.strip():
        return []

    max_chars = max_tokens * 4

    # If content fits in one chunk, return as-is
    if len(content) <= max_chars:
        return [content]

    chunks = []
    current_chunk = ""

    # Split by paragraphs first
    paragraphs = content.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds limit, save current chunk
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # If single paragraph is too long, split by sentences or force split
            if len(para) > max_chars:
                sentences = _split_into_sentences(para)

                # If no sentences found (e.g., no punctuation), force split by words
                if len(sentences) <= 1:
                    words = para.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            current_chunk += " " + word if current_chunk else word
                else:
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting - handles common cases
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# Originality Checking
# =============================================================================


def check_originality(
    content: str,
    sources: list[dict[str, str]],
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Check for similarity between content and source texts.

    Args:
        content: Written content to check
        sources: List of source documents with "url" and "content" keys
        threshold: Similarity threshold (default 0.7 = 70%)

    Returns:
        List of flagged items with:
            - sentence: The flagged sentence
            - similar_to: URL of similar source
            - similarity: Similarity score (0-1)

    Example:
        flags = check_originality(draft, sources, threshold=0.6)
        for f in flags:
            print(f"Possible copy: {f['sentence'][:50]}...")
    """
    if not content or not sources:
        return []

    flagged = []
    sentences = _split_into_sentences(content)

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip short sentences
        if len(sentence) < 20:
            continue

        for source in sources:
            source_content = source.get("content", "")
            source_url = source.get("url", "unknown")

            if not source_content:
                continue

            # Check overall similarity
            similarity = SequenceMatcher(
                None,
                sentence.lower(),
                source_content.lower()
            ).ratio()

            if similarity > threshold:
                flagged.append({
                    "sentence": sentence,
                    "similar_to": source_url,
                    "similarity": round(similarity, 3),
                })
                break  # Only flag once per sentence

            # Also check for exact phrase matches (n-gram)
            if _has_ngram_match(sentence, source_content, n=4):
                flagged.append({
                    "sentence": sentence,
                    "similar_to": source_url,
                    "similarity": 0.8,  # Estimated for n-gram match
                })
                break

    return flagged


def _has_ngram_match(text: str, source: str, n: int = 4) -> bool:
    """Check if text has n-gram overlap with source."""
    text_words = text.lower().split()
    source_words = source.lower().split()

    if len(text_words) < n or len(source_words) < n:
        return False

    # Generate n-grams from text
    text_ngrams = set()
    for i in range(len(text_words) - n + 1):
        ngram = " ".join(text_words[i:i + n])
        text_ngrams.add(ngram)

    # Check if any n-gram exists in source
    for i in range(len(source_words) - n + 1):
        ngram = " ".join(source_words[i:i + n])
        if ngram in text_ngrams:
            return True

    return False


# =============================================================================
# Mermaid Rendering
# =============================================================================


async def render_mermaid(
    mermaid_code: str,
    output_path: str,
    retry_count: int = 2,
) -> str | None:
    """
    Render mermaid diagram to PNG using kroki.io.

    Args:
        mermaid_code: Mermaid diagram source code
        output_path: Path to save the PNG file
        retry_count: Number of retries on failure (default 2)

    Returns:
        Path to saved file, or None if rendering failed

    Example:
        diagram = '''
        graph TD
            A[Start] --> B[Process]
            B --> C[End]
        '''
        path = await render_mermaid(diagram, "output/diagram.png")
    """
    # Encode mermaid for kroki URL
    # kroki expects: base64(zlib_compress(diagram))
    compressed = zlib.compress(mermaid_code.encode("utf-8"), level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

    url = f"https://kroki.io/mermaid/png/{encoded}"

    for attempt in range(retry_count + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Save PNG to file
                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Mermaid diagram saved to {output_path}")
                return output_path

        except Exception as e:
            if attempt < retry_count:
                logger.warning(
                    f"Mermaid render attempt {attempt + 1} failed: {e}"
                )
                await asyncio.sleep(1)
            else:
                logger.error(
                    f"Mermaid render failed after {retry_count + 1} attempts: {e}"
                )
                return None

    return None
