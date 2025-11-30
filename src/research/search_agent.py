"""
Search agent using crawl4ai for DuckDuckGo searches.

Executes web searches and extracts top URLs for research.
"""

import asyncio
from typing import Optional
from urllib.parse import quote_plus, urlparse, parse_qs

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """A single search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(default="", description="Text snippet from the result")


class QuerySearchResults(BaseModel):
    """Search results for a single query."""

    query: str = Field(description="The search query")
    results: list[SearchResult] = Field(default_factory=list)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)


class AggregatedSearchResults(BaseModel):
    """Aggregated search results from multiple queries."""

    queries: list[QuerySearchResults] = Field(default_factory=list)
    all_urls: list[str] = Field(
        default_factory=list, description="Deduplicated list of all URLs"
    )
    url_to_queries: dict[str, list[str]] = Field(
        default_factory=dict, description="Mapping of URLs to queries that found them"
    )
    total_results: int = Field(default=0)
    successful_queries: int = Field(default=0)
    failed_queries: int = Field(default=0)


class SearchAgent:
    """Agent for executing DuckDuckGo searches using crawl4ai."""

    DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        results_per_query: int = 3,
        rate_limit_delay: float = 1.0,
    ) -> None:
        """
        Initialize the search agent.

        Args:
            results_per_query: Number of top results to extract per query (default: 3)
            rate_limit_delay: Delay in seconds between requests (default: 1.0)
        """
        self.settings = get_settings()
        self.results_per_query = results_per_query
        self.rate_limit_delay = rate_limit_delay
        self._crawler = None
        logger.info(
            f"SearchAgent initialized (results_per_query={results_per_query}, "
            f"rate_limit_delay={rate_limit_delay}s)"
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
                logger.debug("AsyncWebCrawler initialized")
            except ImportError as e:
                logger.error(f"crawl4ai not installed: {e}")
                raise ImportError(
                    "crawl4ai is required for search. Install with: pip install crawl4ai"
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

    def _build_search_url(self, query: str) -> str:
        """
        Build DuckDuckGo search URL.

        Args:
            query: Search query string

        Returns:
            DuckDuckGo HTML search URL
        """
        encoded_query = quote_plus(query)
        return f"{self.DUCKDUCKGO_URL}?q={encoded_query}"

    def _extract_real_url(self, ddg_url: str) -> str:
        """
        Extract the real URL from DuckDuckGo's redirect URL.

        DuckDuckGo wraps URLs in redirects like:
        //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com...

        Args:
            ddg_url: The DuckDuckGo redirect URL

        Returns:
            The actual destination URL
        """
        if not ddg_url:
            return ""

        # Handle protocol-relative URLs
        if ddg_url.startswith("//"):
            ddg_url = "https:" + ddg_url

        # Check if it's a DuckDuckGo redirect
        if "duckduckgo.com/l/" in ddg_url:
            parsed = urlparse(ddg_url)
            params = parse_qs(parsed.query)
            if "uddg" in params:
                return params["uddg"][0]

        return ddg_url

    def _parse_search_results(self, html: str, query: str) -> list[SearchResult]:
        """
        Parse DuckDuckGo HTML search results.

        Args:
            html: Raw HTML from DuckDuckGo
            query: Original search query (for logging)

        Returns:
            List of SearchResult objects
        """
        results = []
        soup = BeautifulSoup(html, "html.parser")

        # DuckDuckGo HTML version uses specific classes
        # Try multiple selectors for robustness
        result_links = soup.select("a.result__a")

        if not result_links:
            # Fallback: try other common selectors
            result_links = soup.select(".result__title a")

        if not result_links:
            # Another fallback
            result_links = soup.select(".results_links_deep a.result__url")

        logger.debug(f"Found {len(result_links)} raw results for query: {query}")

        seen_urls = set()
        for link in result_links:
            if len(results) >= self.results_per_query:
                break

            title = link.get_text(strip=True)
            raw_url = link.get("href", "")
            url = self._extract_real_url(raw_url)

            # Skip invalid or duplicate URLs
            if not url or not url.startswith("http"):
                continue
            if url in seen_urls:
                continue

            # Skip DuckDuckGo internal links
            if "duckduckgo.com" in url:
                continue

            seen_urls.add(url)

            # Try to get snippet from sibling elements
            snippet = ""
            parent = link.find_parent(class_="result")
            if parent:
                snippet_elem = parent.select_one(".result__snippet")
                if snippet_elem:
                    snippet = snippet_elem.get_text(strip=True)

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                )
            )

        logger.debug(f"Extracted {len(results)} valid results for query: {query}")
        return results

    async def search(self, query: str) -> QuerySearchResults:
        """
        Execute a DuckDuckGo search and extract top results.

        Args:
            query: Search query string

        Returns:
            QuerySearchResults with title, url, and snippet for each result
        """
        logger.info(f"Searching DuckDuckGo for: {query}")

        try:
            from crawl4ai import CrawlerRunConfig, CacheMode

            crawler = await self._get_crawler()
            search_url = self._build_search_url(query)

            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,  # Always get fresh results
            )

            result = await crawler.arun(url=search_url, config=run_config)

            if not result.success:
                logger.warning(f"Crawl failed for query: {query}")
                return QuerySearchResults(
                    query=query,
                    results=[],
                    success=False,
                    error="Crawl request failed",
                )

            results = self._parse_search_results(result.html, query)

            logger.info(f"Found {len(results)} results for: {query}")
            return QuerySearchResults(
                query=query,
                results=results,
                success=True,
            )

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return QuerySearchResults(
                query=query,
                results=[],
                success=False,
                error=str(e),
            )

    async def search_multiple(
        self,
        queries: list[str],
        deduplicate: bool = True,
    ) -> AggregatedSearchResults:
        """
        Execute multiple searches with rate limiting and deduplication.

        Args:
            queries: List of search query strings
            deduplicate: Whether to deduplicate URLs across queries (default: True)

        Returns:
            AggregatedSearchResults with all results and deduplication info
        """
        logger.info(f"Executing {len(queries)} searches")

        query_results: list[QuerySearchResults] = []
        url_to_queries: dict[str, list[str]] = {}
        all_urls: list[str] = []

        try:
            for i, query in enumerate(queries):
                # Rate limiting between requests
                if i > 0:
                    logger.debug(f"Rate limiting: waiting {self.rate_limit_delay}s")
                    await asyncio.sleep(self.rate_limit_delay)

                result = await self.search(query)
                query_results.append(result)

                # Track URL to query mapping for citations
                for search_result in result.results:
                    url = search_result.url
                    if url not in url_to_queries:
                        url_to_queries[url] = []
                        if deduplicate:
                            all_urls.append(url)
                    url_to_queries[url].append(query)

                    if not deduplicate and url not in all_urls:
                        all_urls.append(url)

        finally:
            # Clean up crawler
            await self._close_crawler()

        successful = sum(1 for r in query_results if r.success)
        failed = len(query_results) - successful
        total_results = sum(len(r.results) for r in query_results)

        logger.info(
            f"Search complete: {successful}/{len(queries)} successful, "
            f"{total_results} total results, {len(all_urls)} unique URLs"
        )

        return AggregatedSearchResults(
            queries=query_results,
            all_urls=all_urls,
            url_to_queries=url_to_queries,
            total_results=total_results,
            successful_queries=successful,
            failed_queries=failed,
        )

    def search_sync(self, query: str) -> QuerySearchResults:
        """
        Synchronous wrapper for search().

        Args:
            query: Search query string

        Returns:
            QuerySearchResults with search results
        """
        return asyncio.run(self._search_and_close(query))

    async def _search_and_close(self, query: str) -> QuerySearchResults:
        """Execute search and close crawler."""
        try:
            return await self.search(query)
        finally:
            await self._close_crawler()

    def search_multiple_sync(
        self,
        queries: list[str],
        deduplicate: bool = True,
    ) -> AggregatedSearchResults:
        """
        Synchronous wrapper for search_multiple().

        Args:
            queries: List of search query strings
            deduplicate: Whether to deduplicate URLs across queries

        Returns:
            AggregatedSearchResults with all results
        """
        return asyncio.run(self.search_multiple(queries, deduplicate))


async def main():
    """Demo usage of SearchAgent."""
    agent = SearchAgent(results_per_query=3, rate_limit_delay=1.5)

    # Single search demo
    print("\n" + "=" * 60)
    print("Single Search Demo")
    print("=" * 60)

    result = await agent._search_and_close("Python asyncio tutorial")
    print(f"\nQuery: {result.query}")
    print(f"Success: {result.success}")
    print(f"Results: {len(result.results)}")
    for i, r in enumerate(result.results, 1):
        print(f"\n  {i}. {r.title}")
        print(f"     URL: {r.url}")
        if r.snippet:
            print(f"     Snippet: {r.snippet[:100]}...")

    # Multiple search demo
    print("\n" + "=" * 60)
    print("Multiple Search Demo")
    print("=" * 60)

    queries = [
        "mem0 AI memory framework",
        "LangChain agent memory",
        "AI agent long term memory",
    ]

    agent2 = SearchAgent(results_per_query=3, rate_limit_delay=1.5)
    results = await agent2.search_multiple(queries, deduplicate=True)

    print(f"\nTotal queries: {len(results.queries)}")
    print(f"Successful: {results.successful_queries}")
    print(f"Failed: {results.failed_queries}")
    print(f"Total results: {results.total_results}")
    print(f"Unique URLs: {len(results.all_urls)}")

    print("\nUnique URLs found:")
    for url in results.all_urls[:10]:
        queries_for_url = results.url_to_queries.get(url, [])
        print(f"  - {url}")
        print(f"    Found via: {queries_for_url}")


if __name__ == "__main__":
    asyncio.run(main())
