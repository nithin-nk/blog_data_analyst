"""Tests for research modules."""

import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock, patch

from bs4 import BeautifulSoup

from src.research.search_agent import (
    SearchAgent,
    SearchResult,
    QuerySearchResults,
    AggregatedSearchResults,
)
from src.research.content_extractor import (
    ContentExtractor,
    ExtractedContent,
    AggregatedExtractedContent,
)


# =============================================================================
# Fixtures for mocking settings
# =============================================================================


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings to avoid reading .env file."""
    mock_settings_obj = MagicMock()
    mock_settings_obj.google_api_key = "test-api-key"
    mock_settings_obj.environment = "test"
    mock_settings_obj.log_level = "DEBUG"

    with patch("src.research.search_agent.get_settings", return_value=mock_settings_obj):
        with patch(
            "src.research.content_extractor.get_settings", return_value=mock_settings_obj
        ):
            yield mock_settings_obj


# =============================================================================
# Sample HTML for mocking DuckDuckGo responses
# =============================================================================

SAMPLE_DDG_HTML = """
<!DOCTYPE html>
<html>
<body>
    <div class="results">
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle1">
                Example Article 1
            </a>
            <div class="result__snippet">
                This is the snippet for the first article about Python.
            </div>
        </div>
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftutorial">
                Tutorial on Programming
            </a>
            <div class="result__snippet">
                A comprehensive tutorial on programming concepts.
            </div>
        </div>
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">
                Python Documentation
            </a>
            <div class="result__snippet">
                Official Python documentation and guides.
            </div>
        </div>
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fmedium.com%2Farticle">
                Medium Article
            </a>
            <div class="result__snippet">
                Another great article from Medium.
            </div>
        </div>
    </div>
</body>
</html>
"""

SAMPLE_DDG_HTML_EMPTY = """
<!DOCTYPE html>
<html>
<body>
    <div class="results">
        <div class="no-results">No results found.</div>
    </div>
</body>
</html>
"""


# =============================================================================
# SearchAgent Tests
# =============================================================================


class TestSearchAgent:
    """Tests for SearchAgent class."""

    def test_initialization_default_values(self):
        """Test SearchAgent initializes with correct default values."""
        agent = SearchAgent()
        assert agent.results_per_query == 3
        assert agent.rate_limit_delay == 1.0
        assert agent._crawler is None

    def test_initialization_custom_values(self):
        """Test SearchAgent initializes with custom values."""
        agent = SearchAgent(results_per_query=5, rate_limit_delay=2.0)
        assert agent.results_per_query == 5
        assert agent.rate_limit_delay == 2.0

    def test_build_search_url(self):
        """Test DuckDuckGo URL building."""
        agent = SearchAgent()

        # Simple query
        url = agent._build_search_url("python tutorial")
        assert url == "https://html.duckduckgo.com/html/?q=python+tutorial"

        # Query with special characters
        url = agent._build_search_url("what is AI?")
        assert "what+is+AI" in url

    def test_extract_real_url_from_ddg_redirect(self):
        """Test extraction of real URL from DuckDuckGo redirect."""
        agent = SearchAgent()

        # Standard DuckDuckGo redirect
        ddg_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage"
        real_url = agent._extract_real_url(ddg_url)
        assert real_url == "https://example.com/page"

        # Direct URL (no redirect)
        direct_url = "https://example.org/direct"
        assert agent._extract_real_url(direct_url) == direct_url

        # Empty URL
        assert agent._extract_real_url("") == ""

    def test_parse_search_results(self):
        """Test HTML parsing extracts correct results."""
        agent = SearchAgent(results_per_query=3)

        results = agent._parse_search_results(SAMPLE_DDG_HTML, "test query")

        assert len(results) == 3
        assert results[0].title == "Example Article 1"
        assert results[0].url == "https://example.com/article1"
        assert "snippet" in results[0].snippet.lower() or results[0].snippet == ""

        assert results[1].url == "https://example.org/tutorial"
        assert results[2].url == "https://docs.python.org/3/"

    def test_parse_search_results_respects_limit(self):
        """Test that parsing respects results_per_query limit."""
        agent = SearchAgent(results_per_query=2)

        results = agent._parse_search_results(SAMPLE_DDG_HTML, "test query")

        assert len(results) == 2

    def test_parse_search_results_empty_html(self):
        """Test parsing empty or no-results HTML."""
        agent = SearchAgent()

        results = agent._parse_search_results(SAMPLE_DDG_HTML_EMPTY, "empty query")

        assert len(results) == 0

    def test_parse_search_results_deduplicates_urls(self):
        """Test that duplicate URLs are removed."""
        agent = SearchAgent()

        # HTML with duplicate URLs
        html_with_dupes = """
        <div class="results">
            <a class="result__a" href="https://example.com">Title 1</a>
            <a class="result__a" href="https://example.com">Title 2</a>
            <a class="result__a" href="https://other.com">Title 3</a>
        </div>
        """

        results = agent._parse_search_results(html_with_dupes, "test")

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs should be removed"


class TestSearchAgentAsync:
    """Async tests for SearchAgent."""

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search execution."""
        agent = SearchAgent(results_per_query=3)

        # Mock the crawler
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = SAMPLE_DDG_HTML

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(agent, "_get_crawler", return_value=mock_crawler):
            result = await agent.search("python asyncio")

        assert result.success is True
        assert result.query == "python asyncio"
        assert len(result.results) == 3
        assert result.error is None

    @pytest.mark.asyncio
    async def test_search_crawl_failure(self):
        """Test handling of crawl failure."""
        agent = SearchAgent()

        mock_result = MagicMock()
        mock_result.success = False

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(agent, "_get_crawler", return_value=mock_crawler):
            result = await agent.search("failing query")

        assert result.success is False
        assert result.error == "Crawl request failed"
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_search_exception_handling(self):
        """Test handling of exceptions during search."""
        agent = SearchAgent()

        with patch.object(
            agent, "_get_crawler", side_effect=Exception("Connection error")
        ):
            result = await agent.search("error query")

        assert result.success is False
        assert "Connection error" in result.error
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_search_multiple_with_deduplication(self):
        """Test multiple searches with URL deduplication."""
        agent = SearchAgent(results_per_query=2, rate_limit_delay=0)

        # Different HTML for each query, but with overlapping URLs
        html1 = """
        <div class="results">
            <a class="result__a" href="https://shared.com/page">Shared Page</a>
            <a class="result__a" href="https://unique1.com">Unique 1</a>
        </div>
        """
        html2 = """
        <div class="results">
            <a class="result__a" href="https://shared.com/page">Shared Page Again</a>
            <a class="result__a" href="https://unique2.com">Unique 2</a>
        </div>
        """

        mock_results = [
            MagicMock(success=True, html=html1),
            MagicMock(success=True, html=html2),
        ]

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(side_effect=mock_results)

        with patch.object(agent, "_get_crawler", return_value=mock_crawler):
            with patch.object(agent, "_close_crawler", new_callable=AsyncMock):
                result = await agent.search_multiple(
                    ["query1", "query2"], deduplicate=True
                )

        assert result.successful_queries == 2
        assert result.failed_queries == 0

        # Should have 3 unique URLs (shared + unique1 + unique2)
        assert len(result.all_urls) == 3

        # Shared URL should map to both queries
        shared_queries = result.url_to_queries.get("https://shared.com/page", [])
        assert len(shared_queries) == 2

    @pytest.mark.asyncio
    async def test_search_multiple_without_deduplication(self):
        """Test multiple searches without URL deduplication."""
        agent = SearchAgent(results_per_query=1, rate_limit_delay=0)

        html = """
        <div class="results">
            <a class="result__a" href="https://example.com">Example</a>
        </div>
        """

        mock_result = MagicMock(success=True, html=html)
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(agent, "_get_crawler", return_value=mock_crawler):
            with patch.object(agent, "_close_crawler", new_callable=AsyncMock):
                result = await agent.search_multiple(
                    ["query1", "query2"], deduplicate=False
                )

        # URL appears in both queries, so should still only appear once in all_urls
        # (deduplicate=False means we don't remove duplicates, but we also don't add
        # the same URL twice to all_urls)
        assert "https://example.com" in result.all_urls

    @pytest.mark.asyncio
    async def test_search_multiple_handles_failures(self):
        """Test that search_multiple handles individual query failures."""
        agent = SearchAgent(results_per_query=2, rate_limit_delay=0)

        success_html = """
        <div class="results">
            <a class="result__a" href="https://success.com">Success</a>
        </div>
        """

        mock_results = [
            MagicMock(success=True, html=success_html),
            MagicMock(success=False),  # This one fails
        ]

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(side_effect=mock_results)

        with patch.object(agent, "_get_crawler", return_value=mock_crawler):
            with patch.object(agent, "_close_crawler", new_callable=AsyncMock):
                result = await agent.search_multiple(["good", "bad"])

        assert result.successful_queries == 1
        assert result.failed_queries == 1
        assert len(result.all_urls) == 1


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestPydanticModels:
    """Tests for Pydantic data models."""

    def test_search_result_creation(self):
        """Test SearchResult model creation."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet text",
        )

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet text"

    def test_search_result_default_snippet(self):
        """Test SearchResult with default empty snippet."""
        result = SearchResult(title="Title", url="https://example.com")

        assert result.snippet == ""

    def test_query_search_results_creation(self):
        """Test QuerySearchResults model creation."""
        results = QuerySearchResults(
            query="test query",
            results=[
                SearchResult(title="R1", url="https://r1.com"),
                SearchResult(title="R2", url="https://r2.com"),
            ],
            success=True,
        )

        assert results.query == "test query"
        assert len(results.results) == 2
        assert results.success is True
        assert results.error is None

    def test_query_search_results_failure(self):
        """Test QuerySearchResults for failed query."""
        results = QuerySearchResults(
            query="failed query",
            results=[],
            success=False,
            error="Network timeout",
        )

        assert results.success is False
        assert results.error == "Network timeout"

    def test_aggregated_search_results_creation(self):
        """Test AggregatedSearchResults model creation."""
        aggregated = AggregatedSearchResults(
            queries=[
                QuerySearchResults(
                    query="q1",
                    results=[SearchResult(title="R1", url="https://r1.com")],
                )
            ],
            all_urls=["https://r1.com"],
            url_to_queries={"https://r1.com": ["q1"]},
            total_results=1,
            successful_queries=1,
            failed_queries=0,
        )

        assert len(aggregated.queries) == 1
        assert len(aggregated.all_urls) == 1
        assert aggregated.total_results == 1


# =============================================================================
# ContentExtractor Tests (existing + new)
# =============================================================================


class TestContentExtractor:
    """Tests for ContentExtractor class."""

    def test_initialization(self):
        """Test ContentExtractor initializes correctly."""
        extractor = ContentExtractor()
        assert extractor is not None

    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  This   is   messy    text  "
        clean = ContentExtractor.clean_text(dirty_text)

        assert clean == "This is messy text"

    def test_clean_text_preserves_single_spaces(self):
        """Test that clean_text preserves proper spacing."""
        text = "Hello World"
        clean = ContentExtractor.clean_text(text)

        assert clean == "Hello World"

    def test_clean_text_handles_newlines(self):
        """Test that clean_text handles newlines."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        clean = ContentExtractor.clean_text(text)

        assert clean == "Line 1 Line 2 Line 3"

    @pytest.mark.asyncio
    async def test_extract_returns_structure(self):
        """Test that extract returns expected structure."""
        extractor = ContentExtractor()

        # Mock the crawler
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = SAMPLE_CONTENT_HTML
        mock_result.markdown = "# Sample Markdown\n\nThis is the content."

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(extractor, "_get_crawler", return_value=mock_crawler):
            result = await extractor.extract("https://example.com")

        assert result.url == "https://example.com"
        assert isinstance(result.title, str)
        assert isinstance(result.markdown, str)
        assert isinstance(result.headings, list)
        assert isinstance(result.code_blocks, list)
        assert result.success is True


# =============================================================================
# Sample HTML for content extraction tests
# =============================================================================

SAMPLE_CONTENT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Example Article - Learn Python</title>
    <meta property="og:title" content="Example Article OG Title">
</head>
<body>
    <header>
        <nav>Navigation content</nav>
    </header>
    <main>
        <article>
            <h1>Getting Started with Python</h1>
            <p>Python is a versatile programming language that is great for beginners.</p>
            
            <h2>Installation</h2>
            <p>You can install Python from python.org or using a package manager.</p>
            
            <h3>Using pip</h3>
            <p>Pip is Python's package installer.</p>
            
            <pre><code class="language-python">
def hello_world():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
            </code></pre>
            
            <h2>Basic Syntax</h2>
            <p>Python uses indentation for code blocks.</p>
            
            <pre><code class="lang-bash">
pip install requests
pip install numpy
            </code></pre>
        </article>
    </main>
    <footer>
        <p>Footer content</p>
    </footer>
    <script>console.log("script");</script>
</body>
</html>
"""

SAMPLE_CONTENT_HTML_MINIMAL = """
<!DOCTYPE html>
<html>
<body>
    <h1>Simple Page</h1>
    <p>Some content here.</p>
</body>
</html>
"""


class TestContentExtractorParsing:
    """Tests for ContentExtractor HTML parsing methods."""

    def test_extract_title_from_title_tag(self):
        """Test title extraction from <title> tag."""
        extractor = ContentExtractor()
        soup = BeautifulSoup(SAMPLE_CONTENT_HTML, "html.parser")

        title = extractor._extract_title(soup)

        assert title == "Example Article - Learn Python"

    def test_extract_title_from_h1_fallback(self):
        """Test title extraction falls back to h1."""
        extractor = ContentExtractor()
        html = "<html><body><h1>H1 Title</h1></body></html>"
        soup = BeautifulSoup(html, "html.parser")

        title = extractor._extract_title(soup)

        assert title == "H1 Title"

    def test_extract_headings(self):
        """Test heading extraction from all levels."""
        extractor = ContentExtractor()
        soup = BeautifulSoup(SAMPLE_CONTENT_HTML, "html.parser")

        headings = extractor._extract_headings(soup)

        assert len(headings) == 4
        assert "H1: Getting Started with Python" in headings
        assert "H2: Installation" in headings
        assert "H3: Using pip" in headings
        assert "H2: Basic Syntax" in headings

    def test_extract_code_blocks(self):
        """Test code block extraction with language detection."""
        extractor = ContentExtractor()
        soup = BeautifulSoup(SAMPLE_CONTENT_HTML, "html.parser")

        code_blocks = extractor._extract_code_blocks(soup)

        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert "def hello_world" in code_blocks[0]["code"]
        assert code_blocks[1]["language"] == "bash"
        assert "pip install" in code_blocks[1]["code"]

    def test_clean_markdown_removes_excessive_blank_lines(self):
        """Test that excessive blank lines are removed."""
        extractor = ContentExtractor()

        markdown = "# Title\n\n\n\n\nParagraph 1\n\n\n\nParagraph 2"
        cleaned = extractor._clean_markdown(markdown)

        # Should have at most 2 consecutive blank lines
        assert "\n\n\n\n" not in cleaned
        assert "# Title" in cleaned
        assert "Paragraph 1" in cleaned
        assert "Paragraph 2" in cleaned

    def test_clean_markdown_truncates_long_content(self):
        """Test that very long content is truncated."""
        extractor = ContentExtractor()

        # Create content longer than 20000 chars
        long_markdown = "# Title\n\n" + "A" * 25000
        cleaned = extractor._clean_markdown(long_markdown)

        assert len(cleaned) <= 20100  # 20000 + truncation message
        assert "[Content truncated...]" in cleaned

    def test_clean_markdown_handles_empty_input(self):
        """Test that empty input returns empty string."""
        extractor = ContentExtractor()

        assert extractor._clean_markdown("") == ""
        assert extractor._clean_markdown(None) == ""


class TestContentExtractorAsync:
    """Async tests for ContentExtractor."""

    @pytest.mark.asyncio
    async def test_extract_success(self):
        """Test successful content extraction."""
        extractor = ContentExtractor(concurrency_limit=3, timeout=30)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = SAMPLE_CONTENT_HTML
        mock_result.markdown = "# Getting Started with Python\n\nPython is a versatile programming language."

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(extractor, "_get_crawler", return_value=mock_crawler):
            result = await extractor.extract(
                "https://example.com/article",
                source_queries=["python tutorial", "getting started python"],
            )

        assert result.success is True
        assert result.url == "https://example.com/article"
        assert result.title == "Example Article - Learn Python"
        assert len(result.headings) == 4
        assert len(result.code_blocks) == 2
        assert "python tutorial" in result.source_queries
        assert result.error is None
        assert "Python" in result.markdown

    @pytest.mark.asyncio
    async def test_extract_crawl_failure(self):
        """Test handling of crawl failure."""
        extractor = ContentExtractor()

        mock_result = MagicMock()
        mock_result.success = False

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        with patch.object(extractor, "_get_crawler", return_value=mock_crawler):
            result = await extractor.extract("https://example.com/failing")

        assert result.success is False
        assert result.error == "Crawl request failed"
        assert result.markdown == ""
        assert result.headings == []

    @pytest.mark.asyncio
    async def test_extract_exception_handling(self):
        """Test handling of exceptions during extraction."""
        extractor = ContentExtractor()

        with patch.object(
            extractor, "_get_crawler", side_effect=Exception("Network error")
        ):
            result = await extractor.extract("https://example.com/error")

        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_extract_multiple_concurrent(self):
        """Test concurrent extraction from multiple URLs."""
        extractor = ContentExtractor(concurrency_limit=3, timeout=30)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = SAMPLE_CONTENT_HTML
        mock_result.markdown = "# Sample Content\n\nThis is markdown content."

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        url_to_queries = {
            "https://example.com/page1": ["query1"],
            "https://example.com/page2": ["query2", "query3"],
        }

        with patch.object(extractor, "_get_crawler", return_value=mock_crawler):
            with patch.object(extractor, "_close_crawler", new_callable=AsyncMock):
                result = await extractor.extract_multiple(
                    urls, url_to_queries=url_to_queries, topic="test topic"
                )

        assert result.topic == "test topic"
        assert result.statistics["total_urls"] == 3
        assert result.statistics["successful"] == 3
        assert result.statistics["failed"] == 0
        assert len(result.contents) == 3
        assert all(c.success for c in result.contents)
        assert all(c.markdown for c in result.contents)

    @pytest.mark.asyncio
    async def test_extract_multiple_handles_failures(self):
        """Test that extract_multiple handles individual failures."""
        extractor = ContentExtractor(concurrency_limit=3, timeout=30)

        success_result = MagicMock()
        success_result.success = True
        success_result.html = SAMPLE_CONTENT_HTML
        success_result.markdown = "# Success\n\nThis worked."

        fail_result = MagicMock()
        fail_result.success = False

        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(side_effect=[success_result, fail_result])

        urls = ["https://example.com/good", "https://example.com/bad"]

        with patch.object(extractor, "_get_crawler", return_value=mock_crawler):
            with patch.object(extractor, "_close_crawler", new_callable=AsyncMock):
                result = await extractor.extract_multiple(urls, topic="test")

        assert result.statistics["successful"] == 1
        assert result.statistics["failed"] == 1
        assert len(result.contents) == 2


class TestExtractedContentModels:
    """Tests for Pydantic data models."""

    def test_extracted_content_creation(self):
        """Test ExtractedContent model creation."""
        content = ExtractedContent(
            url="https://example.com",
            title="Test Title",
            snippet="This is the search snippet preview.",
            markdown="# Test\n\nSome content text here.",
            headings=["H1: Main", "H2: Section"],
            code_blocks=[{"language": "python", "code": "print('hi')"}],
            success=True,
            source_queries=["test query"],
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Title"
        assert content.snippet == "This is the search snippet preview."
        assert content.success is True
        assert len(content.headings) == 2
        assert len(content.code_blocks) == 1
        assert content.error is None

    def test_extracted_content_failure(self):
        """Test ExtractedContent for failed extraction."""
        content = ExtractedContent(
            url="https://example.com",
            success=False,
            error="Connection timeout",
        )

        assert content.success is False
        assert content.error == "Connection timeout"
        assert content.markdown == ""
        assert content.headings == []

    def test_aggregated_extracted_content_creation(self):
        """Test AggregatedExtractedContent model creation."""
        aggregated = AggregatedExtractedContent(
            topic="Test Topic",
            statistics={"total_urls": 3, "successful": 2, "failed": 1},
            contents=[
                ExtractedContent(url="https://a.com", success=True),
                ExtractedContent(url="https://b.com", success=True),
                ExtractedContent(url="https://c.com", success=False, error="Failed"),
            ],
        )

        assert aggregated.topic == "Test Topic"
        assert aggregated.statistics["successful"] == 2
        assert len(aggregated.contents) == 3


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_question_generation_and_search():
    """
    Integration test for 2.1 (question generation), 2.2 (search agent), and 2.3 (content extraction).
    Generates research queries, runs crawl4ai search, and extracts content from URLs.
    """
    from src.planning.question_generator import QuestionGenerator
    from src.research.search_agent import SearchAgent
    from src.research.content_extractor import ContentExtractor

    topic = "Memory for AI agents using mem0"
    context = "Focus on open source Python implementation, architecture, best practices, and recent developments."

    # Step 2.1: Generate questions
    print("\n" + "=" * 60)
    print("Step 2.1: Generate Research Questions")
    print("=" * 60)
    generator = QuestionGenerator(model_name="gemini-2.0-flash")
    questions_result = await generator.generate(topic, context)
    queries = questions_result.questions
    print("\n=== Generated Research Queries ===")
    print(yaml.dump({"topic": topic, "queries": queries}, sort_keys=False, allow_unicode=True))

    # Step 2.2: Search agent
    print("\n" + "=" * 60)
    print("Step 2.2: Search for URLs")
    print("=" * 60)
    agent = SearchAgent(results_per_query=3, rate_limit_delay=1.5)
    search_results = await agent.search_multiple(queries, deduplicate=True)
    print("\n=== Top URLs for Each Query ===")
    for qres in search_results.queries:
        print(f"\nQuery: {qres.query}")
        if qres.success and qres.results:
            for i, r in enumerate(qres.results, 1):
                print(f"  {i}. {r.title}")
                print(f"     URL: {r.url}")
                if r.snippet:
                    print(f"     Snippet: {r.snippet[:100]}...")
        else:
            print("  No results or search failed.")
    print("\n=== Unique URLs Found ===")
    for url in search_results.all_urls:
        print(f"- {url}")
    print("\nTotal unique URLs:", len(search_results.all_urls))

    # Step 2.3: Content extraction
    print("\n" + "=" * 60)
    print("Step 2.3: Extract Content from URLs")
    print("=" * 60)
    
    # Only extract from first 5 URLs to keep test fast
    urls_to_extract = search_results.all_urls[:5]
    print(f"\nExtracting content from {len(urls_to_extract)} URLs (limited for test)...")
    
    # Build url_to_snippet mapping (keep longest snippet for each URL)
    url_to_snippet: dict[str, str] = {}
    for query_result in search_results.queries:
        for result in query_result.results:
            current_snippet = url_to_snippet.get(result.url, "")
            if len(result.snippet) > len(current_snippet):
                url_to_snippet[result.url] = result.snippet
    
    extractor = ContentExtractor(concurrency_limit=5, timeout=30)
    extracted_content = await extractor.extract_multiple(
        urls=urls_to_extract,
        url_to_queries=search_results.url_to_queries,
        url_to_snippet=url_to_snippet,
        topic=topic,
    )
    
    print(f"\n=== Extraction Results ===")
    print(f"Total URLs: {extracted_content.statistics.get('total_urls', 0)}")
    print(f"Successful: {extracted_content.statistics.get('successful', 0)}")
    print(f"Failed: {extracted_content.statistics.get('failed', 0)}")
    
    print("\n=== Extracted Content Summary ===")
    for content in extracted_content.contents:
        status = "✓" if content.success else "✗"
        print(f"\n{status} {content.url}")
        if content.success:
            print(f"   Title: {content.title[:60]}..." if len(content.title) > 60 else f"   Title: {content.title}")
            if content.snippet:
                print(f"   Snippet: {content.snippet[:80]}..." if len(content.snippet) > 80 else f"   Snippet: {content.snippet}")
            print(f"   Markdown: {len(content.markdown)} chars")
            print(f"   Headings: {len(content.headings)}")
            if content.headings:
                for h in content.headings[:3]:
                    print(f"     - {h[:60]}..." if len(h) > 60 else f"     - {h}")
                if len(content.headings) > 3:
                    print(f"     ... and {len(content.headings) - 3} more")
            print(f"   Code blocks: {len(content.code_blocks)}")
            if content.code_blocks:
                for cb in content.code_blocks[:2]:
                    print(f"     - {cb['language']}: {cb['code'][:50]}...")
            print(f"   Source queries: {content.source_queries}")
        else:
            print(f"   Error: {content.error}")
    
    # Assertions
    assert queries, "No queries generated"
    assert search_results.all_urls, "No URLs found"
    assert extracted_content.contents, "No content extracted"
    assert extracted_content.statistics.get("successful", 0) > 0, "No successful extractions"
    
    print("\n" + "=" * 60)
    print("Integration Test Complete!")
    print("=" * 60)
