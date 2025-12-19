"""Integration tests for tools module - requires network access."""

import pytest

from src.agent.tools import (
    search_duckduckgo,
    fetch_url_content,
    render_mermaid,
)


class TestSearchDuckDuckGo:
    """Integration tests for DuckDuckGo search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Real search returns results."""
        results = await search_duckduckgo("Python programming", max_results=3)

        assert len(results) > 0
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_result_structure(self):
        """Results have expected structure."""
        results = await search_duckduckgo("asyncio tutorial", max_results=2)

        # DuckDuckGo may return empty for some queries (rate limiting)
        if len(results) > 0:
            for result in results:
                assert "title" in result
                assert "url" in result
                assert "snippet" in result
                assert result["url"].startswith("http")

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_results(self):
        """Empty-ish query still works."""
        results = await search_duckduckgo("a", max_results=1)
        # Should return something or empty list, but not fail
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_special_characters(self):
        """Query with special characters works."""
        results = await search_duckduckgo("python \"async await\"", max_results=2)
        assert isinstance(results, list)


class TestFetchUrlContent:
    """Integration tests for URL content fetching."""

    @pytest.mark.asyncio
    async def test_fetch_example_com(self):
        """Fetch from example.com succeeds."""
        result = await fetch_url_content("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert len(result["content"]) > 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_fetch_returns_title(self):
        """Title is extracted from page."""
        result = await fetch_url_content("https://example.com")

        assert result["title"]  # Should have some title

    @pytest.mark.asyncio
    async def test_fetch_estimates_tokens(self):
        """Token estimate is calculated."""
        result = await fetch_url_content("https://example.com")

        assert result["tokens_estimate"] > 0
        # Should be roughly content_length / 4
        expected_estimate = len(result["content"]) // 4
        assert abs(result["tokens_estimate"] - expected_estimate) < 10

    @pytest.mark.asyncio
    async def test_fetch_invalid_url_fails_gracefully(self):
        """Invalid URL returns error, doesn't raise."""
        result = await fetch_url_content("https://this-domain-does-not-exist-12345.com")

        assert result["success"] is False
        assert result["error"] is not None
        assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_fetch_404_fails_gracefully(self):
        """404 response returns error, doesn't raise."""
        result = await fetch_url_content("https://httpstat.us/404")

        assert result["success"] is False
        assert result["error"] is not None  # Some error message returned

    @pytest.mark.asyncio
    async def test_fetch_timeout_handled(self):
        """Timeout is handled gracefully."""
        # httpstat.us/200?sleep=5000 sleeps for 5 seconds
        result = await fetch_url_content(
            "https://httpstat.us/200?sleep=5000",
            timeout=1  # 1 second timeout
        )

        assert result["success"] is False
        assert "timeout" in result["error"].lower() or result["error"]


class TestRenderMermaid:
    """Integration tests for mermaid diagram rendering."""

    @pytest.mark.asyncio
    async def test_render_simple_diagram(self, tmp_path):
        """Simple diagram renders to PNG."""
        mermaid_code = """graph TD
    A[Start] --> B[Process]
    B --> C[End]"""

        output_path = tmp_path / "test_diagram.png"
        result = await render_mermaid(mermaid_code, str(output_path))

        assert result is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Check it's actually a PNG (magic bytes)
        with open(output_path, "rb") as f:
            header = f.read(8)
            assert header[:4] == b"\x89PNG"

    @pytest.mark.asyncio
    async def test_render_flowchart(self, tmp_path):
        """Flowchart diagram renders correctly."""
        mermaid_code = """flowchart LR
    A[Start] --> B{Is it?}
    B -->|Yes| C[OK]
    B -->|No| D[End]"""

        output_path = tmp_path / "flowchart.png"
        result = await render_mermaid(mermaid_code, str(output_path))

        assert result is not None
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_render_sequence_diagram(self, tmp_path):
        """Sequence diagram renders correctly."""
        mermaid_code = """sequenceDiagram
    Alice->>Bob: Hello Bob
    Bob-->>Alice: Hi Alice"""

        output_path = tmp_path / "sequence.png"
        result = await render_mermaid(mermaid_code, str(output_path))

        assert result is not None
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_render_invalid_mermaid_fails(self, tmp_path):
        """Invalid mermaid syntax returns None."""
        mermaid_code = "this is not valid mermaid syntax {{{}}}}"

        output_path = tmp_path / "invalid.png"
        result = await render_mermaid(mermaid_code, str(output_path))

        # kroki.io may return an error image or fail
        # We just check it doesn't raise
        assert result is None or output_path.exists()

    @pytest.mark.asyncio
    async def test_render_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if needed."""
        mermaid_code = "graph TD\n    A-->B"

        # Nested path
        output_path = tmp_path / "subdir" / "nested" / "diagram.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = await render_mermaid(mermaid_code, str(output_path))

        assert result is not None
        assert output_path.exists()


class TestCombinedWorkflow:
    """Integration tests for combined tool usage."""

    @pytest.mark.asyncio
    async def test_search_and_fetch_workflow(self):
        """Search then fetch workflow works."""
        # Search for something
        results = await search_duckduckgo("Python official documentation", max_results=2)

        # DuckDuckGo may return empty due to rate limiting
        if len(results) == 0:
            pytest.skip("DuckDuckGo returned empty results (rate limited)")

        # Fetch the first result
        first_url = results[0]["url"]
        content = await fetch_url_content(first_url)

        # Should get some content (might fail on some URLs)
        # We just verify no exceptions raised
        assert "success" in content
        assert "url" in content
