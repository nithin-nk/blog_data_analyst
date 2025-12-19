"""
Nodes module - LangGraph node implementations for the blog agent pipeline.

Each node function takes BlogAgentState and returns a dict of state updates.
Nodes are async functions that handle one phase of the pipeline.
"""

import asyncio
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .key_manager import KeyManager
from .state import (
    BlogAgentState,
    DiscoveryQueries,
    JobManager,
    Phase,
)
from .tools import search_duckduckgo

logger = logging.getLogger(__name__)


# =============================================================================
# Topic Discovery Node (Phase 0.5)
# =============================================================================


async def topic_discovery_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 0.5: Topic Discovery Node.

    Generates search queries from title + context using Gemini Flash-Lite,
    executes DuckDuckGo searches, and compiles deduplicated topic context.

    Args:
        state: Current BlogAgentState containing:
            - title: Blog title (required)
            - context: User context (required)
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - discovery_queries: List of generated search queries
        - topic_context: List of {title, url, snippet} dicts
        - current_phase: Updated to PLANNING
    """
    logger.info(f"Starting topic discovery for: {state.get('title')}")

    title = state.get("title", "")
    context = state.get("context", "")
    job_id = state.get("job_id", "")

    if not title:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Title is required for topic discovery",
        }

    try:
        # Initialize key manager from environment
        key_manager = KeyManager.from_env()

        # Step 1: Generate discovery queries using LLM
        discovery_result = await _generate_discovery_queries(
            title=title,
            context=context,
            key_manager=key_manager,
        )
        queries = discovery_result.queries
        logger.info(f"Generated {len(queries)} discovery queries: {queries}")

        # Step 2: Execute searches
        topic_context = await _execute_searches(
            queries=queries,
            max_results_per_query=5,
            max_total_results=20,
        )
        logger.info(f"Collected {len(topic_context)} unique search results")

        # Step 3: Save checkpoint if job_id provided
        if job_id:
            job_manager = JobManager()
            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.PLANNING.value,
                    "discovery_queries": queries,
                    "topic_context": topic_context,
                },
            )

        return {
            "discovery_queries": queries,
            "topic_context": topic_context,
            "current_phase": Phase.PLANNING.value,
        }

    except RuntimeError as e:
        logger.error(f"Topic discovery failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error in topic discovery: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
        }


async def _generate_discovery_queries(
    title: str,
    context: str,
    key_manager: KeyManager,
    max_retries: int = 3,
) -> DiscoveryQueries:
    """
    Generate search queries using Gemini Flash-Lite with structured output.

    Args:
        title: Blog title
        context: User-provided context
        key_manager: KeyManager for API key rotation
        max_retries: Maximum retry attempts

    Returns:
        DiscoveryQueries Pydantic model with 3-5 queries

    Raises:
        RuntimeError: If all API keys exhausted or max retries exceeded
    """
    prompt = f"""Generate 3-5 search queries to learn about this topic:

Title: "{title}"
Context: "{context}"

Goals:
- Understand what this topic is about
- Find key subtopics and concepts
- Discover recent developments (2024-2025)
- Identify practical use cases

Output JSON: {{ "queries": ["...", "...", ...] }}"""

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            # Initialize model with structured output
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=api_key,
                temperature=0.7,
            )

            # Use with_structured_output for Pydantic validation
            structured_llm = llm.with_structured_output(DiscoveryQueries)

            # Invoke (run in thread pool since langchain may be sync internally)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            # Record usage (approximate - actual tokens would come from response metadata)
            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=100,
            )

            return result

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            # Check for rate limit / quota exhausted errors
            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning(f"Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            # For other errors, retry with backoff
            logger.error(f"Query generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

    raise RuntimeError(f"Failed to generate discovery queries after {max_retries} attempts: {last_error}")


def _extract_domain(url: str) -> str:
    """Extract domain from URL for deduplication."""
    try:
        # Handle URLs like https://www.example.com/path
        if "://" in url:
            domain = url.split("://")[1].split("/")[0]
        else:
            domain = url.split("/")[0]
        # Remove www. prefix for consistency
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except (IndexError, AttributeError):
        return ""


async def _execute_searches(
    queries: list[str],
    max_results_per_query: int = 5,
    max_total_results: int = 20,
    max_per_domain: int = 1,
) -> list[dict[str, str]]:
    """
    Execute DuckDuckGo searches for all queries and deduplicate results.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per individual query
        max_total_results: Max total unique results to return
        max_per_domain: Max results per domain for diversity (default 1)

    Returns:
        Deduplicated list of {title, url, snippet} dicts with domain diversity
    """
    all_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}

    # Execute searches concurrently
    search_tasks = [
        search_duckduckgo(query, max_results=max_results_per_query) for query in queries
    ]

    query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for query, results in zip(queries, query_results):
        if isinstance(results, Exception):
            logger.warning(f"Search failed for '{query}': {results}")
            continue

        for result in results:
            url = result.get("url", "")

            # Skip if already seen or empty
            if not url or url in seen_urls:
                continue

            # Check domain limit for diversity
            domain = _extract_domain(url)
            if domain and domain_counts.get(domain, 0) >= max_per_domain:
                logger.debug(f"Skipping {url} - domain {domain} at limit")
                continue

            seen_urls.add(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            all_results.append(
                {
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                }
            )

            # Stop if we have enough
            if len(all_results) >= max_total_results:
                break

        if len(all_results) >= max_total_results:
            break

    return all_results[:max_total_results]
