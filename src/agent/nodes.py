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
    BlogPlan,
    DiscoveryQueries,
    JobManager,
    Phase,
)
from .tools import search_duckduckgo

logger = logging.getLogger(__name__)

# Target word counts by length
TARGET_WORDS_MAP = {
    "short": 800,
    "medium": 1500,
    "long": 2500,
}


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


# =============================================================================
# Planning Node (Phase 1)
# =============================================================================


def _format_topic_context_snippets(
    topic_context: list[dict[str, str]], max_snippets: int = 15
) -> str:
    """
    Format topic context for inclusion in the planning prompt.

    Args:
        topic_context: List of {title, url, snippet} dicts from discovery
        max_snippets: Maximum number of snippets to include (default 15)

    Returns:
        Formatted string with numbered snippets for the prompt
    """
    if not topic_context:
        return "No topic research available. Plan based on your knowledge."

    snippets = []
    for i, item in enumerate(topic_context[:max_snippets], 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("snippet", "")

        snippets.append(f"{i}. [{title}]({url})\n   {snippet}")

    return "\n\n".join(snippets)


def _build_planning_prompt(
    title: str, context: str, target_words: int, topic_snippets: str
) -> str:
    """
    Build the planning prompt for Gemini.

    Args:
        title: Blog title
        context: User-provided context
        target_words: Total target word count
        topic_snippets: Formatted topic context snippets

    Returns:
        Complete prompt string
    """
    return f"""You are planning a technical blog post.

Blog structure must follow:
1. Hook (optional) - story, surprising stat, or provocative question
2. Problem - what's broken with current approach
3. Why - why new approach matters
4. Subtopics - 4-6 deep_dive sections (user will select which ones to include)
5. Conclusion - practical takeaways

For each section, provide:
- id (unique string identifier, e.g., "hook", "problem", "implementation")
- title (section heading, null for hook)
- role (hook/problem/why/implementation/deep_dive/conclusion)
- search_queries (2-3 specific queries for research)
- needs_code (true/false - whether section needs code examples)
- needs_diagram (true if architecture/flow explanation needed)
- target_words (distribute {target_words} total across REQUIRED sections only)
- optional (true for 2 extra deep_dive sections user can choose from)

## Required vs Optional Sections
- hook, problem, why, conclusion: ALWAYS set optional=false (these are required)
- deep_dive/implementation sections: Generate 4-6 total, mark 2 as optional=true
- The 2 optional sections are extra choices for the user to pick from
- Word counts for optional sections should be estimated (will be pro-rated if selected)

## Blog Topic
Title: "{title}"
Context: {context}

## Topic Research (from web search)
The following snippets provide current context about this topic:
{topic_snippets}

Use this research to inform your subtopic selection.
Focus on aspects that appear important based on this context.

Generate a blog plan with 7-8 sections total (including 2 optional deep_dive sections).
Ensure target_words for required sections sum to approximately {target_words}.
Output as structured BlogPlan."""


async def _generate_blog_plan(
    title: str,
    context: str,
    target_words: int,
    topic_context: list[dict[str, str]],
    key_manager: KeyManager,
    max_retries: int = 3,
) -> BlogPlan:
    """
    Generate blog plan using Gemini Flash-Lite with structured output.

    Args:
        title: Blog title
        context: User-provided context
        target_words: Total target word count
        topic_context: List of {title, url, snippet} from discovery
        key_manager: KeyManager for API key rotation
        max_retries: Maximum retry attempts

    Returns:
        BlogPlan Pydantic model with sections

    Raises:
        RuntimeError: If all API keys exhausted or max retries exceeded
    """
    # Format topic snippets
    topic_snippets = _format_topic_context_snippets(topic_context)

    # Build prompt
    prompt = _build_planning_prompt(title, context, target_words, topic_snippets)

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
            structured_llm = llm.with_structured_output(BlogPlan)

            # Invoke (run in thread pool since langchain may be sync internally)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            # Record usage
            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=500,
            )

            return result

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            # Check for rate limit / quota exhausted errors
            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            # For other errors, retry with backoff
            logger.error(f"Plan generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(
        f"Failed to generate blog plan after {max_retries} attempts: {last_error}"
    )


async def planning_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 1: Planning Node.

    Generates a structured blog plan with sections from title, context,
    target length, and topic context (from discovery phase).

    Args:
        state: Current BlogAgentState containing:
            - title: Blog title (required)
            - context: User context (required)
            - target_length: "short" | "medium" | "long" (default "medium")
            - topic_context: List of {title, url, snippet} from discovery
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - plan: BlogPlan as dict
        - current_phase: Updated to RESEARCHING
    """
    logger.info(f"Starting planning for: {state.get('title')}")

    title = state.get("title", "")
    context = state.get("context", "")
    target_length = state.get("target_length", "medium")
    topic_context = state.get("topic_context", [])
    job_id = state.get("job_id", "")

    # Validate required inputs
    if not title:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Title is required for planning",
        }

    # Map target_length to target_words
    target_words = TARGET_WORDS_MAP.get(target_length, 1500)

    try:
        # Initialize key manager from environment
        key_manager = KeyManager.from_env()

        # Generate blog plan using LLM
        plan = await _generate_blog_plan(
            title=title,
            context=context,
            target_words=target_words,
            topic_context=topic_context,
            key_manager=key_manager,
        )

        # Convert to dict for state storage
        plan_dict = plan.model_dump()
        logger.info(
            f"Generated plan with {len(plan.sections)} sections, {target_words} target words"
        )

        # Save checkpoint if job_id provided
        if job_id:
            job_manager = JobManager()
            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.RESEARCHING.value,
                    "plan": plan_dict,
                },
            )

        return {
            "plan": plan_dict,
            "current_phase": Phase.RESEARCHING.value,
        }

    except RuntimeError as e:
        logger.error(f"Planning failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error in planning: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
        }
