"""
Nodes module - LangGraph node implementations for the blog agent pipeline.

Each node function takes BlogAgentState and returns a dict of state updates.
Nodes are async functions that handle one phase of the pipeline.
"""

import asyncio
import hashlib
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    LLM_MODEL_LITE,
    LLM_TEMPERATURE_LOW,
    LLM_TEMPERATURE_MEDIUM,
    MAX_VALIDATION_RETRIES,
    MIN_SOURCES_PER_SECTION,
    TARGET_WORDS_MAP,
)
from .key_manager import KeyManager
from .state import (
    AlternativeQueries,
    BlogAgentState,
    BlogPlan,
    DiscoveryQueries,
    JobManager,
    Phase,
    SourceValidationList,
)
from .tools import fetch_url_content, search_duckduckgo

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


# =============================================================================
# Research Node (Phase 2)
# =============================================================================


def _hash_url(url: str) -> str:
    """Create a short hash key for a URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


async def _research_section(
    section_id: str,
    search_queries: list[str],
    existing_cache: dict[str, dict],
    max_urls_per_query: int = 3,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Research a single section by searching and fetching content.

    Args:
        section_id: ID of the section being researched
        search_queries: List of search queries for this section
        existing_cache: Current research cache to check for duplicates
        max_urls_per_query: Max URLs to fetch per query

    Returns:
        Tuple of (section_sources, updated_cache)
        - section_sources: List of {url, title, content, tokens_estimate, section_id}
        - updated_cache: Cache with any new fetched content
    """
    section_sources: list[dict] = []
    seen_urls: set[str] = set(existing_cache.keys())
    cache_updates: dict[str, dict] = {}

    for query in search_queries:
        # Search for URLs
        search_results = await search_duckduckgo(query, max_results=max_urls_per_query + 2)

        # Fetch content from each URL
        for result in search_results[:max_urls_per_query]:
            url = result.get("url", "")
            if not url:
                continue

            url_hash = _hash_url(url)

            # Check if already in cache
            if url_hash in seen_urls:
                # Use cached content if available
                if url_hash in existing_cache:
                    cached = existing_cache[url_hash]
                    if cached.get("content"):
                        section_sources.append({
                            **cached,
                            "section_id": section_id,
                        })
                continue

            seen_urls.add(url_hash)

            # Fetch content
            fetch_result = await fetch_url_content(url)

            if fetch_result["success"] and fetch_result.get("content"):
                source_data = {
                    "url": url,
                    "url_hash": url_hash,
                    "title": fetch_result.get("title", result.get("title", "")),
                    "content": fetch_result["content"],
                    "tokens_estimate": fetch_result.get("tokens_estimate", 0),
                    "section_id": section_id,
                }
                section_sources.append(source_data)
                cache_updates[url_hash] = {
                    "url": url,
                    "title": source_data["title"],
                    "content": source_data["content"],
                    "tokens_estimate": source_data["tokens_estimate"],
                }
            else:
                logger.warning(f"Failed to fetch {url}: {fetch_result.get('error')}")

        # Stop if we have enough sources for this section
        if len(section_sources) >= 6:
            break

    return section_sources, cache_updates


async def research_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 2: Research Node.

    Fetches content from web for each section's search queries.
    Stores results in research_cache for validation phase.

    Args:
        state: Current BlogAgentState containing:
            - plan: BlogPlan with sections and search_queries
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - research_cache: Dict of url_hash -> content data
        - current_phase: Updated to VALIDATING_SOURCES
    """
    logger.info("Starting research phase")

    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    job_id = state.get("job_id", "")
    existing_cache = state.get("research_cache", {})

    if not sections:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "No sections found in plan",
        }

    try:
        research_cache = dict(existing_cache)
        section_sources_map: dict[str, list[dict]] = {}

        # Research each section (including optional ones)
        for section in sections:
            section_id = section.get("id", "")
            search_queries = section.get("search_queries", [])

            if not search_queries:
                logger.info(f"Section {section_id} has no search queries, skipping")
                section_sources_map[section_id] = []
                continue

            logger.info(f"Researching section: {section_id} with {len(search_queries)} queries")

            sources, cache_updates = await _research_section(
                section_id=section_id,
                search_queries=search_queries,
                existing_cache=research_cache,
                max_urls_per_query=3,
            )

            research_cache.update(cache_updates)
            section_sources_map[section_id] = sources

            logger.info(f"Found {len(sources)} sources for section {section_id}")

        # Save checkpoint if job_id provided
        if job_id:
            job_manager = JobManager()
            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.VALIDATING_SOURCES.value,
                    "research_cache": research_cache,
                },
            )

        total_sources = sum(len(s) for s in section_sources_map.values())
        logger.info(f"Research complete: {total_sources} total sources across {len(sections)} sections")

        return {
            "research_cache": research_cache,
            "current_phase": Phase.VALIDATING_SOURCES.value,
        }

    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Research failed: {e}",
        }


# =============================================================================
# Validate Sources Node (Phase 2.5)
# =============================================================================


def _build_validation_prompt(
    blog_title: str,
    section_title: str,
    section_role: str,
    sources: list[dict],
    all_sections: list[dict] | None = None,
) -> str:
    """
    Build prompt for source validation.

    Args:
        blog_title: Title of the blog post
        section_title: Title of the section
        section_role: Role of the section (hook, problem, why, etc.)
        sources: List of source dicts with url, title, content
        all_sections: Full list of blog sections for context

    Returns:
        Prompt string for LLM
    """
    # Build blog structure context
    blog_structure = ""
    if all_sections:
        blog_structure = "\n## Blog Structure (all planned sections):\n"
        for s in all_sections:
            opt_marker = " [optional]" if s.get("optional") else ""
            title = s.get("title") or f"({s.get('role', 'section')})"
            blog_structure += f"- {title}{opt_marker}\n"
        blog_structure += "\n"

    sources_text = ""
    for i, source in enumerate(sources, 1):
        # Truncate content to ~1000 chars for prompt
        content = source.get("content", "")[:1000]
        sources_text += f"""
Source {i}:
- URL: {source.get('url', '')}
- Title: {source.get('title', '')}
- Content preview: {content}...
"""

    return f"""You are evaluating research sources for a technical blog about:
"{blog_title}"
{blog_structure}
## Current Section Being Evaluated:
"{section_title}" (role: {section_role})

For each source, evaluate:
1. Is it relevant to THIS SPECIFIC section's topic (not just the overall blog)?
2. What is the quality? (high/medium/low)
3. Should we use this source for this section? (true/false)
4. Brief reason for your decision

{sources_text}

Evaluate all {len(sources)} sources. Output as SourceValidationList."""


async def _generate_alternative_queries(
    blog_title: str,
    section: dict,
    original_queries: list[str],
    key_manager: KeyManager,
    max_retries: int = 3,
) -> list[str]:
    """
    Generate alternative search queries when sources are insufficient.

    Args:
        blog_title: Title of the blog
        section: Section dict with id, title, role
        original_queries: Queries that didn't yield enough sources
        key_manager: KeyManager for API key rotation
        max_retries: Max LLM retry attempts

    Returns:
        List of 2-3 alternative search queries
    """
    section_title = section.get("title") or section.get("id", "")
    section_role = section.get("role", "")

    prompt = f"""The following search queries didn't yield enough quality sources:
{original_queries}

For this blog section:
Title: "{section_title}"
Role: {section_role}
Blog: "{blog_title}"

Generate 2-3 DIFFERENT search queries that might find better sources.
Focus on: tutorials, documentation, case studies, benchmarks.
Avoid repeating the same keywords from the original queries.

Output as AlternativeQueries."""

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_LITE,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_MEDIUM,
            )

            structured_llm = llm.with_structured_output(AlternativeQueries)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=100,
            )

            logger.info(f"Generated {len(result.queries)} alternative queries: {result.queries}")
            return result.queries

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            logger.error(f"Alternative query generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(f"Failed to generate alternative queries after {max_retries} attempts: {last_error}")


async def _validate_section_sources(
    blog_title: str,
    section: dict,
    sources: list[dict],
    key_manager: KeyManager,
    all_sections: list[dict] | None = None,
    min_sources: int = 4,
    max_retries: int = 3,
) -> list[dict]:
    """
    Validate sources for a section using LLM.

    Args:
        blog_title: Title of the blog
        section: Section dict with id, title, role
        sources: List of source dicts to validate
        key_manager: KeyManager for API key rotation
        all_sections: Full list of blog sections for context
        min_sources: Minimum validated sources required
        max_retries: Max LLM retry attempts

    Returns:
        List of validated source dicts with quality and reason added
    """
    if not sources:
        return []

    section_id = section.get("id", "")
    section_title = section.get("title") or section_id
    section_role = section.get("role", "")

    prompt = _build_validation_prompt(
        blog_title, section_title, section_role, sources, all_sections
    )

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_LITE,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_LOW,
            )

            structured_llm = llm.with_structured_output(SourceValidationList)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=200,
            )

            # Filter to sources where use=true
            validated = []
            for i, validation in enumerate(result.sources):
                if validation.use and i < len(sources):
                    source = sources[i]
                    validated.append({
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "content": source.get("content", ""),
                        "tokens_estimate": source.get("tokens_estimate", 0),
                        "quality": validation.quality,
                        "reason": validation.reason,
                    })

            logger.info(f"Section {section_id}: {len(validated)}/{len(sources)} sources validated")

            # Warn if below minimum but don't fail
            if len(validated) < min_sources:
                logger.warning(
                    f"Section {section_id} has only {len(validated)} validated sources "
                    f"(minimum: {min_sources})"
                )

            return validated

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            logger.error(f"Validation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(f"Failed to validate sources after {max_retries} attempts: {last_error}")


async def validate_sources_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 2.5: Validate Sources Node.

    Evaluates fetched sources for quality and relevance using LLM.
    Filters to keep only useful sources for each section.

    Args:
        state: Current BlogAgentState containing:
            - plan: BlogPlan with sections
            - research_cache: Fetched content from research phase
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - validated_sources: Dict of section_id -> list of validated sources
        - current_phase: Updated to WRITING
    """
    logger.info("Starting source validation phase")

    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    research_cache = state.get("research_cache", {})
    job_id = state.get("job_id", "")
    blog_title = plan.get("blog_title", state.get("title", ""))

    if not sections:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "No sections found in plan",
        }

    try:
        key_manager = KeyManager.from_env()
        validated_sources: dict[str, list[dict]] = {}

        for section in sections:
            section_id = section.get("id", "")

            # Gather sources for this section from cache
            section_sources = [
                {**data, "url_hash": url_hash}
                for url_hash, data in research_cache.items()
                if data.get("content")  # Only include sources with content
            ]

            if not section_sources:
                logger.warning(f"No sources found for section {section_id}")
                validated_sources[section_id] = []
                continue

            # Limit sources per section to avoid huge prompts
            sources_to_validate = section_sources[:10]

            logger.info(f"Validating {len(sources_to_validate)} sources for section {section_id}")

            # First attempt: validate existing sources
            validated = await _validate_section_sources(
                blog_title=blog_title,
                section=section,
                sources=sources_to_validate,
                key_manager=key_manager,
                all_sections=sections,
                min_sources=MIN_SOURCES_PER_SECTION,
            )

            # Track queries that have been used
            used_queries = set(section.get("search_queries", []))

            # Retry loop if insufficient sources
            retry_count = 0
            while len(validated) < MIN_SOURCES_PER_SECTION and retry_count < MAX_VALIDATION_RETRIES:
                retry_count += 1
                logger.info(
                    f"Section {section_id}: {len(validated)} sources (need {MIN_SOURCES_PER_SECTION}), "
                    f"retry {retry_count}/{MAX_VALIDATION_RETRIES}"
                )

                # Generate alternative queries
                alt_queries = await _generate_alternative_queries(
                    blog_title=blog_title,
                    section=section,
                    original_queries=list(used_queries),
                    key_manager=key_manager,
                )
                used_queries.update(alt_queries)

                # Research with new queries
                new_sources, cache_updates = await _research_section(
                    section_id=section_id,
                    search_queries=alt_queries,
                    existing_cache=research_cache,
                )
                research_cache.update(cache_updates)

                # Validate new sources (excluding already validated URLs)
                existing_urls = {s["url"] for s in validated}
                new_to_validate = [s for s in new_sources if s.get("url") not in existing_urls]

                if new_to_validate:
                    new_validated = await _validate_section_sources(
                        blog_title=blog_title,
                        section=section,
                        sources=new_to_validate,
                        key_manager=key_manager,
                        all_sections=sections,
                        min_sources=MIN_SOURCES_PER_SECTION,
                    )
                    validated.extend(new_validated)

                logger.info(f"After retry {retry_count}: {len(validated)} sources for {section_id}")

            validated_sources[section_id] = validated

        # Save checkpoint if job_id provided
        if job_id:
            job_manager = JobManager()
            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.WRITING.value,
                    "validated_sources": validated_sources,
                },
            )

        total_validated = sum(len(s) for s in validated_sources.values())
        logger.info(f"Validation complete: {total_validated} sources validated across {len(sections)} sections")

        return {
            "validated_sources": validated_sources,
            "research_cache": research_cache,  # Include updated cache from retries
            "current_phase": Phase.WRITING.value,
        }

    except RuntimeError as e:
        logger.error(f"Source validation failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
        }
    except Exception as e:
        logger.error(f"Unexpected error in source validation: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
        }
