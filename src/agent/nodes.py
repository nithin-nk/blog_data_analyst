"""
Nodes module - LangGraph node implementations for the blog agent pipeline.

Each node function takes BlogAgentState and returns a dict of state updates.
Nodes are async functions that handle one phase of the pipeline.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    GEMINI_PRICING,
    LLM_MODEL_FULL,
    LLM_MODEL_LITE,
    LLM_TEMPERATURE_LOW,
    LLM_TEMPERATURE_MEDIUM,
    MAX_RESEARCH_QUERIES_PER_ISSUE,
    MAX_RESEARCH_URLS_PER_QUERY,
    MAX_SECTION_RETRIES,
    MAX_VALIDATION_RETRIES,
    MIN_SOURCES_PER_SECTION,
    QUERY_DIVERSIFICATION_MODIFIERS,
    STYLE_GUIDE,
    TARGET_WORDS_MAP,
)
from .punchy_examples import get_example_for_role
from .key_manager import KeyManager
from .state import (
    AlternativeQueries,
    BlogAgentState,
    BlogPlan,
    ContentStrategy,
    DiscoveryQueries,
    ExistingArticleSummary,
    FinalCriticResult,
    FinalCriticScore,
    JobManager,
    Phase,
    PreviewValidationResult,
    ReplanningFeedback,
    SectionCriticResult,
    SectionFeasibilityScore,
    SectionSuggestion,
    SourceValidationList,
    TransitionFix,
    UniquenessAnalysisResult,
    UniquenessCheck,
    phase_is_past,
)
from .tools import fetch_url_content, search_duckduckgo

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _preserve_key_manager(state: BlogAgentState) -> dict:
    """
    Return dict preserving key_manager from state.

    Used when nodes skip to ensure key_manager persists through state merges.
    LangGraph merges returned dicts into state, so non-serializable objects like
    key_manager must be explicitly passed through.
    """
    km = state.get("key_manager")
    return {"key_manager": km} if km else {}


# =============================================================================
# Metrics Tracking Helpers
# =============================================================================


def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate cost in USD for an LLM call."""
    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING.get("gemini-2.5-flash-lite", {}))
    input_cost = (tokens_in / 1_000_000) * pricing.get("input", 0)
    output_cost = (tokens_out / 1_000_000) * pricing.get("output", 0)
    return input_cost + output_cost


def _init_node_metrics(state: dict, node_name: str) -> tuple[dict, float]:
    """
    Initialize metrics tracking for a node.

    Returns:
        Tuple of (node_metrics dict, start_time)
    """
    metrics = state.get("metrics", {})
    if node_name not in metrics:
        metrics[node_name] = {
            "duration_s": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "calls": 0,
            "cost": 0.0,
        }
    return metrics, time.time()


def _record_llm_call(
    metrics: dict,
    node_name: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """
    Record an LLM call and return the cost.

    Args:
        metrics: The full metrics dict
        node_name: Name of the current node
        model: LLM model used
        tokens_in: Input tokens
        tokens_out: Output tokens

    Returns:
        Cost of this call in USD
    """
    cost = _calculate_cost(model, tokens_in, tokens_out)
    node_metrics = metrics.get(node_name, {})
    node_metrics["tokens_in"] = node_metrics.get("tokens_in", 0) + tokens_in
    node_metrics["tokens_out"] = node_metrics.get("tokens_out", 0) + tokens_out
    node_metrics["calls"] = node_metrics.get("calls", 0) + 1
    node_metrics["cost"] = node_metrics.get("cost", 0.0) + cost
    metrics[node_name] = node_metrics

    logger.info(
        f"  └─ {model}: {tokens_in:,} in / {tokens_out:,} out (${cost:.4f})"
    )
    return cost


def _finalize_node_metrics(
    metrics: dict,
    node_name: str,
    start_time: float,
) -> dict:
    """
    Finalize metrics for a node by recording duration.

    Returns:
        The updated metrics dict
    """
    duration = time.time() - start_time
    if node_name in metrics:
        metrics[node_name]["duration_s"] = duration
    return metrics


def _analyze_sentence_lengths(content: str) -> dict[str, Any]:
    """
    Analyze sentence structure metrics for punchy style evaluation.

    Splits content into sentences and calculates:
    - Average sentence length in words
    - Maximum sentence length
    - Count of long sentences (> 20 words)
    - Count of very long sentences (> 25 words)
    - Count of semicolons (compound sentences)
    - Percentage of simple sentences (<= 18 words)

    Args:
        content: Markdown content to analyze

    Returns:
        Dict with metrics:
        - avg_length: float (average words per sentence)
        - max_length: int (longest sentence in words)
        - long_sentences: int (count of sentences > 20 words)
        - very_long_sentences: int (count of sentences > 25 words)
        - semicolons: int (count of semicolons in content)
        - percent_simple: float (% of sentences <= 18 words)
        - total_sentences: int (total sentence count)
    """
    import re

    # Remove code blocks (they shouldn't count toward sentence metrics)
    content_without_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Split into sentences (. ! ?) followed by space or end of string
    # This regex handles common cases while avoiding splits on "e.g." or "i.e."
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", content_without_code)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {
            "avg_length": 0.0,
            "max_length": 0,
            "long_sentences": 0,
            "very_long_sentences": 0,
            "semicolons": 0,
            "percent_simple": 100.0,
            "total_sentences": 0,
        }

    # Calculate word counts per sentence
    word_counts = [len(s.split()) for s in sentences]

    avg_length = sum(word_counts) / len(word_counts)
    max_length = max(word_counts)
    long_sentences = sum(1 for wc in word_counts if wc > 20)
    very_long_sentences = sum(1 for wc in word_counts if wc > 25)
    simple_sentences = sum(1 for wc in word_counts if wc <= 18)
    percent_simple = (simple_sentences / len(word_counts)) * 100

    # Count semicolons (indicator of compound sentences)
    semicolons = content_without_code.count(";")

    return {
        "avg_length": round(avg_length, 1),
        "max_length": max_length,
        "long_sentences": long_sentences,
        "very_long_sentences": very_long_sentences,
        "semicolons": semicolons,
        "percent_simple": round(percent_simple, 1),
        "total_sentences": len(sentences),
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
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.TOPIC_DISCOVERY):
        logger.info("Skipping topic_discovery_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "topic_discovery"
    metrics, start_time = _init_node_metrics(state, node_name)
    logger.info(f"Starting topic discovery for: {state.get('title')}")

    title = state.get("title", "")
    context = state.get("context", "")
    job_id = state.get("job_id", "")

    if not title:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Title is required for topic discovery",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    try:
        # Initialize key manager from environment
        key_manager = KeyManager.from_env()

        # Step 1: Generate discovery queries using LLM
        discovery_result = await _generate_discovery_queries(
            title=title,
            context=context,
            key_manager=key_manager,
            metrics=metrics,
            node_name=node_name,
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
                    "current_phase": Phase.CONTENT_LANDSCAPE.value,
                    "discovery_queries": queries,
                    "topic_context": topic_context,
                },
            )

        return {
            "discovery_queries": queries,
            "topic_context": topic_context,
            "current_phase": Phase.CONTENT_LANDSCAPE.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except RuntimeError as e:
        logger.error(f"Topic discovery failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    except Exception as e:
        logger.error(f"Unexpected error in topic discovery: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


async def _generate_discovery_queries(
    title: str,
    context: str,
    key_manager: KeyManager,
    max_retries: int = 3,
    metrics: dict | None = None,
    node_name: str = "topic_discovery",
) -> DiscoveryQueries:
    """
    Generate search queries using Gemini Flash-Lite with structured output.

    Args:
        title: Blog title
        context: User-provided context
        key_manager: KeyManager for API key rotation
        max_retries: Maximum retry attempts
        metrics: Optional metrics dict to update
        node_name: Name of the calling node for metrics

    Returns:
        DiscoveryQueries Pydantic model with 3-5 queries

    Raises:
        RuntimeError: If all API keys exhausted or max retries exceeded
    """
    model = LLM_MODEL_LITE
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
                model=model,
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

            # Estimate tokens
            tokens_in = len(prompt) // 4
            tokens_out = 100

            # Record usage in key manager
            key_manager.record_usage(api_key, tokens_in=tokens_in, tokens_out=tokens_out)

            # Record metrics if provided
            if metrics is not None:
                _record_llm_call(metrics, node_name, model, tokens_in, tokens_out)

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
# Content Landscape Analysis Node (Phase 0.6)
# =============================================================================


def _select_top_urls(topic_context: list[dict], max_count: int = 10) -> list[dict]:
    """
    Select top N URLs from topic context for content analysis.

    Args:
        topic_context: List of {title, url, snippet} from topic discovery
        max_count: Maximum URLs to select

    Returns:
        Top N URL entries (already deduplicated by topic_discovery)
    """
    return topic_context[:max_count]


def _format_article_summaries(articles: list[ExistingArticleSummary]) -> str:
    """
    Format analyzed articles for the strategy synthesis prompt.

    Args:
        articles: List of ExistingArticleSummary models

    Returns:
        Formatted string with article summaries
    """
    formatted = []
    for i, article in enumerate(articles, 1):
        strengths = ", ".join(article.strengths) if article.strengths else "None identified"
        weaknesses = ", ".join(article.weaknesses) if article.weaknesses else "None identified"
        key_points = ", ".join(article.key_points_covered) if article.key_points_covered else "None"

        formatted.append(f"""
Article {i}: {article.title}
URL: {article.url}
Angle: {article.main_angle}
Strengths: {strengths}
Weaknesses: {weaknesses}
Covers: {key_points}
""")
    return "\n".join(formatted)


async def _analyze_single_article(
    article: dict,
    blog_title: str,
    key_manager: KeyManager,
    max_retries: int = 3,
) -> ExistingArticleSummary | None:
    """
    Analyze a single article using LLM.

    Args:
        article: Dict with url, title, content
        blog_title: Blog topic being researched
        key_manager: API key manager
        max_retries: Maximum retry attempts

    Returns:
        ExistingArticleSummary model or None on failure
    """
    url = article.get("url", "")
    title = article.get("title", "Untitled")
    content = article.get("content", "")[:8000]  # Limit to 8k chars

    prompt = f"""Analyze this article on "{blog_title}":

URL: {url}
Title: {title}
Content (first 8k chars):
{content}

Extract:
1. main_angle: What unique perspective does this article take?
2. strengths: What does it do well? (2-3 points)
3. weaknesses: What's missing or weak? (2-3 points)
4. key_points_covered: Main topics/sections covered (3-5 bullet points)

Output as ExistingArticleSummary model."""

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_LITE,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_LOW,
            )

            structured_llm = llm.with_structured_output(ExistingArticleSummary)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=200,
            )

            return result

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    logger.error("All API keys exhausted during article analysis")
                    return None
                continue

            logger.error(f"Article analysis attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    logger.error(f"Failed to analyze article after {max_retries} attempts: {last_error}")
    return None


async def _analyze_article_with_semaphore(
    semaphore: asyncio.Semaphore,
    article: dict,
    blog_title: str,
    key_manager: KeyManager,
) -> ExistingArticleSummary | None:
    """
    Analyze single article with concurrency limit.

    Args:
        semaphore: Asyncio semaphore for concurrency control
        article: Article dict to analyze
        blog_title: Blog topic
        key_manager: API key manager

    Returns:
        ExistingArticleSummary or None
    """
    async with semaphore:
        return await _analyze_single_article(article, blog_title, key_manager)


async def _synthesize_content_strategy(
    blog_title: str,
    context: str,
    analyzed_articles: list[ExistingArticleSummary],
    key_manager: KeyManager,
    max_retries: int = 3,
) -> ContentStrategy:
    """
    Synthesize content strategy from analyzed articles.

    Args:
        blog_title: Blog topic
        context: User-provided context
        analyzed_articles: List of analyzed article summaries
        key_manager: API key manager
        max_retries: Maximum retry attempts

    Returns:
        ContentStrategy model

    Raises:
        RuntimeError: If synthesis fails after all retries
    """
    formatted_summaries = _format_article_summaries(analyzed_articles)

    prompt = f"""You are analyzing the content landscape for a blog on: "{blog_title}"

Context provided by user:
{context}

I've analyzed {len(analyzed_articles)} top articles on this topic.
Here's what they cover:

{formatted_summaries}

Your task: Create a ContentStrategy that ensures our blog is UNIQUE and VALUABLE.

Requirements:
1. Identify 3-5 content gaps (what's missing, shallow, or wrong in existing articles)
2. Choose a unique_angle that fills these gaps (e.g., "focus on production pitfalls others ignore", "provide concrete benchmarks", "compare 3 real implementations")
3. Select target_persona (who needs this most: junior_engineer, senior_architect, data_scientist, devops_engineer)
4. Define reader_problem (specific problem they're solving)
5. List differentiation_requirements (specific things our blog MUST include to stand out)

Examples of good unique angles:
- "Production-focused guide covering error handling, monitoring, and scaling (others focus only on basics)"
- "Benchmark-driven comparison of 3 caching strategies with real numbers (others are purely theoretical)"
- "Cost optimization angle: how to reduce LLM API bills by 60% (others don't mention costs)"
- "Edge cases and limitations deep-dive (others only show happy path)"

Output as ContentStrategy model."""

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_LITE,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_MEDIUM,
            )

            structured_llm = llm.with_structured_output(ContentStrategy)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: structured_llm.invoke(prompt)
            )

            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=500,
            )

            return result

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

            logger.error(f"Strategy synthesis attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(f"Failed to synthesize content strategy after {max_retries} attempts: {last_error}")


async def content_landscape_analysis_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 0.6: Content Landscape Analysis Node.

    Analyzes top existing articles to identify content gaps and determine
    a unique angle for differentiation BEFORE planning.

    Process:
    1. Select top 10 URLs from topic_context
    2. Fetch full content for each (via fetch_url_content) - skip on failure
    3. LLM analyzes each article → ExistingArticleSummary (3 concurrent via semaphore)
    4. LLM synthesizes all → ContentStrategy
    5. Save to content_strategy.json

    Args:
        state: Current BlogAgentState containing:
            - title: Blog title (required)
            - context: User context (required)
            - topic_context: List of {title, url, snippet} from discovery
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - content_strategy: ContentStrategy as dict
        - current_phase: Updated to PLANNING
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.CONTENT_LANDSCAPE):
        logger.info("Skipping content_landscape_analysis_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "content_landscape"
    metrics, start_time = _init_node_metrics(state, node_name)
    logger.info(f"Starting content landscape analysis for: {state.get('title')}")

    title = state.get("title", "")
    context = state.get("context", "")
    topic_context = state.get("topic_context", [])
    job_id = state.get("job_id", "")

    if not title:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Title is required for content landscape analysis",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    if not topic_context:
        logger.warning("No topic context available, creating default content strategy without landscape insights")

        # Create minimal default strategy (as dict, bypassing Pydantic validation)
        # This ensures content_strategy is always available, even when landscape analysis skips
        strategy_dict = {
            "unique_angle": "Practical technical guide with concrete examples",
            "target_persona": "senior_engineer",
            "reader_problem": f"Understanding and implementing {title}",
            "gaps_to_fill": [
                {
                    "gap_type": "insufficient_depth",
                    "description": "No landscape analysis available",
                    "opportunity": "Provide comprehensive coverage based on user context",
                }
            ],
            "existing_content_summary": "No existing content analyzed due to missing topic context",
            "analyzed_articles": [],  # Empty - no articles available
            "differentiation_requirements": [
                "Include working code examples",
                "Focus on practical implementation details",
            ],
        }

        # Save checkpoint if job_id exists
        if job_id:
            job_manager = JobManager()
            job_dir = job_manager.get_job_dir(job_id)

            import json
            strategy_path = job_dir / "content_strategy.json"
            with open(strategy_path, "w") as f:
                json.dump(strategy_dict, f, indent=2)
            logger.info(f"Saved default content strategy to {strategy_path}")

            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.PLANNING.value,
                    "content_strategy": strategy_dict,
                },
            )

        return {
            "content_strategy": strategy_dict,
            "current_phase": Phase.PLANNING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    try:
        key_manager = KeyManager.from_env()

        # Step 1: Select top 10 URLs
        top_urls = _select_top_urls(topic_context, max_count=10)
        logger.info(f"Selected {len(top_urls)} URLs for content analysis")

        # Step 2: Fetch full content for each URL
        articles_content = []
        for url_data in top_urls:
            url = url_data.get("url", "")
            if not url:
                continue

            try:
                result = await fetch_url_content(url)
                if result.get("success") and result.get("content"):
                    articles_content.append({
                        "url": url,
                        "title": url_data.get("title", result.get("title", "Untitled")),
                        "content": result["content"],
                    })
                    logger.info(f"Fetched content from: {url}")
                else:
                    logger.warning(f"Failed to fetch content from {url}: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                continue

        logger.info(f"Successfully fetched {len(articles_content)} articles")

        if len(articles_content) < 3:
            logger.warning(f"Only {len(articles_content)} articles fetched, proceeding with limited analysis")

        # Step 3: Analyze each article with LLM (3 concurrent)
        semaphore = asyncio.Semaphore(3)
        tasks = [
            _analyze_article_with_semaphore(semaphore, article, title, key_manager)
            for article in articles_content
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None results
        analyzed_articles = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Article analysis failed with exception: {result}")
            elif result is not None:
                analyzed_articles.append(result)

        # Record metrics for article analysis (estimate: ~2000 tokens in, 500 out per article)
        num_analyzed = len(analyzed_articles)
        if num_analyzed > 0:
            tokens_in = num_analyzed * 2000
            tokens_out = num_analyzed * 500
            _record_llm_call(metrics, node_name, LLM_MODEL_LITE, tokens_in, tokens_out)

        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles")

        if len(analyzed_articles) < 3:
            logger.warning("Insufficient articles analyzed, creating minimal strategy")
            # Create a minimal content strategy as dict (bypassing validation)
            # Since we don't have enough articles, we return a simplified strategy
            strategy_dict = {
                "unique_angle": "Focus on practical implementation with concrete examples",
                "target_persona": "senior_engineer",
                "reader_problem": f"Understanding and implementing {title}",
                "gaps_to_fill": [
                    {
                        "gap_type": "insufficient_depth",
                        "description": "Limited content available for comprehensive analysis",
                        "opportunity": "Provide thorough coverage of the topic",
                    }
                ],
                "existing_content_summary": "Limited existing content available for analysis",
                "analyzed_articles": [a.model_dump() for a in analyzed_articles],
                "differentiation_requirements": [
                    "Include working code examples",
                    "Provide concrete benchmarks where applicable",
                ],
            }

            # Step 5: Save checkpoint
            if job_id:
                job_manager = JobManager()
                job_dir = job_manager.get_job_dir(job_id)

                # Save content_strategy.json
                import json
                strategy_path = job_dir / "content_strategy.json"
                with open(strategy_path, "w") as f:
                    json.dump(strategy_dict, f, indent=2)
                logger.info(f"Saved minimal content strategy to {strategy_path}")

                job_manager.save_state(
                    job_id,
                    {
                        "current_phase": Phase.PLANNING.value,
                        "content_strategy": strategy_dict,
                    },
                )

            return {
                "content_strategy": strategy_dict,
                "current_phase": Phase.PLANNING.value,
                "metrics": _finalize_node_metrics(metrics, node_name, start_time),
            }
        else:
            # Step 4: Synthesize content strategy
            content_strategy = await _synthesize_content_strategy(
                blog_title=title,
                context=context,
                analyzed_articles=analyzed_articles,
                key_manager=key_manager,
            )
            # Record synthesis metrics (estimate: ~3000 tokens in, 500 out)
            _record_llm_call(metrics, node_name, LLM_MODEL_LITE, 3000, 500)

        logger.info(f"Content strategy: unique_angle='{content_strategy.unique_angle}'")
        logger.info(f"Gaps to fill: {len(content_strategy.gaps_to_fill)}")

        # Convert to dict for state storage
        strategy_dict = content_strategy.model_dump()

        # Step 5: Save checkpoint
        if job_id:
            job_manager = JobManager()
            job_dir = job_manager.get_job_dir(job_id)

            # Save content_strategy.json
            import json
            strategy_path = job_dir / "content_strategy.json"
            with open(strategy_path, "w") as f:
                json.dump(strategy_dict, f, indent=2)
            logger.info(f"Saved content strategy to {strategy_path}")

            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.PLANNING.value,
                    "content_strategy": strategy_dict,
                },
            )

        return {
            "content_strategy": strategy_dict,
            "current_phase": Phase.PLANNING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except RuntimeError as e:
        logger.error(f"Content landscape analysis failed: {e}")
        logger.warning("Creating default content strategy to allow pipeline to continue")

        # Create minimal default strategy instead of failing
        strategy_dict = {
            "unique_angle": "Practical technical guide with concrete examples",
            "target_persona": "senior_engineer",
            "reader_problem": f"Understanding and implementing {title}",
            "gaps_to_fill": [
                {
                    "gap_type": "insufficient_depth",
                    "description": "Landscape analysis failed",
                    "opportunity": "Provide comprehensive coverage based on user context",
                }
            ],
            "existing_content_summary": f"Content analysis failed: {e}",
            "analyzed_articles": [],
            "differentiation_requirements": [
                "Include working code examples",
                "Focus on practical implementation details",
            ],
        }

        # Save checkpoint if job_id exists
        if job_id:
            job_manager = JobManager()
            job_dir = job_manager.get_job_dir(job_id)

            import json
            strategy_path = job_dir / "content_strategy.json"
            with open(strategy_path, "w") as f:
                json.dump(strategy_dict, f, indent=2)
            logger.info(f"Saved default content strategy to {strategy_path}")

            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.PLANNING.value,
                    "content_strategy": strategy_dict,
                },
            )

        return {
            "content_strategy": strategy_dict,
            "current_phase": Phase.PLANNING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    except Exception as e:
        logger.error(f"Unexpected error in content landscape analysis: {e}")
        logger.warning("Creating default content strategy to allow pipeline to continue")

        # Create minimal default strategy instead of failing
        strategy_dict = {
            "unique_angle": "Practical technical guide with concrete examples",
            "target_persona": "senior_engineer",
            "reader_problem": f"Understanding and implementing {title}",
            "gaps_to_fill": [
                {
                    "gap_type": "insufficient_depth",
                    "description": "Landscape analysis encountered an error",
                    "opportunity": "Provide comprehensive coverage based on user context",
                }
            ],
            "existing_content_summary": f"Content analysis failed with error: {e}",
            "analyzed_articles": [],
            "differentiation_requirements": [
                "Include working code examples",
                "Focus on practical implementation details",
            ],
        }

        # Save checkpoint if job_id exists
        if job_id:
            job_manager = JobManager()
            job_dir = job_manager.get_job_dir(job_id)

            import json
            strategy_path = job_dir / "content_strategy.json"
            with open(strategy_path, "w") as f:
                json.dump(strategy_dict, f, indent=2)
            logger.info(f"Saved default content strategy to {strategy_path}")

            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.PLANNING.value,
                    "content_strategy": strategy_dict,
                },
            )

        return {
            "content_strategy": strategy_dict,
            "current_phase": Phase.PLANNING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


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


def _format_content_gaps(gaps: list[dict]) -> str:
    """Format content gaps for planning prompt."""
    if not gaps:
        return "No specific gaps identified."

    formatted = []
    for i, gap in enumerate(gaps, 1):
        gap_type = gap.get("gap_type", "unknown")
        description = gap.get("description", "")
        opportunity = gap.get("opportunity", "")
        formatted.append(f"{i}. [{gap_type}] {description}\n   → Opportunity: {opportunity}")
    return "\n".join(formatted)


def _format_differentiation_requirements(requirements: list[str]) -> str:
    """Format differentiation requirements for planning prompt."""
    if not requirements:
        return "No specific requirements."
    return "\n".join(f"- {req}" for req in requirements)


def _build_planning_prompt(
    title: str,
    context: str,
    target_words: int,
    topic_snippets: str,
    content_strategy: dict | None = None,
    replanning_feedback: str | None = None,
    rejected_sections: list[dict[str, Any]] | None = None,
    scratchpad: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build the planning prompt for Gemini.

    Args:
        title: Blog title
        context: User-provided context
        target_words: Total target word count
        topic_snippets: Formatted topic context snippets
        content_strategy: ContentStrategy dict from landscape analysis (optional)
        replanning_feedback: Feedback from preview validation if replanning
        rejected_sections: Sections that failed validation
        scratchpad: Planning iteration history to prevent repeating mistakes

    Returns:
        Complete prompt string
    """
    # Build gap mapping section if content_strategy available
    gap_mapping_section = ""
    if content_strategy:
        gaps = content_strategy.get("gaps_to_fill", [])
        gap_examples = []
        for gap in gaps[:2]:  # Show 2 examples
            gap_type = gap.get("gap_type", "")
            description = gap.get("description", "")
            gap_examples.append(f'  - gap_type: "{gap_type}"\n    description: "{description}"')

        gap_mapping_section = f"""
## CRITICAL GAP MAPPING REQUIREMENT

For EACH section in your plan, you MUST:
1. Set gap_addressed field to one of the gap_types from the content gaps above
2. Provide gap_justification explaining HOW this section fills that specific gap

Example:
{{
  "id": "cache-invalidation",
  "title": "Cache Invalidation Strategies",
  "gap_addressed": "missing_edge_cases",
  "gap_justification": "Existing articles skip invalidation. This section covers TTL strategies, manual invalidation, and consistency patterns."
}}

**ALL content gaps must be addressed by at least one section.**
**Sections without proper gap mapping will be REJECTED.**
"""

    # Build uniqueness verification section
    uniqueness_section = ""
    if content_strategy:
        analyzed_articles = content_strategy.get("analyzed_articles", [])
        article_summaries = []
        for article in analyzed_articles[:3]:  # Show top 3
            title = article.get("title", "Unknown")
            key_points = article.get("key_points_covered", [])
            points_str = ", ".join(key_points[:3])
            article_summaries.append(f"  - {title}: {points_str}")

        uniqueness_section = f"""
## UNIQUENESS VERIFICATION

Analyzed articles cover these topics:
{chr(10).join(article_summaries)}

Your sections MUST NOT overlap >70% with these existing articles. Ensure uniqueness by:
- Taking a different angle or perspective
- Going deeper into specifics they skip
- Covering edge cases and production considerations they miss
- Providing concrete examples and code they don't have

Sections that duplicate existing content will be REJECTED.
"""

    # Build planning history section from scratchpad
    planning_history_section = ""
    if scratchpad and len(scratchpad) > 0:
        history_entries = []
        for entry in scratchpad:
            iteration = entry.get("iteration", 0)
            plan_snapshot = entry.get("plan_snapshot", {})
            passed = entry.get("passed", False)
            rejected_ids = entry.get("rejected_section_ids", [])
            feedback_text = entry.get("feedback_text", "")

            # Show what sections were tried
            sections_tried = plan_snapshot.get("sections", [])
            section_summaries = []
            for s in sections_tried:
                section_id = s.get("id", "")
                title = s.get("title", "")
                gap = s.get("gap_addressed", "")
                status = "✓ PASSED" if section_id not in rejected_ids else "✗ REJECTED"
                section_summaries.append(f"    - [{status}] {section_id}: \"{title}\" (gap: {gap})")

            sections_text = "\n".join(section_summaries) if section_summaries else "    (no sections)"

            # Build entry text
            if passed:
                status_text = "✓ PASSED VALIDATION"
            else:
                status_text = f"✗ FAILED - Rejected: {', '.join(rejected_ids)}"

            entry_text = f"""
### Iteration {iteration + 1}: {status_text}
**Sections Attempted:**
{sections_text}

**What Went Wrong:**
{feedback_text if feedback_text else "(validation passed)"}
"""
            history_entries.append(entry_text)

        planning_history_section = f"""
## PLANNING ITERATION HISTORY

You have made {len(scratchpad)} previous attempt(s) at planning this blog. Learn from past failures:

{''.join(history_entries)}

**CRITICAL INSTRUCTION:**
- DO NOT repeat the same mistakes from previous iterations
- If a section was rejected for low information availability, do NOT propose similar obscure topics
- If a section had high overlap, do NOT propose the same angle again
- If gap mapping failed, ensure ALL sections map to valid gaps this time
- Review the feedback carefully and address ALL issues raised
"""

    # Build replanning section if this is a replanning iteration
    replanning_section = ""
    if replanning_feedback:
        rejected_ids = [s.get("id", "") for s in (rejected_sections or [])]
        rejected_list = ", ".join(rejected_ids) if rejected_ids else "None"

        replanning_section = f"""
## REPLANNING REQUIRED

Your previous plan failed validation. Issues found:

{replanning_feedback}

**Instructions for replanning:**
1. Review the iteration history above to avoid repeating mistakes
2. Replace or improve the rejected sections
3. Ensure ALL content gaps are properly addressed
4. Verify new sections have strong information availability
5. Avoid overlap with analyzed articles (check uniqueness)
6. Properly set gap_addressed and gap_justification for ALL sections

Rejected section IDs: {rejected_list}
"""

    # Build content strategy section if available
    strategy_section = ""
    if content_strategy:
        unique_angle = content_strategy.get("unique_angle", "N/A")
        target_persona = content_strategy.get("target_persona", "experienced engineer")
        reader_problem = content_strategy.get("reader_problem", "N/A")
        gaps = content_strategy.get("gaps_to_fill", [])
        requirements = content_strategy.get("differentiation_requirements", [])
        existing_summary = content_strategy.get("existing_content_summary", "N/A")

        strategy_section = f"""
## CONTENT STRATEGY (from landscape analysis)
**Unique Angle**: {unique_angle}
**Target Reader**: {target_persona}
**Reader Problem**: {reader_problem}

**What top articles already cover**:
{existing_summary}

**Content Gaps to Fill**:
{_format_content_gaps(gaps)}

**Differentiation Requirements** (MUST include these):
{_format_differentiation_requirements(requirements)}

**IMPORTANT**: Your plan must:
1. Take the unique angle identified above
2. Fill the content gaps others missed
3. Meet ALL differentiation requirements
4. Serve the target reader's specific problem
5. Do NOT just rehash what existing articles already cover well
"""

    return f"""You are planning a technical blog post with short, punchy writing. Each section should be FOCUSED and SPECIFIC.

Blog structure should be:
1. Hook (optional) - brief problem statement or context (2-3 paragraphs, no heading)
2. Problem/Why section - what's broken and why it matters (can combine into one)
3. Core Implementation - code-heavy section showing the main approach
4. Technique sections - 3-5 specific techniques/patterns (e.g., "Validate Before Storing", "Tune Similarity Threshold")
5. Drawbacks/Tradeoffs (if relevant) - honest limitations
6. Conclusion - brief practical takeaways

CRITICAL: SECTION GRANULARITY FOR PUNCHY WRITING
- Each section covers ONE specific technique or idea
- Break broad topics into focused, bite-sized sections
- Granular sections = naturally shorter, punchier writing

Examples:
❌ BAD (too broad): "Caching Strategies and Performance Tuning"
   → Forces long, complex explanations that kill punchy style

✅ GOOD (granular, focused):
   → "Cache Lookups" (one section)
   → "Similarity Thresholds" (separate section)
   → "Performance Impact" (separate section)
   Each section stays focused, sentences stay short

❌ BAD: "Advanced Redis Features"
✅ GOOD: "Vector Search with RediSearch" + "TTL Management"

IMPORTANT GUIDELINES:
- Prefer granular, technique-focused sections over generic theory sections
- Each technique section should have a specific, actionable title (not "Best Practices")
- Implementation sections MUST have needs_code=true
- Code should be 50-70% of implementation sections (PRIMARY content)
- Include a "Drawbacks" or "Tradeoffs" section when the topic has real limitations
- Limit to 6-8 sections MAXIMUM (prevents over-slicing)

For each section, provide:
- id (unique string identifier, e.g., "hook", "problem", "implementation")
- title (section heading, null for hook)
- role (hook/problem/why/implementation/deep_dive/conclusion/tradeoffs)
- search_queries (2-3 specific queries - prefer "X code example", "X github", "X implementation")
- gap_addressed (REQUIRED if content_strategy available - which gap_type this section fills)
- gap_justification (REQUIRED if content_strategy available - explain HOW this section fills the gap)
- needs_code (true for ALL implementation/deep_dive sections)
- needs_diagram (true if architecture/flow explanation needed)
- target_words (distribute {target_words} total across REQUIRED sections only)
- optional (true for extra sections user can choose from)

## Required vs Optional Sections
- hook, problem/why, conclusion: ALWAYS set optional=false (3-4 sections)
- Core implementation: ALWAYS set optional=false (1 section showing primary approach)
- Deep dives/comparisons: Generate 7-10 total, ALL marked optional=true
  - Each should be self-contained and address a unique angle
  - Examples: "Redis vs Alternatives", "Production Pitfalls", "Benchmarking", "Advanced Patterns", "Edge Cases"
- tradeoffs section: optional=false if topic has significant drawbacks

{gap_mapping_section}
{uniqueness_section}
{planning_history_section}
{replanning_section}

## Blog Topic
Title: "{title}"
Context: {context}
{strategy_section}
## Topic Research (from web search)
The following snippets provide current context about this topic:
{topic_snippets}

Use this research to inform your subtopic selection.
Focus on aspects that appear important based on this context.

Generate a blog plan with 10-15 sections total:
- 3-5 REQUIRED sections (hook, problem/why, core implementation, conclusion): optional=false
- 7-10 OPTIONAL sections (deep dives, comparisons, edge cases, production concerns): optional=true

User will select 2-4 optional sections to include (final blog will have 5-7 sections).

For optional sections:
- Each should address a distinct aspect or technique
- Include variety: comparisons, benchmarks, production pitfalls, advanced patterns, edge cases
- Provide strong gap_justification so user understands value

Ensure target_words for required sections sum to approximately {target_words}.
Output as structured BlogPlan."""


async def _generate_blog_plan(
    title: str,
    context: str,
    target_words: int,
    topic_context: list[dict[str, str]],
    key_manager: KeyManager,
    content_strategy: dict | None = None,
    replanning_feedback: str | None = None,
    rejected_sections: list[dict[str, Any]] | None = None,
    scratchpad: list[dict[str, Any]] | None = None,
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
        content_strategy: ContentStrategy dict from landscape analysis (optional)
        replanning_feedback: Feedback from preview validation if replanning
        rejected_sections: Sections that failed validation
        scratchpad: Planning iteration history to prevent repeating mistakes
        max_retries: Maximum retry attempts

    Returns:
        BlogPlan Pydantic model with sections

    Raises:
        RuntimeError: If all API keys exhausted or max retries exceeded
    """
    # Format topic snippets
    topic_snippets = _format_topic_context_snippets(topic_context)

    # Build prompt with content strategy and replanning context
    prompt = _build_planning_prompt(
        title, context, target_words, topic_snippets, content_strategy,
        replanning_feedback, rejected_sections, scratchpad
    )

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
    target length, topic context (from discovery phase), and content strategy
    (from landscape analysis phase).

    Args:
        state: Current BlogAgentState containing:
            - title: Blog title (required)
            - context: User context (required)
            - target_length: "short" | "medium" | "long" (default "medium")
            - topic_context: List of {title, url, snippet} from discovery
            - content_strategy: ContentStrategy dict from landscape analysis (optional)
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - plan: BlogPlan as dict
        - current_phase: Updated to RESEARCHING
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.PLANNING):
        logger.info("Skipping planning_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "planning"
    metrics, start_time = _init_node_metrics(state, node_name)

    # Get planning iteration (for replanning loop)
    iteration = state.get("planning_iteration", 0)
    replanning_feedback = state.get("replanning_feedback", "")
    rejected_sections = state.get("rejected_sections", [])
    preview_validation_scratchpad = state.get("preview_validation_scratchpad", [])

    if iteration > 0:
        logger.info(f"Replanning (iteration {iteration + 1}/3) for: {state.get('title')}")
    else:
        logger.info(f"Starting planning for: {state.get('title')}")

    title = state.get("title", "")
    context = state.get("context", "")
    target_length = state.get("target_length", "medium")
    topic_context = state.get("topic_context", [])
    content_strategy = state.get("content_strategy")
    job_id = state.get("job_id", "")

    # Content strategy is recommended but not strictly required
    if not content_strategy:
        logger.warning("Content strategy not available - planning without gap mapping")
        logger.warning("Landscape analysis may have been skipped due to API quota or missing topic context")
    else:
        logger.info(f"Using content strategy: unique_angle='{content_strategy.get('unique_angle')}'")

    # Validate required inputs
    if not title:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Title is required for planning",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    # Map target_length to target_words
    target_words = TARGET_WORDS_MAP.get(target_length, 1500)

    try:
        # Initialize key manager from environment
        key_manager = KeyManager.from_env()

        # Generate blog plan using LLM with content strategy and replanning context
        plan = await _generate_blog_plan(
            title=title,
            context=context,
            target_words=target_words,
            topic_context=topic_context,
            key_manager=key_manager,
            content_strategy=content_strategy,
            replanning_feedback=replanning_feedback if replanning_feedback else None,
            rejected_sections=rejected_sections if rejected_sections else None,
            scratchpad=preview_validation_scratchpad if preview_validation_scratchpad else None,
        )

        # Record LLM call metrics (estimate: ~1500 tokens in, 500 out for plan)
        _record_llm_call(metrics, node_name, LLM_MODEL_LITE, 1500, 500)

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
                    "current_phase": Phase.PREVIEW_VALIDATION.value,
                    "plan": plan_dict,
                    "planning_iteration": iteration,
                },
            )

        return {
            "plan": plan_dict,
            "current_phase": Phase.PREVIEW_VALIDATION.value,
            "planning_iteration": iteration,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except RuntimeError as e:
        logger.error(f"Planning failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    except Exception as e:
        logger.error(f"Unexpected error in planning: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


# =============================================================================
# Preview Validation Node (Phase 1.5)
# =============================================================================


def _validate_gap_mapping(
    plan: dict[str, Any], content_strategy: dict[str, Any]
) -> dict[str, Any]:
    """
    Validate that all content gaps are addressed by sections.

    Args:
        plan: BlogPlan as dict
        content_strategy: ContentStrategy as dict

    Returns:
        Dict with validation results
    """
    sections = plan.get("sections", [])
    gaps_to_fill = content_strategy.get("gaps_to_fill", [])

    # Extract gap types from ContentStrategy
    gap_types = {gap.get("gap_type") for gap in gaps_to_fill}

    # Extract gap types addressed by sections
    sections_with_gaps = [s for s in sections if s.get("gap_addressed")]
    addressed_gaps = {s.get("gap_addressed") for s in sections_with_gaps}

    # Find missing gaps
    missing_gaps = gap_types - addressed_gaps

    # Find sections without gap mapping
    sections_without_mapping = [
        s.get("id") for s in sections if not s.get("gap_addressed")
    ]

    all_gaps_covered = len(missing_gaps) == 0 and len(sections_without_mapping) == 0

    return {
        "all_gaps_covered": all_gaps_covered,
        "missing_gaps": list(missing_gaps),
        "sections_without_mapping": sections_without_mapping,
        "total_gaps": len(gap_types),
        "addressed_gaps": len(addressed_gaps),
    }


async def _test_single_section_feasibility(
    section: dict[str, Any], semaphore: asyncio.Semaphore
) -> SectionFeasibilityScore:
    """
    Test information availability for a single section.

    Args:
        section: PlanSection as dict
        semaphore: Semaphore to limit concurrency

    Returns:
        SectionFeasibilityScore with feasibility assessment
    """
    async with semaphore:
        section_id = section.get("id", "unknown")
        search_queries = section.get("search_queries", [])

        # Select 1-2 sample queries to test
        sample_queries = search_queries[:2] if len(search_queries) >= 2 else search_queries

        if not sample_queries:
            return SectionFeasibilityScore(
                section_id=section_id,
                sample_queries_tested=[],
                sources_found=0,
                snippet_quality="poor",
                information_availability=0,
                is_feasible=False,
                concerns=["No search queries defined for this section"],
            )

        # Execute searches
        all_results = []
        for query in sample_queries:
            try:
                results = await search_duckduckgo(query, max_results=5)
                all_results.extend(results)
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        sources_found = len(all_results)

        # Assess snippet quality
        snippet_quality = _assess_snippet_quality(all_results)

        # Calculate information availability score (0-100)
        score = 0
        concerns = []

        if sources_found >= 8 and snippet_quality in ["excellent", "good"]:
            score = 90
        elif sources_found >= 5 and snippet_quality in ["excellent", "good"]:
            score = 75
        elif sources_found >= 3 and snippet_quality in ["good", "weak"]:
            score = 60
        elif sources_found >= 3:
            score = 50
            concerns.append(f"Found {sources_found} sources but snippet quality is {snippet_quality}")
        else:
            score = max(20, sources_found * 10)
            concerns.append(f"Only {sources_found} sources found (need 3+)")

        is_feasible = score >= 60

        if not is_feasible and not concerns:
            concerns.append(f"Information availability score {score} below threshold 60")

        return SectionFeasibilityScore(
            section_id=section_id,
            sample_queries_tested=sample_queries,
            sources_found=sources_found,
            snippet_quality=snippet_quality,
            information_availability=score,
            is_feasible=is_feasible,
            concerns=concerns,
        )


def _assess_snippet_quality(results: list[dict[str, str]]) -> str:
    """
    Assess quality of search result snippets.

    Args:
        results: List of search results with 'snippet' field

    Returns:
        Quality rating: "excellent" | "good" | "weak" | "poor"
    """
    if not results:
        return "poor"

    snippets = [r.get("snippet", "") for r in results]
    avg_length = sum(len(s) for s in snippets) / len(snippets)

    # Count technical indicators (code, numbers, specific terms)
    technical_indicators = 0
    for snippet in snippets:
        if any(
            indicator in snippet.lower()
            for indicator in ["code", "example", "implementation", "function", "class", "api"]
        ):
            technical_indicators += 1

    technical_ratio = technical_indicators / len(snippets) if snippets else 0

    # Quality assessment
    if avg_length > 150 and technical_ratio >= 0.5:
        return "excellent"
    elif avg_length > 100 and technical_ratio >= 0.3:
        return "good"
    elif avg_length > 50:
        return "weak"
    else:
        return "poor"


async def _test_section_feasibility_parallel(
    sections: list[dict[str, Any]], max_concurrent: int = 3
) -> list[SectionFeasibilityScore]:
    """
    Test feasibility of all sections in parallel.

    Args:
        sections: List of PlanSection dicts
        max_concurrent: Maximum concurrent searches

    Returns:
        List of SectionFeasibilityScore results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_test_single_section_feasibility(section, semaphore) for section in sections]
    return await asyncio.gather(*tasks)


def _build_uniqueness_prompt(
    sections: list[dict[str, Any]], content_strategy: dict[str, Any]
) -> str:
    """
    Build prompt for LLM-based uniqueness checking.

    Args:
        sections: List of PlanSection dicts
        content_strategy: ContentStrategy dict with analyzed_articles

    Returns:
        Formatted prompt string
    """
    analyzed_articles = content_strategy.get("analyzed_articles", [])

    # Format sections
    sections_text = []
    for section in sections:
        sections_text.append(f"""
**Section ID**: {section.get('id', 'unknown')}
**Title**: {section.get('title', 'N/A')}
**Role**: {section.get('role', 'N/A')}
**Gap Addressed**: {section.get('gap_addressed', 'N/A')}
**Gap Justification**: {section.get('gap_justification', 'N/A')}
**Search Queries**: {', '.join(section.get('search_queries', []))}
""")

    # Format analyzed articles
    articles_text = []
    for article in analyzed_articles:
        key_points = "\n  - ".join(article.get("key_points_covered", []))
        strengths = "\n  - ".join(article.get("strengths", []))
        weaknesses = "\n  - ".join(article.get("weaknesses", []))

        articles_text.append(f"""
**Article**: {article.get('title', 'Unknown')}
**URL**: {article.get('url', 'N/A')}
**Main Angle**: {article.get('main_angle', 'N/A')}
**Key Points Covered**:
  - {key_points}
**Strengths**:
  - {strengths}
**Weaknesses**:
  - {weaknesses}
""")

    return f"""You are a technical content analyst evaluating a blog plan for uniqueness.

## TASK
Analyze each planned section and determine if it overlaps >70% with existing articles.

## PLANNED SECTIONS
{chr(10).join(sections_text)}

## ANALYZED EXISTING ARTICLES
{chr(10).join(articles_text)}

## EVALUATION CRITERIA

For each section, determine overlap percentage (0-100):
- **0-30%**: Completely unique angle or topic
- **31-50%**: Related but different focus
- **51-70%**: Similar but adds new perspective (ACCEPTABLE)
- **71-100%**: Substantive overlap, appears to rehash existing content (REJECT)

Consider:
- Is the section covering the same specific topic/technique as an existing article?
- Is the angle substantively different from what existing articles cover?
- Does it add genuinely new information or just restate existing content?
- **Topical similarity is OK** if the approach/perspective is genuinely different
- Focus on **substantive overlap**, not just shared keywords

For each section, identify:
1. overlap_percentage (0-100)
2. is_unique (true if <=70%, false if >70%)
3. overlapping_articles (list of URLs with high overlap)
4. concerns (what content is being rehashed, if applicable)

Provide an overall_assessment summarizing uniqueness across all sections.

Output structured UniquenessAnalysisResult."""


async def _check_sections_uniqueness_llm(
    sections: list[dict[str, Any]],
    content_strategy: dict[str, Any],
    key_manager: KeyManager,
    max_retries: int = 3,
) -> UniquenessAnalysisResult:
    """
    LLM-based uniqueness checking using Gemini Flash.

    Args:
        sections: List of PlanSection dicts
        content_strategy: ContentStrategy dict with analyzed_articles
        key_manager: KeyManager for API key rotation
        max_retries: Maximum retry attempts

    Returns:
        UniquenessAnalysisResult with uniqueness checks for all sections
    """
    prompt = _build_uniqueness_prompt(sections, content_strategy)

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            # Initialize LLM with structured output
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.3,  # Lower temp for consistent evaluation
            )

            structured_llm = llm.with_structured_output(UniquenessAnalysisResult)

            # Invoke
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

            logger.info(f"LLM uniqueness check complete: {len(result.uniqueness_checks)} sections analyzed")
            return result

        except Exception as e:
            error_str = str(e).lower()
            last_error = e

            # Check for rate limit
            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            # For other errors, retry with backoff
            logger.error(f"Uniqueness check attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(
        f"Failed to check uniqueness after {max_retries} attempts: {last_error}"
    )


def _build_replanning_feedback_prompt(
    plan: dict[str, Any],
    content_strategy: dict[str, Any],
    gap_validation: dict[str, Any] | None = None,
    weak_sections: list[SectionFeasibilityScore] | None = None,
    non_unique_sections: list[UniquenessCheck] | None = None,
) -> str:
    """
    Build prompt for LLM to generate concrete replanning feedback.

    Args:
        plan: Original blog plan with sections
        content_strategy: Content strategy with gaps and analyzed articles
        gap_validation: Gap mapping validation results
        weak_sections: Sections with low feasibility scores
        non_unique_sections: Sections with high overlap

    Returns:
        Formatted prompt string for LLM
    """
    sections = plan.get("sections", [])
    gaps = content_strategy.get("gaps", [])

    prompt_parts = [
        "You are a blog planning validator. A blog plan has been created but some sections failed validation.",
        "Your task is to provide CONCRETE, ACTIONABLE suggestions to fix each rejected section.",
        "",
        "## Original Blog Plan",
        f"**Topic**: {plan.get('topic', 'Unknown')}",
        f"**Target Length**: {plan.get('target_length', 'medium')}",
        f"**Total Sections**: {len(sections)}",
        "",
        "## Content Strategy Gaps to Address",
        "These are the unique angles/gaps we identified from analyzing existing articles:",
    ]

    for i, gap in enumerate(gaps, 1):
        prompt_parts.append(f"{i}. **{gap.get('gap', 'Unknown')}**: {gap.get('description', '')}")

    prompt_parts.extend(
        [
            "",
            "## Validation Failures",
            "",
        ]
    )

    # Gap mapping issues
    if gap_validation and not gap_validation.get("all_gaps_covered"):
        missing_gaps = gap_validation.get("missing_gaps", [])
        sections_without_mapping = gap_validation.get("sections_without_mapping", [])

        if missing_gaps:
            prompt_parts.extend(
                [
                    "### Gap Mapping Issues",
                    f"**Missing Gaps**: {', '.join(missing_gaps)}",
                    "These content gaps are not addressed by any section in the plan.",
                    "",
                ]
            )

        if sections_without_mapping:
            prompt_parts.extend(
                [
                    f"**Sections Without Gap Mapping**: {', '.join(sections_without_mapping)}",
                    "These sections don't map to any identified gap.",
                    "",
                ]
            )

    # Weak sections
    if weak_sections:
        prompt_parts.extend(
            [
                "### Weak Sections (Low Information Availability)",
                "These sections have insufficient information available online:",
                "",
            ]
        )

        for score in weak_sections:
            section_info = next((s for s in sections if s.get("id") == score.section_id), {})
            prompt_parts.extend(
                [
                    f"**Section ID**: {score.section_id}",
                    f"**Title**: {section_info.get('title', 'Unknown')}",
                    f"**Role**: {section_info.get('role', 'Unknown')}",
                    f"**Information Score**: {score.information_availability}/100",
                    f"**Concerns**: {'; '.join(score.concerns)}",
                    f"**Search Queries Tried**: {', '.join(section_info.get('search_queries', []))}",
                    "",
                ]
            )

    # Non-unique sections
    if non_unique_sections:
        prompt_parts.extend(
            [
                "### Non-Unique Sections (High Overlap with Existing Articles)",
                "These sections are too similar to content that already exists:",
                "",
            ]
        )

        for check in non_unique_sections:
            section_info = next((s for s in sections if s.get("id") == check.section_id), {})
            prompt_parts.extend(
                [
                    f"**Section ID**: {check.section_id}",
                    f"**Title**: {section_info.get('title', 'Unknown')}",
                    f"**Overlap**: {check.overlap_percentage:.1f}%",
                    f"**Concerns**: {check.concerns}",
                    "",
                ]
            )

    prompt_parts.extend(
        [
            "## Your Task",
            "",
            "For EACH rejected section, provide:",
            "1. **Issue**: Clear explanation of why it was rejected",
            "2. **Suggested Title**: Alternative section title that addresses a different angle",
            "3. **Suggested Angle**: Specific approach to take (e.g., 'Focus on X instead of Y')",
            "4. **Alternative Gap** (optional): If the section can't be salvaged, suggest a different gap to address",
            "",
            "Make your suggestions CONCRETE and SPECIFIC:",
            "- ❌ BAD: 'Make it more unique'",
            "- ✅ GOOD: 'Instead of \"Caching Basics\", try \"When NOT to Use Caching\" (addresses the tradeoffs gap)'",
            "",
            "Also provide:",
            "- **Summary**: High-level overview of all issues (2-3 sentences)",
            "- **General Guidance**: Overall advice for replanning (focus on what to avoid, what to prioritize)",
            "",
            "Return structured output with:",
            "- summary: string",
            "- section_suggestions: list of {section_id, issue, suggested_title, suggested_angle, alternative_gap}",
            "- general_guidance: string",
        ]
    )

    return "\n".join(prompt_parts)


async def _generate_replanning_feedback_llm(
    plan: dict[str, Any],
    content_strategy: dict[str, Any],
    key_manager: KeyManager,
    gap_validation: dict[str, Any] | None = None,
    weak_sections: list[SectionFeasibilityScore] | None = None,
    non_unique_sections: list[UniquenessCheck] | None = None,
    max_retries: int = 3,
) -> ReplanningFeedback:
    """
    Generate concrete replanning feedback using LLM.

    Uses Gemini Flash to analyze validation failures and provide specific,
    actionable suggestions for improving each rejected section.

    Args:
        plan: Original blog plan
        content_strategy: Content strategy with gaps
        key_manager: API key manager
        gap_validation: Gap mapping validation results
        weak_sections: Sections with low feasibility
        non_unique_sections: Sections with high overlap
        max_retries: Maximum retry attempts

    Returns:
        ReplanningFeedback object with concrete suggestions

    Raises:
        RuntimeError: If all retries fail
    """
    prompt = _build_replanning_feedback_prompt(
        plan, content_strategy, gap_validation, weak_sections, non_unique_sections
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            api_key = key_manager.get_best_key()

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.3,  # Low temp for consistent, structured feedback
            )

            structured_llm = llm.with_structured_output(ReplanningFeedback)

            logger.info(
                f"Generating replanning feedback via LLM (attempt {attempt + 1}/{max_retries})..."
            )
            result = await structured_llm.ainvoke(prompt)

            # Track usage
            key_manager.record_usage(api_key, tokens_in=len(prompt) // 4, tokens_out=500)

            logger.info(f"✓ Generated feedback with {len(result.section_suggestions)} suggestions")
            return result

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Handle rate limiting
            if "429" in error_msg or "resource_exhausted" in error_msg:
                logger.warning(f"Rate limit hit, rotating key (attempt {attempt + 1})")
                key_manager.mark_rate_limited(api_key)

            logger.error(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(
        f"Failed to generate replanning feedback after {max_retries} attempts: {last_error}"
    )


def _build_replanning_feedback(
    gap_validation: dict[str, Any] | None = None,
    weak_sections: list[SectionFeasibilityScore] | None = None,
    non_unique_sections: list[UniquenessCheck] | None = None,
) -> str:
    """
    Build feedback message for replanning.

    Args:
        gap_validation: Gap mapping validation results
        weak_sections: Sections with low feasibility scores
        non_unique_sections: Sections with high overlap

    Returns:
        Formatted feedback string
    """
    feedback_parts = []

    # Gap mapping issues
    if gap_validation and not gap_validation.get("all_gaps_covered"):
        missing_gaps = gap_validation.get("missing_gaps", [])
        sections_without_mapping = gap_validation.get("sections_without_mapping", [])

        if missing_gaps:
            feedback_parts.append(
                f"**Missing Content Gaps**: The following gaps are not addressed by any section: {', '.join(missing_gaps)}"
            )

        if sections_without_mapping:
            feedback_parts.append(
                f"**Sections Without Gap Mapping**: The following sections don't map to any gap: {', '.join(sections_without_mapping)}"
            )

    # Weak sections (low information availability)
    if weak_sections:
        feedback_parts.append("**Weak Sections (Low Information Availability)**:")
        for score in weak_sections:
            concerns_str = "; ".join(score.concerns)
            feedback_parts.append(
                f"  - {score.section_id}: Score {score.information_availability}/100. {concerns_str}"
            )

    # Non-unique sections
    if non_unique_sections:
        feedback_parts.append("**Non-Unique Sections (High Overlap with Existing Articles)**:")
        for check in non_unique_sections:
            feedback_parts.append(
                f"  - {check.section_id}: {check.overlap_percentage:.1f}% overlap. {check.concerns}"
            )

    return "\n\n".join(feedback_parts)


async def _trigger_replanning(
    reason: str,
    iteration: int,
    plan: dict[str, Any],
    content_strategy: dict[str, Any],
    key_manager: KeyManager,
    gap_validation: dict[str, Any] | None = None,
    weak_sections: list[SectionFeasibilityScore] | None = None,
    non_unique_sections: list[UniquenessCheck] | None = None,
    scratchpad: list[dict[str, Any]] | None = None,
    job_id: str = "",
) -> dict[str, Any]:
    """
    Trigger replanning by returning to PLANNING phase with LLM-generated feedback.

    Args:
        reason: Reason for replanning
        iteration: Current planning iteration
        plan: Original blog plan
        content_strategy: Content strategy with gaps
        key_manager: API key manager
        gap_validation: Gap validation results
        weak_sections: Weak sections
        non_unique_sections: Non-unique sections
        scratchpad: Planning iteration scratchpad (will be updated with feedback)
        job_id: Job ID for disk persistence

    Returns:
        State update dict to trigger replanning
    """
    if scratchpad is None:
        scratchpad = []
    # Generate concrete feedback using LLM
    try:
        feedback_obj = await _generate_replanning_feedback_llm(
            plan, content_strategy, key_manager, gap_validation, weak_sections, non_unique_sections
        )

        # Format feedback as string for state
        feedback_parts = [
            f"## {feedback_obj.summary}",
            "",
            "## Specific Suggestions",
            "",
        ]

        for suggestion in feedback_obj.section_suggestions:
            feedback_parts.extend(
                [
                    f"### Section: {suggestion.section_id}",
                    f"**Issue**: {suggestion.issue}",
                    f"**Suggested Title**: {suggestion.suggested_title}",
                    f"**Suggested Angle**: {suggestion.suggested_angle}",
                ]
            )
            if suggestion.alternative_gap:
                feedback_parts.append(f"**Alternative Gap**: {suggestion.alternative_gap}")
            feedback_parts.append("")

        feedback_parts.extend(
            [
                "## General Guidance",
                feedback_obj.general_guidance,
            ]
        )

        feedback = "\n".join(feedback_parts)

        # Update scratchpad entry with LLM feedback
        if scratchpad:
            scratchpad[-1]["llm_feedback"] = {
                "summary": feedback_obj.summary,
                "section_suggestions": [
                    {
                        "section_id": s.section_id,
                        "issue": s.issue,
                        "suggested_title": s.suggested_title,
                        "suggested_angle": s.suggested_angle,
                    }
                    for s in feedback_obj.section_suggestions
                ],
                "general_guidance": feedback_obj.general_guidance,
            }
            scratchpad[-1]["feedback_text"] = feedback

    except Exception as e:
        logger.error(f"LLM feedback generation failed: {e}. Falling back to static feedback.")
        # Fallback to static method if LLM fails
        feedback = _build_replanning_feedback(gap_validation, weak_sections, non_unique_sections)

        # Update scratchpad with static feedback
        if scratchpad:
            scratchpad[-1]["feedback_text"] = feedback
            scratchpad[-1]["llm_feedback"] = None

    # Save scratchpad to disk
    _save_planning_scratchpad(job_id, scratchpad)

    rejected_section_ids = []
    if weak_sections:
        rejected_section_ids.extend([s.section_id for s in weak_sections])
    if non_unique_sections:
        rejected_section_ids.extend([s.section_id for s in non_unique_sections])

    return {
        "current_phase": Phase.PLANNING.value,
        "planning_iteration": iteration + 1,
        "replanning_feedback": feedback,
        "rejected_sections": [{"id": sid, "reason": reason} for sid in rejected_section_ids],
        "preview_validation_scratchpad": scratchpad,
    }


def _save_planning_scratchpad(job_id: str, scratchpad: list[dict[str, Any]]) -> None:
    """Save planning scratchpad to disk."""
    if not job_id:
        return

    from pathlib import Path
    import json

    job_dir = Path.home() / ".blog_agent" / "jobs" / job_id
    scratchpad_dir = job_dir / "planning_scratchpad"
    scratchpad_dir.mkdir(parents=True, exist_ok=True)

    scratchpad_file = scratchpad_dir / "iterations.json"
    with open(scratchpad_file, "w") as f:
        json.dump(scratchpad, f, indent=2, default=str)


def _load_planning_scratchpad(job_id: str) -> list[dict[str, Any]]:
    """Load planning scratchpad from disk."""
    if not job_id:
        return []

    from pathlib import Path
    import json

    scratchpad_file = (
        Path.home() / ".blog_agent" / "jobs" / job_id / "planning_scratchpad" / "iterations.json"
    )

    if not scratchpad_file.exists():
        return []

    with open(scratchpad_file) as f:
        return json.load(f)


def _build_planning_scratchpad_entry(
    iteration: int,
    plan: dict[str, Any],
    gap_validation: dict[str, Any] | None,
    feasibility_scores: list[SectionFeasibilityScore],
    uniqueness_result: Any,  # UniquenessAnalysisResult
    passed: bool,
    rejected_section_ids: list[str],
    llm_feedback: Any | None = None,  # ReplanningFeedback
) -> dict[str, Any]:
    """Build a single scratchpad entry for planning iteration."""
    from datetime import datetime

    entry = {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat(),
        # Plan snapshot
        "plan_snapshot": {
            "topic": plan.get("topic", ""),
            "sections": [
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "gap_addressed": s.get("gap_addressed"),
                }
                for s in plan.get("sections", [])
            ],
        },
        # Validation results
        "gap_validation": gap_validation,
        "feasibility_scores": [
            {
                "section_id": fs.section_id,
                "score": fs.information_availability,
                "is_feasible": fs.is_feasible,
                "concerns": fs.concerns,
            }
            for fs in feasibility_scores
        ],
        "uniqueness_checks": [
            {
                "section_id": uc.section_id,
                "section_title": uc.section_title,
                "overlap_percentage": uc.overlap_percentage,
                "is_unique": uc.is_unique,
                "concerns": uc.concerns,
            }
            for uc in uniqueness_result.uniqueness_checks
        ],
        # Outcome
        "passed": passed,
        "rejected_section_ids": rejected_section_ids,
        # Feedback (will be added later if fails)
        "llm_feedback": None,
        "feedback_text": "",
    }

    if llm_feedback:
        entry["llm_feedback"] = {
            "summary": llm_feedback.summary,
            "section_suggestions": [
                {
                    "section_id": s.section_id,
                    "issue": s.issue,
                    "suggested_title": s.suggested_title,
                    "suggested_angle": s.suggested_angle,
                }
                for s in llm_feedback.section_suggestions
            ],
            "general_guidance": llm_feedback.general_guidance,
        }

    return entry


async def preview_validation_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 1.5: Preview Validation Node.

    Validates planned sections before research:
    1. Gap mapping completeness
    2. Information availability (test sample queries)
    3. Uniqueness vs analyzed articles
    4. Decision: proceed OR trigger replanning (max 3 iterations)

    Args:
        state: Current blog agent state

    Returns:
        State updates
    """
    node_name = "preview_validation"
    metrics, start_time = _init_node_metrics(state, node_name)

    logger.info("Starting preview validation")

    # Skip if already past this phase
    if phase_is_past(state.get("current_phase", ""), Phase.PREVIEW_VALIDATION):
        logger.info("Skipping preview validation (already past this phase)")
        return {"metrics": _finalize_node_metrics(metrics, node_name, start_time)}

    # Get state
    plan = state.get("plan")
    content_strategy = state.get("content_strategy")
    iteration = state.get("planning_iteration", 0)
    key_manager = state.get("key_manager")
    job_id = state.get("job_id", "")

    # Initialize scratchpad
    scratchpad = state.get("preview_validation_scratchpad", [])
    if not scratchpad:
        scratchpad = []

    if not plan:
        logger.error("No plan found in state")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "No plan found for preview validation",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    if not content_strategy:
        logger.error("No content strategy found - landscape analysis required")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Content strategy required. Landscape analysis must run first.",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    if not key_manager:
        logger.error("No key manager found in state")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "Key manager required for LLM validation",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    sections = plan.get("sections", [])
    logger.info(f"Validating {len(sections)} sections (iteration {iteration + 1}/3)")

    try:
        # Step 1: Validate gap mapping
        logger.info("Validating gap mapping...")
        gap_validation = _validate_gap_mapping(plan, content_strategy)

        if not gap_validation["all_gaps_covered"]:
            logger.warning(
                f"Gap mapping incomplete: {gap_validation['missing_gaps']} gaps not addressed"
            )
            return await _trigger_replanning(
                reason="gap_mapping_incomplete",
                iteration=iteration,
                plan=plan,
                content_strategy=content_strategy,
                key_manager=key_manager,
                gap_validation=gap_validation,
            )

        logger.info("✓ Gap mapping complete")

        # Step 2: Test information availability (parallel)
        logger.info("Testing information availability for sections...")
        feasibility_scores = await _test_section_feasibility_parallel(sections)

        # Step 3: Check uniqueness using LLM
        logger.info("Checking section uniqueness via LLM...")
        uniqueness_result = await _check_sections_uniqueness_llm(
            sections, content_strategy, key_manager
        )
        uniqueness_checks = uniqueness_result.uniqueness_checks

        # Step 4: Decision
        weak_sections = [s for s in feasibility_scores if not s.is_feasible]
        non_unique = [c for c in uniqueness_checks if not c.is_unique]

        logger.info(f"Weak sections: {len(weak_sections)}, Non-unique: {len(non_unique)}")

        # All sections pass
        if len(weak_sections) == 0 and len(non_unique) == 0:
            logger.info("✓ All sections passed preview validation")

            # Build scratchpad entry
            scratchpad_entry = _build_planning_scratchpad_entry(
                iteration=iteration,
                plan=plan,
                gap_validation=gap_validation,
                feasibility_scores=feasibility_scores,
                uniqueness_result=uniqueness_result,
                passed=True,
                rejected_section_ids=[],
                llm_feedback=None,
            )
            scratchpad.append(scratchpad_entry)

            # Save scratchpad to disk
            _save_planning_scratchpad(job_id, scratchpad)

            # Convert to dicts for state
            validation_result = PreviewValidationResult(
                section_scores=feasibility_scores,
                weak_sections=[],
                all_sections_pass=True,
                recommendation="proceed",
                feedback_for_replanning="",
            )

            return {
                "current_phase": Phase.SECTION_SELECTION.value,
                "preview_validation_result": validation_result.model_dump(),
                "uniqueness_checks": [c.model_dump() for c in uniqueness_checks],
                "preview_validation_scratchpad": scratchpad,
                "metrics": _finalize_node_metrics(metrics, node_name, start_time),
            }

        # Some sections failed - check iteration limit
        if iteration >= 2:
            logger.error(f"Max iterations (3) reached. Cannot create viable plan.")

            # Build final scratchpad entry
            rejected_section_ids = [s.section_id for s in weak_sections] + [c.section_id for c in non_unique]
            scratchpad_entry = _build_planning_scratchpad_entry(
                iteration=iteration,
                plan=plan,
                gap_validation=gap_validation,
                feasibility_scores=feasibility_scores,
                uniqueness_result=uniqueness_result,
                passed=False,
                rejected_section_ids=rejected_section_ids,
                llm_feedback=None,
            )
            scratchpad_entry["failure_reason"] = "max_iterations_reached"
            scratchpad.append(scratchpad_entry)

            # Save scratchpad to disk
            _save_planning_scratchpad(job_id, scratchpad)

            return {
                "current_phase": Phase.FAILED.value,
                "error_message": "Cannot create viable plan after 3 attempts. Please review and manually edit plan.json.",
                "preview_validation_result": {
                    "section_scores": [s.model_dump() for s in feasibility_scores],
                    "weak_sections": [s.section_id for s in weak_sections],
                    "all_sections_pass": False,
                    "recommendation": "manual_intervention",
                    "feedback_for_replanning": _build_replanning_feedback(
                        None, weak_sections, non_unique
                    ),
                },
                "preview_validation_scratchpad": scratchpad,
                "metrics": _finalize_node_metrics(metrics, node_name, start_time),
            }

        # Trigger replanning
        logger.warning(f"Preview validation failed. Triggering replanning (attempt {iteration + 2}/3)")

        # Build scratchpad entry (feedback will be added by _trigger_replanning)
        rejected_section_ids = [s.section_id for s in weak_sections] + [c.section_id for c in non_unique]
        scratchpad_entry = _build_planning_scratchpad_entry(
            iteration=iteration,
            plan=plan,
            gap_validation=gap_validation,
            feasibility_scores=feasibility_scores,
            uniqueness_result=uniqueness_result,
            passed=False,
            rejected_section_ids=rejected_section_ids,
            llm_feedback=None,  # Will be added by _trigger_replanning
        )
        scratchpad.append(scratchpad_entry)

        return await _trigger_replanning(
            reason="sections_failed_validation",
            iteration=iteration,
            plan=plan,
            content_strategy=content_strategy,
            key_manager=key_manager,
            weak_sections=weak_sections,
            non_unique_sections=non_unique,
            gap_validation=gap_validation,
            scratchpad=scratchpad,
            job_id=job_id,
        )

    except Exception as e:
        logger.error(f"Preview validation failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Preview validation error: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


# =============================================================================
# Section Selection Node (Phase 1.6)
# =============================================================================


def _parse_selection_input(selection: str, max_index: int) -> list[int]:
    """
    Parse user selection input into list of indices.

    Supports:
    - "1,3,5" → [1, 3, 5]
    - "1-4" → [1, 2, 3, 4]
    - "1,3-5,7" → [1, 3, 4, 5, 7]

    Args:
        selection: User input string
        max_index: Maximum valid index

    Returns:
        List of selected indices (1-based)

    Raises:
        ValueError: If input is invalid
    """
    indices = set()

    for part in selection.split(","):
        part = part.strip()

        if "-" in part:
            # Range: "1-4"
            try:
                start, end = part.split("-")
                start, end = int(start.strip()), int(end.strip())
                if start < 1 or end > max_index or start > end:
                    raise ValueError(f"Invalid range: {part}")
                indices.update(range(start, end + 1))
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Single number
            try:
                num = int(part)
                if num < 1 or num > max_index:
                    raise ValueError(f"Number out of range: {num}")
                indices.add(num)
            except ValueError:
                raise ValueError(f"Invalid number: {part}")

    return sorted(list(indices))


async def section_selection_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Present optional blog sections to user for selection.

    User selects which optional sections to include in the final blog.
    Required sections are always included automatically.

    Args:
        state: BlogAgentState with completed plan

    Returns:
        State update with selected_section_ids
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, Confirm

    start_time = time.time()
    node_name = "section_selection"
    metrics = state.get("metrics", [])

    console = Console()
    plan = state.get("plan", {})
    sections = plan.get("sections", [])

    # Separate required and optional
    required = [s for s in sections if not s.get("optional", False)]
    optional = [s for s in sections if s.get("optional", False)]

    # Edge case: no optional sections
    if not optional:
        console.print("[yellow]No optional sections to select. Proceeding with all required sections.[/yellow]")
        return {
            "selected_section_ids": [s["id"] for s in required],
            "section_selection_skipped": True,
            "current_phase": Phase.RESEARCHING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    # Display header
    console.print("\n[bold blue]📋 Section Selection[/bold blue]")
    console.print(f"Total sections generated: {len(sections)}")
    console.print(f"Required sections: {len(required)} (always included)")
    console.print(f"Optional sections: {len(optional)} (select 2-4)\n")

    # Show required sections (auto-included)
    console.print("[bold green]✓ Required Sections (always included):[/bold green]")
    for i, sec in enumerate(required, 1):
        console.print(f"  {i}. [bold]{sec.get('title', sec['id'])}[/bold] ({sec['role']}, ~{sec['target_words']}w)")

    # Show optional sections in table
    console.print(f"\n[bold yellow]Optional Sections (select 2-4):[/bold yellow]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Title", style="white", width=30)
    table.add_column("Role", style="green", width=15)
    table.add_column("Gap Addressed", style="yellow", width=18)
    table.add_column("Words", style="magenta", width=6)
    table.add_column("Code", style="blue", width=4)
    table.add_column("Why Include?", style="white", width=50)

    for i, sec in enumerate(optional, 1):
        justification = sec.get("gap_justification", "No justification provided")
        truncated = justification[:47] + "..." if len(justification) > 50 else justification
        table.add_row(
            str(i),
            sec.get("title", sec["id"])[:30],
            sec["role"],
            sec.get("gap_addressed", "N/A"),
            str(sec["target_words"]),
            "✓" if sec.get("needs_code") else "-",
            truncated
        )

    console.print(table)

    # Get user input
    console.print("\n[bold]Selection Options:[/bold]")
    console.print("  • Enter numbers (e.g., '1,3,5' or '1-4' or '1,3-5')")
    console.print("  • Press Enter to skip selection (include all sections)")
    console.print("  • Type 'all' to include all optional sections")

    while True:
        selection = Prompt.ask(
            "\nSelect optional sections (2-4 recommended)",
            default=""
        )

        # Handle special cases
        if not selection or selection.lower() == "skip":
            if Confirm.ask("Skip selection and include all sections?", default=False):
                selected_ids = [s["id"] for s in sections]
                skipped = True
                break

        if selection.lower() == "all":
            selected_ids = [s["id"] for s in sections]
            skipped = True
            console.print("[green]All sections selected.[/green]")
            break

        # Parse selection
        try:
            selected_indices = _parse_selection_input(selection, len(optional))

            # Validate count (recommend 2-4 but allow any)
            if len(selected_indices) < 1:
                console.print("[red]Please select at least 1 optional section.[/red]")
                continue

            if len(selected_indices) > len(optional):
                console.print(f"[red]Cannot select more than {len(optional)} sections.[/red]")
                continue

            # Warning for too few/many
            if len(selected_indices) < 2:
                if not Confirm.ask(f"Only {len(selected_indices)} section selected. Continue?", default=False):
                    continue
            elif len(selected_indices) > 4:
                if not Confirm.ask(f"{len(selected_indices)} sections selected (blog will be long). Continue?", default=False):
                    continue

            # Build selected IDs: required + selected optional
            selected_optional = [optional[i-1] for i in selected_indices]
            selected_ids = [s["id"] for s in required] + [s["id"] for s in selected_optional]
            skipped = False

            # Confirmation
            console.print(f"\n[bold]Final selection ({len(selected_ids)} sections):[/bold]")
            for s in required + selected_optional:
                console.print(f"  ✓ {s.get('title', s['id'])}")

            if Confirm.ask("\nProceed with this selection?", default=True):
                break
            else:
                console.print("[yellow]Selection cancelled. Please try again.[/yellow]\n")

        except ValueError as e:
            console.print(f"[red]Invalid input: {e}[/red]")
            continue

    console.print(f"\n[green]✓ Selection complete. Proceeding with {len(selected_ids)} sections.[/green]")

    # Save to checkpoint
    job_manager = JobManager()
    job_id = state.get("job_id", "")
    if job_id:
        selection_file = job_manager.get_job_dir(job_id) / "selected_sections.json"
        with open(selection_file, "w") as f:
            json.dump({
                "selected_section_ids": selected_ids,
                "selection_skipped": skipped,
                "timestamp": datetime.utcnow().isoformat(),
            }, f, indent=2)

    return {
        "selected_section_ids": selected_ids,
        "section_selection_skipped": skipped,
        "current_phase": Phase.RESEARCHING.value,
        "metrics": _finalize_node_metrics(metrics, node_name, start_time),
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
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.RESEARCHING):
        logger.info("Skipping research_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "research"
    metrics, start_time = _init_node_metrics(state, node_name)
    logger.info("Starting research phase")

    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    job_id = state.get("job_id", "")
    existing_cache = state.get("research_cache", {})

    if not sections:
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "No sections found in plan",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
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
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Research failed: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
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
    retry_attempt: int = 0,
    failed_sources: list[dict] | None = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Generate alternative search queries with diversification strategy.

    Args:
        blog_title: Title of the blog
        section: Section dict with id, title, role
        original_queries: Queries that didn't yield enough sources
        key_manager: KeyManager for API key rotation
        retry_attempt: Which retry this is (0-indexed), used for diversification
        failed_sources: Sources that were fetched but failed validation
        max_retries: Max LLM retry attempts

    Returns:
        List of 2-3 alternative search queries
    """
    section_title = section.get("title") or section.get("id", "")
    section_role = section.get("role", "")

    # Select modifier based on retry attempt for diversification
    modifiers = QUERY_DIVERSIFICATION_MODIFIERS[
        retry_attempt % len(QUERY_DIVERSIFICATION_MODIFIERS)
    ]
    modifier_hint = f"Focus on finding: {', '.join(modifiers)}"

    # Build context about failed sources to avoid similar domains
    failed_context = ""
    if failed_sources:
        failed_urls = [s.get("url", "") for s in failed_sources[:5]]
        failed_context = f"""

These sources were found but rejected (avoid similar sites):
{failed_urls}"""

    prompt = f"""The following search queries didn't yield enough quality sources:
{original_queries}

For this blog section:
Title: "{section_title}"
Role: {section_role}
Blog: "{blog_title}"

{modifier_hint}
{failed_context}

Generate 2-3 COMPLETELY DIFFERENT search queries that:
1. Use different keywords and phrasing than the original queries
2. Target different types of sources ({', '.join(modifiers)})
3. Avoid the same domains as rejected sources

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

            # Handle rate limiting
            if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                logger.warning("Rate limited on key, rotating...")
                key_manager.mark_rate_limited(api_key)

                next_key = key_manager.get_next_key(api_key)
                if next_key is None:
                    raise RuntimeError("All API keys exhausted or rate-limited")
                continue

            # Retry on parsing/validation errors (LLM returned incomplete output)
            if "field required" in error_str or "validation error" in error_str:
                logger.warning(
                    f"Source validation parsing failed (attempt {attempt + 1}/{max_retries}), retrying..."
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue

            # For other errors, log and retry with backoff
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
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.VALIDATING_SOURCES):
        logger.info("Skipping validate_sources_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "validate_sources"
    metrics, start_time = _init_node_metrics(state, node_name)
    llm_call_count = 0  # Track number of validation calls for metrics
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
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
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
            llm_call_count += 1

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

                # Identify sources that were fetched but failed validation
                validated_urls = {s["url"] for s in validated}
                failed_sources = [
                    s for s in section_sources
                    if s.get("url") not in validated_urls
                ]

                # Generate alternative queries with diversification
                alt_queries = await _generate_alternative_queries(
                    blog_title=blog_title,
                    section=section,
                    original_queries=list(used_queries),
                    key_manager=key_manager,
                    retry_attempt=retry_count - 1,  # 0-indexed for modifier selection
                    failed_sources=failed_sources,
                )
                llm_call_count += 1
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
                    llm_call_count += 1
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

        # Record LLM metrics (estimate: ~2500 tokens in, 300 out per validation call)
        if llm_call_count > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                llm_call_count * 2500, llm_call_count * 300
            )

        return {
            "validated_sources": validated_sources,
            "research_cache": research_cache,  # Include updated cache from retries
            "current_phase": Phase.WRITING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except RuntimeError as e:
        logger.error(f"Source validation failed: {e}")
        # Record any LLM calls that happened before failure
        if llm_call_count > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                llm_call_count * 2500, llm_call_count * 300
            )
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    except Exception as e:
        logger.error(f"Unexpected error in source validation: {e}")
        # Record any LLM calls that happened before failure
        if llm_call_count > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                llm_call_count * 2500, llm_call_count * 300
            )
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


# =============================================================================
# Write Section Node (Phase 3)
# =============================================================================


def _format_sources_for_prompt(sources: list[dict], max_sources: int = 5) -> str:
    """
    Format validated sources for inclusion in writer prompt.

    Args:
        sources: List of source dicts with url, title, content
        max_sources: Maximum number of sources to include

    Returns:
        Formatted string with source content for the prompt
    """
    if not sources:
        return "No research sources available. Write based on your knowledge."

    formatted = []
    for i, source in enumerate(sources[:max_sources], 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        content = source.get("content", "")[:2000]  # Limit content length

        formatted.append(f"""
Source {i}: [{title}]({url})
{content}
""")

    return "\n---\n".join(formatted)


def _get_previous_sections_text(
    section_drafts: dict[str, str],
    sections: list[dict],
) -> str:
    """
    Get text from previously written sections for context.

    Args:
        section_drafts: Dict of section_id -> markdown content
        sections: List of section dicts (in order)

    Returns:
        Concatenated text from previous sections
    """
    if not section_drafts or not sections:
        return ""

    previous_text = []
    for section in sections:
        section_id = section.get("id", "")
        if section_id in section_drafts:
            title = section.get("title") or section.get("role", "Section")
            content = section_drafts[section_id]
            previous_text.append(f"## {title}\n\n{content}")

    return "\n\n".join(previous_text)


def _build_writer_prompt(
    section: dict,
    sources: list[dict],
    previous_sections_text: str,
    style_guide: str,
    blog_title: str,
) -> str:
    """
    Build the writer prompt for a section.

    Args:
        section: Section dict with id, title, role, target_words, needs_code, needs_diagram
        sources: List of validated sources for this section
        previous_sections_text: Text from previously written sections
        style_guide: Writing style guidelines
        blog_title: Title of the blog post

    Returns:
        Complete prompt string for section writing
    """
    section_title = section.get("title") or section.get("id", "Section")
    section_role = section.get("role", "")
    target_words = section.get("target_words", 200)
    needs_code = section.get("needs_code", False)
    needs_diagram = section.get("needs_diagram", False)

    formatted_sources = _format_sources_for_prompt(sources)

    # Build role-specific instructions with sentence structure requirements
    role_instructions = ""

    # Sentence structure requirements (common to all roles)
    sentence_requirements = """
SENTENCE STRUCTURE (CRITICAL):
- 10-18 words per sentence (target range)
- Average 12-16 words for this section
- One complete thought per sentence
- No semicolons (use natural connectors like 'because', 'when', 'which' instead)
- Avoid deeply nested clauses
- Active voice preferred
- Use fragments when natural ("Still, important.")
- Combine related ideas with conjunctions rather than breaking into choppy fragments
"""

    if section_role == "hook":
        role_instructions = f"""
This is the HOOK section. Grab attention immediately with punchy, direct language.

{sentence_requirements}

CONTENT GUIDELINES:
- Start with a surprising stat, provocative question, or relatable problem
- Do NOT use a title/heading for this section
- 2-3 paragraphs max
- Short, punchy sentences that hook the reader

{get_example_for_role("hook")}
"""
    elif section_role == "problem":
        role_instructions = f"""
This is the PROBLEM section. Explain what's broken. Use bullets. Be direct.

{sentence_requirements}

CONTENT GUIDELINES:
- State the problem directly, no hype
- Use specific, relatable examples
- Show WHY it fails (e.g., exact match vs semantic)
- Bullet points work well for listing issues
- Keep each point short and punchy

{get_example_for_role("problem")}
"""
    elif section_role == "why":
        role_instructions = f"""
This is the WHY section. Explain why the new approach matters. Stay punchy.

{sentence_requirements}

CONTENT GUIDELINES:
- Connect solution to problems described earlier
- Highlight key benefits in short sentences
- Set up the reader for implementation
- No fluff or marketing speak

{get_example_for_role("why")}
"""
    elif section_role in ["implementation", "deep_dive"]:
        role_instructions = f"""
This is an IMPLEMENTATION section. Code is 50-70% of this section. Let code speak.

{sentence_requirements}

CONTENT GUIDELINES:
- 50-70% of this section should be code examples
- Explain in 1-2 sentences max before/after each code block
- Show complete, runnable code with imports (not snippets)
- Each code block demonstrates ONE specific technique
- No long narratives - minimal commentary
- Code is the PRIMARY teaching tool

{get_example_for_role("implementation")}
"""
    elif section_role == "tradeoffs":
        role_instructions = f"""
This is the DRAWBACKS section. Be honest. Be direct. Use bullets.

{sentence_requirements}

CONTENT GUIDELINES:
- List specific drawbacks and edge cases
- Mention when this approach is NOT suitable
- Quantify overhead if possible (latency, cost, complexity)
- Short bullet points work best
- No sugar-coating

{get_example_for_role("tradeoffs")}
"""
    elif section_role == "conclusion":
        role_instructions = f"""
This is the CONCLUSION section. Quick summary. Clear next steps. Stay punchy.

{sentence_requirements}

CONTENT GUIDELINES:
- Summarize key points (don't repeat)
- Give clear next steps
- 1 paragraph max
- Action-oriented, brief

{get_example_for_role("conclusion")}
"""

    # Code/diagram requirements
    requirements = []
    if needs_code:
        requirements.append("- MUST include working code examples with imports")
    if needs_diagram:
        requirements.append("- MUST include a mermaid diagram (```mermaid\\n...\\n```)")
    requirements_text = "\n".join(requirements) if requirements else "- No special requirements"

    prompt = f"""You are writing a section for a technical blog post.

## Blog Title
"{blog_title}"

## Section to Write
Title: "{section_title}"
Role: {section_role}
Target word count: ~{target_words} words (±20%)

{role_instructions}

## Writing Style Guide
{style_guide}

## Requirements
{requirements_text}

## Research Sources
Use these sources for accurate, up-to-date information:
{formatted_sources}

## Previously Written Sections
{previous_sections_text if previous_sections_text else "(This is the first section)"}

## Instructions
Write the "{section_title}" section now. Output ONLY the section content in markdown format.
Do not include the section title as a heading (it will be added automatically).
Focus on delivering value to the reader with specific, actionable content."""

    return prompt


async def _write_section(
    section: dict,
    sources: list[dict],
    previous_sections_text: str,
    blog_title: str,
    key_manager: KeyManager,
    max_retries: int = 3,
) -> str:
    """
    Write a single section using Gemini Flash.

    Args:
        section: Section dict with id, title, role, etc.
        sources: List of validated sources for this section
        previous_sections_text: Text from previously written sections
        blog_title: Title of the blog post
        key_manager: KeyManager for API key rotation
        max_retries: Maximum retry attempts

    Returns:
        Generated markdown content for the section

    Raises:
        RuntimeError: If all API keys exhausted or max retries exceeded
    """
    from .config import LLM_MODEL_FULL, LLM_TEMPERATURE_MEDIUM, STYLE_GUIDE

    prompt = _build_writer_prompt(
        section=section,
        sources=sources,
        previous_sections_text=previous_sections_text,
        style_guide=STYLE_GUIDE,
        blog_title=blog_title,
    )

    last_error = None

    for attempt in range(max_retries):
        api_key = key_manager.get_best_key()

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_FULL,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_MEDIUM,
            )

            # Invoke (run in thread pool since langchain may be sync internally)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: llm.invoke(prompt)
            )

            # Extract content from AIMessage
            content = result.content if hasattr(result, "content") else str(result)

            # Record usage
            key_manager.record_usage(
                api_key,
                tokens_in=len(prompt) // 4,
                tokens_out=len(content) // 4,
            )

            section_id = section.get("id", "unknown")
            logger.info(f"Successfully wrote section '{section_id}' ({len(content.split())} words)")

            return content

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

            logger.error(f"Section writing attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    raise RuntimeError(f"Failed to write section after {max_retries} attempts: {last_error}")


def _build_critic_prompt(
    section: dict,
    content: str,
    target_words: int,
    blog_title: str = "",
    context: str = "",
    scratchpad: list[dict] | None = None,
    style_guide: str = STYLE_GUIDE,
) -> str:
    """
    Build critic prompt for 8-dimension section evaluation.

    Args:
        section: Section metadata (title, role, needs_code, needs_diagram)
        content: Written section markdown
        target_words: Target word count for length evaluation
        blog_title: Overall blog title for context
        context: User-provided context about the blog topic
        scratchpad: Refinement history (ReAct-style) showing previous attempts
        style_guide: Writing style guidelines for voice evaluation

    Returns:
        Critic evaluation prompt
    """
    section_title = section.get("title") or section.get("id", "")
    section_role = section.get("role", "")
    needs_code = section.get("needs_code", False)
    needs_diagram = section.get("needs_diagram", False)

    # Count actual words
    actual_words = len(content.split())

    # Build context section
    context_info = ""
    if blog_title:
        context_info += f"\n- Blog title: {blog_title}"
    if context:
        context_info += f"\n- Blog context: {context}"

    # Build refinement history section
    history_text = ""
    if scratchpad:
        history_text = "\n**Refinement History:**\n"
        for entry in scratchpad:
            attempt = entry['attempt']
            score = entry['score']
            addressed = entry.get('addressed_issues', [])
            new = entry.get('new_issues', [])

            if attempt == 0:
                history_text += f"\nAttempt {attempt} (Initial write): Score {score:.1f}/10\n"
                history_text += f"  Issues identified: {', '.join(entry['issues']) if entry['issues'] else 'None'}\n"
            else:
                score_changes = entry.get('score_changes', {})
                changes_str = ', '.join([f"{k}: {v}" for k, v in score_changes.items() if v != "0"])

                history_text += f"\nAttempt {attempt} (Refinement): Score {score:.1f}/10\n"
                history_text += f"  Addressed: {', '.join(addressed) if addressed else 'None'}\n"
                history_text += f"  New issues: {', '.join(new) if new else 'None'}\n"
                if changes_str:
                    history_text += f"  Score changes: {changes_str}\n"
        history_text += "\n"

    # Analyze sentence structure metrics
    metrics = _analyze_sentence_lengths(content)

    prompt = f"""You are an expert technical blog critic. Evaluate this section on 8 dimensions (1-10 scale).

**Overall Blog Context:**{context_info}

**Section Metadata:**
- Title: {section_title}
- Role: {section_role}
- Target words: {target_words}
- Needs code: {needs_code}
- Needs diagram: {needs_diagram}

**Section Content:**
```markdown
{content}
```

**Actual word count:** {actual_words}

**Sentence Structure Metrics (for voice evaluation):**
- Average sentence length: {metrics['avg_length']} words
- Longest sentence: {metrics['max_length']} words
- Sentences > 20 words: {metrics['long_sentences']}
- Sentences > 25 words: {metrics['very_long_sentences']}
- Semicolons: {metrics['semicolons']}
- Simple sentences (≤18 words): {metrics['percent_simple']:.1f}%
- Total sentences: {metrics['total_sentences']}

**Writing Style Guidelines:**
{style_guide}
{history_text}
**Evaluation Criteria:**

1. **technical_accuracy (1-10):** Are technical claims correct? Any misinformation?
2. **completeness (1-10):** Does it cover all necessary aspects? Any gaps?
3. **code_quality (1-10):** If code present: Is it runnable, includes imports, follows best practices? (10 if no code needed)
4. **clarity (1-10):** Is it easy to understand? Clear explanations?
5. **voice (1-10):** PUNCHY STYLE EVALUATION with HARD METRICS:
   - Start at 10, then apply deductions:
   - If avg_length > 18 words: Deduct 2 points
   - If avg_length > 22 words: Deduct 3 points (total)
   - If any sentence > 30 words: Deduct 1 point per sentence
   - If semicolons > 0: Deduct 1 point
   - If simple sentences (≤18 words) < 65%: Deduct 1 point
   - If contains excessive transition words ("therefore", "however", "moreover", "furthermore" used 3+ times): Deduct 1 point
   - If uses "leverage", "utilize", "facilitate" instead of simple verbs: Deduct 1 point

   **Target metrics for score 10:**
   - Avg: 12-16 words
   - No sentences > 22 words
   - No semicolons
   - Simple sentences (≤18 words) > 70%
   - Complete thoughts, not fragmented ideas

   **Score 7-9:** Avg 14-18 words, 1-2 sentences up to 25 words
   **Score 4-6:** Avg 17-22 words, several sentences 25-30 words
   **Score 1-3:** Avg > 22 words, academic/formal style, incomplete thoughts split awkwardly
6. **originality (1-10):** Original insights, not just paraphrasing?
7. **length (1-10):** Word count appropriate? (8-10: ±20% of target, 5-7: ±40%, 1-4: >40% off)
8. **diagram_quality (1-10):** If diagram present: Is it clear and helpful? (10 if no diagram needed)

**Pass threshold:** Average score >= 8.0

For each dimension scoring below 8, create a CriticIssue with:
- dimension: The failing dimension name
- location: Where in the section (e.g., "paragraph 2", "code block 1")
- problem: What's wrong
- suggestion: Specific fix

**For completeness dimension specifically:**
If section is missing important information that requires additional research (e.g., missing comparisons, benchmarks, examples), set:
- needs_research = True
- suggested_queries = ["query 1", "query 2"] (2 specific search queries to find the missing info)

Examples of research-worthy gaps:
- "Missing Redis vs Memcached comparison" → needs_research=True, queries=["Redis Memcached comparison 2025", "Redis vs Memcached use cases"]
- "No performance benchmarks provided" → needs_research=True, queries=["[topic] performance benchmarks 2025", "[topic] benchmark comparison"]

Also identify any claims that need fact-checking (quantitative claims, benchmark numbers, version-specific features).

Output as SectionCriticResult with scores, overall_pass, issues, and fact_check_needed.
"""

    return prompt


async def _critic_section(
    section: dict,
    content: str,
    key_manager: KeyManager,
    blog_title: str = "",
    context: str = "",
    scratchpad: list[dict] | None = None,
) -> SectionCriticResult:
    """
    Call Gemini Flash-Lite to critique a section.

    Args:
        section: Section metadata
        content: Written section content
        key_manager: API key manager
        blog_title: Overall blog title for context
        context: User-provided context about the blog topic
        scratchpad: Refinement history (ReAct-style) showing previous attempts

    Returns:
        SectionCriticResult with scores and feedback
    """
    target_words = section.get("target_words", 200)

    prompt = _build_critic_prompt(
        section=section,
        content=content,
        target_words=target_words,
        blog_title=blog_title,
        context=context,
        scratchpad=scratchpad,
        style_guide=STYLE_GUIDE,
    )

    # Use Flash-Lite for critic (cheaper, faster)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_LITE,
        temperature=LLM_TEMPERATURE_LOW,
        google_api_key=key_manager.get_best_key(),
    )

    # Structured output for SectionCriticResult
    llm_structured = llm.with_structured_output(SectionCriticResult)

    # Run async LLM call
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: llm_structured.invoke(prompt),
    )

    # Calculate average score
    scores_dict = result.scores.model_dump()
    avg_score = sum(scores_dict.values()) / len(scores_dict)

    logger.info(
        f"Critic result for {section.get('id')}: "
        f"avg_score={avg_score:.1f}, "
        f"pass={result.overall_pass}, "
        f"issues={len(result.issues)}"
    )

    return result


def _build_refiner_prompt(
    section: dict,
    content: str,
    critic_issues: list[dict],
    scores: dict,
    scratchpad: list[dict] | None = None,
    style_guide: str = STYLE_GUIDE,
    additional_sources: list[dict] | None = None,
) -> str:
    """
    Build refinement prompt for improving a section based on critic feedback.

    Args:
        section: Section metadata (title, role, target_words)
        content: Current section content that failed critic
        critic_issues: List of CriticIssue dicts from SectionCriticResult
        scores: CriticScore dict with dimension scores
        scratchpad: Refinement history (ReAct-style) showing previous attempts
        style_guide: Writing style guidelines
        additional_sources: Extra sources fetched via dynamic research

    Returns:
        Refinement prompt string
    """
    section_title = section.get("title") or section.get("id", "")
    section_role = section.get("role", "")
    target_words = section.get("target_words", 200)

    # Format issues for prompt
    issues_text = "\n".join([
        f"**{i+1}. {issue['dimension']}** (score: {scores.get(issue['dimension'], '?')}/10)\n"
        f"   - Location: {issue['location']}\n"
        f"   - Problem: {issue['problem']}\n"
        f"   - Suggestion: {issue['suggestion']}\n"
        for i, issue in enumerate(critic_issues)
    ])

    # Calculate average score
    score_values = list(scores.values())
    avg_score = sum(score_values) / len(score_values) if score_values else 0

    # Build refinement history
    history_text = ""
    if scratchpad:
        history_text = "\n**Previous Refinement Attempts:**\n"
        for entry in scratchpad:  # Show all attempts so far
            attempt = entry['attempt']
            score = entry['score']
            addressed = entry.get('addressed_issues', [])

            if attempt == 0:
                history_text += f"\nAttempt {attempt} (Initial write): Score {score:.1f}/10\n"
                history_text += f"  Issues: {', '.join(entry['issues']) if entry['issues'] else 'None'}\n"
            else:
                history_text += f"\nAttempt {attempt}: Score {score:.1f}/10\n"
                history_text += f"  Tried to fix: {', '.join(addressed) if addressed else 'Unknown'}\n"
                history_text += f"  Result: {', '.join(entry['issues']) if entry['issues'] else 'No remaining issues'}\n"

        history_text += "\n**Important:** Learn from previous attempts. Don't repeat the same approach if it didn't work.\n"

    # Build additional sources section (from dynamic research)
    sources_text = ""
    if additional_sources:
        sources_text = "\n**Additional Research (Just Fetched):**\n\n"
        sources_text += "The critic identified missing information. Here are additional sources to fill the gaps:\n\n"
        for i, source in enumerate(additional_sources, 1):
            url = source.get('url', 'Unknown')
            title = source.get('title', 'Untitled')
            content = source.get('content', '')
            # Truncate content to first 500 chars to keep prompt manageable
            content_preview = content[:500] + "..." if len(content) > 500 else content
            sources_text += f"{i}. **{title}**\n"
            sources_text += f"   URL: {url}\n"
            sources_text += f"   Content: {content_preview}\n\n"
        sources_text += "**Use these sources to incorporate missing information, comparisons, benchmarks, or examples.**\n"

    prompt = f"""You are refining a technical blog section that did not pass quality review.

**Section Metadata:**
- Title: {section_title}
- Role: {section_role}
- Target words: {target_words}

**Current Content:**
```markdown
{content}
```

**Quality Scores:** {avg_score:.1f}/10 average (need 8.0+ to pass)
{history_text}
**Issues to Fix:**
{issues_text}
{sources_text}
**Writing Style Guidelines to Follow:**
{style_guide}

**Instructions:**
1. Read the current content carefully
2. Address EACH issue listed above with the suggested fixes
3. Preserve parts that are working well (high-scoring dimensions)
4. **CRITICAL FOR VOICE/STYLE ISSUES:**
   - Ensure sentences express complete thoughts (10-18 words target)
   - Combine choppy fragments into complete sentences using natural connectors (because, when, which)
   - Use simple words. Avoid complex constructions.
   - Replace excessive "however, therefore, moreover" with natural flow or fragments when appropriate
   - Replace "leverage, utilize, facilitate" with simple verbs (use, help, enable)
   - **Preserve original insights** - improve sentence structure only, don't lose unique ideas
   - Don't break complete thoughts into awkward fragments just to hit word count
5. Follow the writing style guidelines above (direct, opinionated, short sentences, no fluff)
6. Keep word count near target ({target_words} words, ±20%)
7. Ensure code examples are runnable with imports
8. Use original insights, not paraphrasing
9. If additional research sources are provided, incorporate relevant information to fill knowledge gaps

Output ONLY the refined markdown section. Do not include explanations or meta-commentary.
"""

    return prompt


async def _refine_section(
    section: dict,
    content: str,
    critic_result: SectionCriticResult,
    key_manager: KeyManager,
    scratchpad: list[dict] | None = None,
    additional_sources: list[dict] | None = None,
) -> str:
    """
    Refine a section based on critic feedback.

    Args:
        section: Section metadata
        content: Current section content that failed
        critic_result: CriticResult with issues and scores
        key_manager: API key manager
        scratchpad: Refinement history (ReAct-style)
        additional_sources: Extra sources fetched via dynamic research

    Returns:
        Refined markdown content
    """
    # Convert Pydantic models to dicts for prompt
    issues_list = [issue.model_dump() for issue in critic_result.issues]
    scores_dict = critic_result.scores.model_dump()

    # Build refinement prompt
    prompt = _build_refiner_prompt(
        section=section,
        content=content,
        critic_issues=issues_list,
        scores=scores_dict,
        scratchpad=scratchpad,
        style_guide=STYLE_GUIDE,
        additional_sources=additional_sources,
    )

    # Use Gemini Flash (same as _write_section)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_FULL,
        temperature=LLM_TEMPERATURE_LOW,
        google_api_key=key_manager.get_best_key(),
    )

    # Run async LLM call
    loop = asyncio.get_event_loop()
    refined_content = await loop.run_in_executor(
        None,
        lambda: llm.invoke(prompt).content,
    )

    logger.info(
        f"Refined section {section.get('id')}: "
        f"{len(content)} → {len(refined_content)} chars"
    )

    return refined_content


def _build_scratchpad_entry(
    attempt: int,
    critic_result: SectionCriticResult,
    previous_entry: dict | None = None,
    research_queries: list[str] | None = None,
    sources_fetched: int = 0,
) -> dict:
    """
    Build a scratchpad entry from critic result.

    Tracks refinement history for ReAct-style learning:
    - What score was achieved
    - Which issues were addressed vs. new
    - How scores changed per dimension
    - Whether dynamic research was performed

    Args:
        attempt: Attempt number (0 = initial write, 1+ = refinements)
        critic_result: CriticResult with scores and issues
        previous_entry: Previous scratchpad entry for comparison
        research_queries: Queries executed for dynamic research (optional)
        sources_fetched: Number of sources successfully fetched (default 0)

    Returns:
        Scratchpad entry dict with attempt, score, issues, changes, research info
    """
    # Extract scores and calculate average
    scores_dict = critic_result.scores.model_dump()
    current_score = sum(scores_dict.values()) / 8

    # Calculate score changes from previous attempt
    score_changes = {}
    if previous_entry:
        prev_scores = previous_entry["scores_breakdown"]
        for dim, score in scores_dict.items():
            prev_score = prev_scores.get(dim, score)
            diff = score - prev_score
            if diff > 0:
                score_changes[dim] = f"+{diff}"
            elif diff < 0:
                score_changes[dim] = str(diff)
            else:
                score_changes[dim] = "0"

    # Extract issue summaries (dimension: problem)
    issues = [f"{issue.dimension}: {issue.problem}" for issue in critic_result.issues]

    # Determine addressed and new issues
    addressed_issues = []
    new_issues = []
    if previous_entry:
        prev_issues_set = set(previous_entry["issues"])
        current_issues_set = set(issues)
        addressed_issues = list(prev_issues_set - current_issues_set)
        new_issues = list(current_issues_set - prev_issues_set)
    else:
        # First attempt: all issues are new
        new_issues = issues

    # Build scratchpad entry with research tracking
    entry = {
        "attempt": attempt,
        "score": round(current_score, 2),
        "scores_breakdown": scores_dict,
        "score_changes": score_changes,
        "issues": issues,
        "addressed_issues": addressed_issues,
        "new_issues": new_issues,
    }

    # Add research tracking fields if research was performed
    if research_queries:
        entry["research_performed"] = True
        entry["research_queries"] = research_queries
        entry["sources_fetched"] = sources_fetched
    else:
        entry["research_performed"] = False

    return entry


async def write_section_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 3: Write Section Node.

    Writes ONE section per invocation. The graph router controls looping
    through all sections.

    Args:
        state: Current BlogAgentState containing:
            - plan: BlogPlan with sections
            - validated_sources: Dict of section_id -> sources
            - current_section_index: Index of section to write
            - section_drafts: Dict of section_id -> content (previous sections)
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - section_drafts: Updated with new section content
        - current_section_index: Incremented for next section
        - current_phase: Stays WRITING (graph controls phase transition)
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.WRITING):
        logger.info("Skipping write_section_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "write_section"
    metrics, start_time = _init_node_metrics(state, node_name)
    write_calls = 0  # Track write calls (full model)
    critic_calls = 0  # Track critic calls (lite model)
    refine_calls = 0  # Track refine calls (full model)
    logger.info("Starting write section node")

    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    validated_sources = state.get("validated_sources", {})
    current_idx = state.get("current_section_index", 0)
    section_drafts = dict(state.get("section_drafts", {}))
    job_id = state.get("job_id", "")
    blog_title = plan.get("blog_title", state.get("title", ""))

    # Filter to selected sections (or fallback to required only)
    selected_ids = state.get("selected_section_ids", [])
    if selected_ids:
        # Use user selection
        required_sections = [s for s in sections if s["id"] in selected_ids]
    else:
        # Fallback: filter out optional (backward compatibility)
        required_sections = [s for s in sections if not s.get("optional")]

    if current_idx >= len(required_sections):
        logger.info("All sections written, moving to assembly")
        return {
            "current_section_index": current_idx,
            "section_drafts": section_drafts,
            "current_phase": Phase.ASSEMBLING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    section = required_sections[current_idx]
    section_id = section.get("id", f"section_{current_idx}")
    section_title = section.get("title") or section.get("role", "Section")

    logger.info(f"Writing section {current_idx + 1}/{len(required_sections)}: {section_title}")

    try:
        key_manager = KeyManager.from_env()

        # Get sources for this section
        sources = validated_sources.get(section_id, [])

        # Get previous sections for context
        previous_text = _get_previous_sections_text(section_drafts, required_sections[:current_idx])

        # Write the section
        content = await _write_section(
            section=section,
            sources=sources,
            previous_sections_text=previous_text,
            blog_title=blog_title,
            key_manager=key_manager,
        )
        write_calls += 1
        logger.info(f"Section '{section_title}' written ({len(content.split())} words)")

        # Initialize scratchpad for refinement tracking
        scratchpad = []

        # Critique the section with scratchpad context
        critic_result = await _critic_section(
            section=section,
            content=content,
            key_manager=key_manager,
            blog_title=blog_title,
            context=state.get("context", ""),
            scratchpad=scratchpad,
        )
        critic_calls += 1

        # Add initial entry to scratchpad
        scratchpad.append(_build_scratchpad_entry(0, critic_result))

        initial_score = scratchpad[0]["score"]
        logger.info(
            f"Critic for {section_id} (attempt 0): pass={critic_result.overall_pass}, "
            f"score={initial_score:.1f}/10, issues={len(critic_result.issues)}"
        )

        # Refine loop: improve section if it failed quality gate
        retry_count = 0
        best_content = content
        best_score = initial_score
        best_critic_result = critic_result

        while not critic_result.overall_pass and retry_count < MAX_SECTION_RETRIES:
            retry_count += 1
            logger.info(
                f"Section '{section_id}' failed quality gate (score {best_score:.1f}/10). "
                f"Refining... (attempt {retry_count}/{MAX_SECTION_RETRIES})"
            )

            # Check if any issues need dynamic research
            additional_sources = []
            research_queries_executed = []

            for issue in critic_result.issues:
                if issue.needs_research and issue.suggested_queries:
                    logger.info(
                        f"Dynamic research triggered for issue: {issue.dimension} - {issue.problem}"
                    )

                    # Execute suggested queries (limit to MAX_RESEARCH_QUERIES_PER_ISSUE)
                    for query in issue.suggested_queries[:MAX_RESEARCH_QUERIES_PER_ISSUE]:
                        logger.info(f"Searching: {query}")
                        research_queries_executed.append(query)

                        # Search DuckDuckGo
                        search_results = await search_duckduckgo(query)

                        # Fetch content from top URLs
                        for result in search_results[:MAX_RESEARCH_URLS_PER_QUERY]:
                            url = result.get('url', '')
                            if not url:
                                continue

                            logger.info(f"Fetching: {url}")
                            content_result = await fetch_url_content(url)

                            if content_result and content_result.get('content'):
                                additional_sources.append({
                                    'url': url,
                                    'title': result.get('title', 'Untitled'),
                                    'content': content_result['content'],
                                })
                                logger.info(
                                    f"Fetched {len(content_result['content'])} chars from {url}"
                                )

            if additional_sources:
                logger.info(
                    f"Dynamic research: {len(research_queries_executed)} queries, "
                    f"{len(additional_sources)} sources fetched"
                )

            # Refine content using critic feedback + scratchpad history + additional sources
            content = await _refine_section(
                section=section,
                content=content,
                critic_result=critic_result,
                key_manager=key_manager,
                scratchpad=scratchpad,
                additional_sources=additional_sources if additional_sources else None,
            )
            refine_calls += 1

            # Re-evaluate refined content with scratchpad context
            critic_result = await _critic_section(
                section=section,
                content=content,
                key_manager=key_manager,
                blog_title=blog_title,
                context=state.get("context", ""),
                scratchpad=scratchpad,
            )
            critic_calls += 1

            # Add refinement entry to scratchpad with research tracking
            scratchpad.append(
                _build_scratchpad_entry(
                    retry_count,
                    critic_result,
                    scratchpad[-1],
                    research_queries=research_queries_executed if research_queries_executed else None,
                    sources_fetched=len(additional_sources),
                )
            )

            # Track best version
            current_score = scratchpad[-1]["score"]
            if current_score > best_score:
                best_score = current_score
                best_content = content
                best_critic_result = critic_result
                logger.info(f"Refinement improved score: {best_score:.1f}/10")
            else:
                logger.info(
                    f"Refinement did not improve score: {current_score:.1f}/10 "
                    f"(best: {best_score:.1f}/10)"
                )

        # Use best version (either passed or best attempt)
        if critic_result.overall_pass:
            final_content = content
            final_critic_result = critic_result
            logger.info(
                f"Section '{section_id}' passed quality gate after {retry_count} refinement(s) "
                f"(final score: {scratchpad[-1]['score']:.1f}/10)"
            )
        else:
            final_content = best_content
            final_critic_result = best_critic_result
            logger.warning(
                f"Section '{section_id}' did not pass after {retry_count} refinement(s). "
                f"Using best version (score {best_score:.1f}/10)"
            )

        # Save scratchpad to disk
        if job_id:
            job_manager = JobManager()
            refinement_log_path = job_manager.get_job_dir(job_id) / "refinement_logs"
            refinement_log_path.mkdir(exist_ok=True)
            log_file = refinement_log_path / f"section_{section_id}_log.json"
            import json
            with open(log_file, "w") as f:
                json.dump(scratchpad, f, indent=2)
            logger.info(f"Saved refinement log to {log_file}")

        # Save to section_drafts (use final_content, not content)
        section_drafts[section_id] = final_content

        # Save critic result (use final_critic_result to match final_content)
        section_reviews = dict(state.get("section_reviews", {}))
        section_reviews[section_id] = final_critic_result.model_dump()

        # Save checkpoint
        if job_id:
            job_manager = JobManager()
            drafts_dir = job_manager.get_job_dir(job_id) / "drafts" / "sections"
            drafts_dir.mkdir(parents=True, exist_ok=True)
            (drafts_dir / f"{section_id}.md").write_text(content)

            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.WRITING.value,
                    "current_section_index": current_idx + 1,
                    "section_drafts": section_drafts,
                    "section_reviews": section_reviews,
                },
            )

        logger.info(f"Section '{section_title}' completed with critic evaluation")

        # Record LLM metrics
        # Write/refine use full model (~2000 in, ~800 out), critic uses lite model (~1500 in, ~300 out)
        if write_calls + refine_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_FULL,
                (write_calls + refine_calls) * 2000, (write_calls + refine_calls) * 800
            )
        if critic_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                critic_calls * 1500, critic_calls * 300
            )

        return {
            "section_drafts": section_drafts,
            "section_reviews": section_reviews,
            "current_section_index": current_idx + 1,
            "current_phase": Phase.WRITING.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except RuntimeError as e:
        logger.error(f"Section writing failed: {e}")
        # Record any LLM calls before failure
        if write_calls + refine_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_FULL,
                (write_calls + refine_calls) * 2000, (write_calls + refine_calls) * 800
            )
        if critic_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                critic_calls * 1500, critic_calls * 300
            )
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": str(e),
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    except Exception as e:
        logger.error(f"Unexpected error in section writing: {e}")
        # Record any LLM calls before failure
        if write_calls + refine_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_FULL,
                (write_calls + refine_calls) * 2000, (write_calls + refine_calls) * 800
            )
        if critic_calls > 0:
            _record_llm_call(
                metrics, node_name, LLM_MODEL_LITE,
                critic_calls * 1500, critic_calls * 300
            )
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Unexpected error: {e}",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


# =============================================================================
# Final Assembly Node (Phase 4)
# =============================================================================


MAX_FINAL_CRITIC_ITERATIONS = 2


def _build_final_critic_prompt(
    draft: str,
    plan: dict,
    blog_title: str,
    scratchpad: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build prompt for 7-dimension whole-blog evaluation.

    Evaluates:
    1. coherence - Do sections flow logically?
    2. voice_consistency - Same author voice throughout?
    3. no_redundancy - No repeated information?
    4. narrative_arc - Clear progression?
    5. hook_effectiveness - Opening captures attention?
    6. conclusion_strength - Clear takeaways?
    7. overall_polish - Professional quality?

    Also identifies transition issues between sections.

    Args:
        draft: Complete blog markdown
        plan: Blog plan with sections
        blog_title: Title of the blog
        scratchpad: Assembly iteration history to track improvements

    Returns:
        Final critic evaluation prompt
    """
    sections = plan.get("sections", [])
    section_ids = [s.get("id", "") for s in sections if not s.get("optional")]
    section_list = "\n".join(f"- {sid}" for sid in section_ids)

    word_count = len(draft.split())

    # Analyze overall sentence structure metrics for entire blog
    overall_metrics = _analyze_sentence_lengths(draft)

    # Build assembly history section from scratchpad
    assembly_history_section = ""
    if scratchpad and len(scratchpad) > 0:
        history_entries = []
        for entry in scratchpad:
            iteration = entry.get("iteration", 0)
            critic_scores = entry.get("critic_scores", {})
            overall_score = entry.get("overall_score", 0)
            passed = entry.get("passed", False)
            transition_fixes = entry.get("transition_fixes", [])
            fixes_applied = entry.get("fixes_applied", [])
            score_changes = entry.get("score_changes", {})

            # Format scores with changes
            scores_text = []
            for dim, score in critic_scores.items():
                change = score_changes.get(dim, 0)
                change_str = f" ({change:+.1f})" if change != 0 else ""
                scores_text.append(f"    - {dim}: {score:.1f}{change_str}")
            scores_str = "\n".join(scores_text) if scores_text else "    (no scores)"

            # Format transition fixes
            transitions_text = []
            for fix in transition_fixes:
                between = fix.get("between", [])
                issue = fix.get("issue", "")
                transitions_text.append(f"    - Between {' → '.join(between)}: {issue}")
            transitions_str = "\n".join(transitions_text) if transitions_text else "    (no transition issues)"

            # Format fixes applied
            fixes_str = "\n".join(f"    - {fix}" for fix in fixes_applied) if fixes_applied else "    (no fixes applied)"

            # Build entry text
            status_text = "✓ PASSED" if passed else "✗ NEEDS IMPROVEMENT"

            entry_text = f"""
### Iteration {iteration + 1}: {status_text} (Overall: {overall_score:.1f}/10)
**Scores:**
{scores_str}

**Transition Issues Identified:**
{transitions_str}

**Fixes Applied:**
{fixes_str}
"""
            history_entries.append(entry_text)

        assembly_history_section = f"""
## ASSEMBLY ITERATION HISTORY

You have evaluated this blog {len(scratchpad)} time(s) before. Track improvements and avoid regressions:

{''.join(history_entries)}

**CRITICAL INSTRUCTIONS:**
- DO NOT regress on scores that improved in previous iterations
- If a dimension improved (positive score change), maintain that improvement
- If transition fixes were applied, verify they actually improved the flow
- If a score keeps dropping despite fixes, identify the root cause clearly
- Focus critique on dimensions that haven't improved yet
"""

    prompt = f"""You are an expert technical blog editor. Evaluate this complete blog post on 7 whole-blog dimensions (1-10 scale).

**Blog Title:** {blog_title}

**Section Structure:**
{section_list}

**Complete Blog Draft:**
```markdown
{draft}
```

**Word Count:** {word_count}

**Overall Sentence Structure Metrics:**
- Average sentence length: {overall_metrics['avg_length']} words
- Longest sentence: {overall_metrics['max_length']} words
- Sentences > 20 words: {overall_metrics['long_sentences']}
- Simple sentences (≤18 words): {overall_metrics['percent_simple']:.1f}%

{assembly_history_section}

**Evaluation Criteria (7 Dimensions):**

1. **coherence (1-10):** Do sections flow logically? Do ideas connect well between sections? Is there a clear thread throughout?

2. **voice_consistency (1-10):** PUNCHY STYLE CONSISTENCY across all sections.
   - Check if sentence length is consistent throughout (avg should be 12-16 words across all sections)
   - Verify NO sections drift into formal/academic style (check for overly long sentences)
   - Flag any section that has avg > 18 words
   - Check for consistent conversational tone
   - No jarring shifts in technical level or formality
   - Ensure complete thoughts, not awkwardly fragmented ideas

   **Deduct points if:**
   - Overall avg > 18 words: Deduct 2 points
   - Any section significantly longer/shorter than others (>4 word avg difference): Deduct 1 point
   - Mix of punchy and formal styles: Deduct 2 points
   - Choppy fragmented writing that breaks complete thoughts: Deduct 1 point

3. **no_redundancy (1-10):** No repeated information across sections? Each section adds new value without rehashing previous content?

4. **narrative_arc (1-10):** Clear progression from beginning (hook/problem) through middle (solution/implementation) to end (conclusion/next steps)?

5. **hook_effectiveness (1-10):** Does the opening immediately capture attention? Does it set up the reader's problem clearly?

6. **conclusion_strength (1-10):** Does the ending provide clear, actionable takeaways? Strong call to action or next steps?

7. **overall_polish (1-10):** Professional quality? No rough edges, typos, or awkward phrasing?

**Pass threshold:** Average score >= 8.0 (overall_pass = True if all scores average >= 8)

**Transition Analysis:**
Identify any weak transitions between sections. For each weak transition, provide:
- between: [section_id_1, section_id_2]
- issue: What's wrong with the transition
- suggestion: How to improve it

**Output Requirements:**
- scores: All 7 dimension scores
- overall_pass: True if average >= 8.0
- transition_fixes: List of transition issues (can be empty if transitions are good)
- praise: 1-2 sentences on what's working well
- issues: General issues to address (list of strings)
- reading_time_minutes: Estimated reading time
- word_count: Total word count

Output as FinalCriticResult.
"""

    return prompt


async def _final_critic(
    draft: str,
    plan: dict,
    blog_title: str,
    key_manager: KeyManager,
    scratchpad: list[dict[str, Any]] | None = None,
    max_retries: int = 3,
) -> FinalCriticResult:
    """
    Call Gemini Flash for final critique of whole blog.

    Uses Flash (not Lite) for more complex whole-blog evaluation.

    Args:
        draft: Complete blog markdown
        plan: Blog plan with sections
        blog_title: Blog title
        key_manager: API key manager
        scratchpad: Assembly iteration history to track improvements
        max_retries: Max retries on failure

    Returns:
        FinalCriticResult with scores and transition fixes
    """
    prompt = _build_final_critic_prompt(draft, plan, blog_title, scratchpad)

    for attempt in range(max_retries):
        try:
            # Use Flash (not Lite) for complex whole-blog evaluation
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_FULL,
                temperature=LLM_TEMPERATURE_LOW,
                google_api_key=key_manager.get_best_key(),
            )

            llm_structured = llm.with_structured_output(FinalCriticResult)

            # Run async LLM call
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llm_structured.invoke(prompt),
            )

            # Calculate average score
            scores_dict = result.scores.model_dump()
            avg_score = sum(scores_dict.values()) / len(scores_dict)

            logger.info(
                f"Final critic: avg_score={avg_score:.1f}, "
                f"pass={result.overall_pass}, "
                f"transition_fixes={len(result.transition_fixes)}"
            )

            return result

        except Exception as e:
            logger.warning(f"Final critic attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Return a passing result on complete failure to not block pipeline
                logger.error("Final critic failed, returning default pass result")
                word_count = len(draft.split())
                reading_time = max(1, round(word_count / 200))
                return FinalCriticResult(
                    scores=FinalCriticScore(
                        coherence=8,
                        voice_consistency=8,
                        no_redundancy=8,
                        narrative_arc=8,
                        hook_effectiveness=8,
                        conclusion_strength=8,
                        overall_polish=8,
                    ),
                    overall_pass=True,
                    transition_fixes=[],
                    praise="Unable to evaluate, defaulting to pass.",
                    issues=["Final critic evaluation failed"],
                    reading_time_minutes=reading_time,
                    word_count=word_count,
                )


def _build_transition_fix_prompt(
    draft: str,
    fixes: list[TransitionFix],
) -> str:
    """
    Build prompt to improve transitions between sections.

    Args:
        draft: Current blog markdown
        fixes: List of transition issues to address

    Returns:
        Transition fix prompt
    """
    fixes_text = ""
    for i, fix in enumerate(fixes, 1):
        between = fix.between if isinstance(fix.between, list) else [fix.between]
        section_pair = " → ".join(between)
        fixes_text += f"""
{i}. **{section_pair}**
   - Issue: {fix.issue}
   - Suggestion: {fix.suggestion}
"""

    prompt = f"""You are an expert technical blog editor. Improve the transitions between sections based on the issues identified.

**Current Blog Draft:**
```markdown
{draft}
```

**Transition Issues to Fix:**
{fixes_text}

**Instructions:**
1. Find each transition point mentioned above
2. Improve the transition by adding a connecting sentence or paragraph
3. Ensure the flow feels natural and logical
4. Maintain the same voice and technical level
5. Don't add unnecessary fluff - keep transitions concise

**Important:**
- Return the COMPLETE blog with improved transitions
- Only modify the transition areas - don't change other content
- Keep the same markdown structure (headers, code blocks, etc.)

Return the improved blog markdown only, no explanations.
"""

    return prompt


async def _apply_transition_fixes(
    draft: str,
    fixes: list[TransitionFix],
    key_manager: KeyManager,
    max_retries: int = 3,
) -> str:
    """
    Refine transitions between sections based on critic feedback.

    Args:
        draft: Current blog markdown
        fixes: List of transition issues to fix
        key_manager: API key manager
        max_retries: Max retries on failure

    Returns:
        Refined blog markdown with improved transitions
    """
    if not fixes:
        return draft

    prompt = _build_transition_fix_prompt(draft, fixes)

    for attempt in range(max_retries):
        try:
            # Use Flash for rewriting
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_FULL,
                temperature=LLM_TEMPERATURE_MEDIUM,
                google_api_key=key_manager.get_best_key(),
            )

            # Run async LLM call
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llm.invoke(prompt),
            )

            # Extract content from response
            refined_draft = result.content.strip()

            # Remove markdown code fence if present
            if refined_draft.startswith("```markdown"):
                refined_draft = refined_draft[len("```markdown") :].strip()
            if refined_draft.startswith("```"):
                refined_draft = refined_draft[3:].strip()
            if refined_draft.endswith("```"):
                refined_draft = refined_draft[:-3].strip()

            logger.info(f"Applied {len(fixes)} transition fixes")
            return refined_draft

        except Exception as e:
            logger.warning(f"Transition fix attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Return original draft on failure
                logger.error("Transition fixes failed, keeping original draft")
                return draft


def _combine_sections(
    section_drafts: dict[str, str],
    plan: dict,
    blog_title: str,
    selected_section_ids: list[str] | None = None,
) -> str:
    """
    Combine all section drafts into a single markdown document.

    Args:
        section_drafts: Dict of section_id -> markdown content
        plan: Blog plan with sections list
        blog_title: Title of the blog post
        selected_section_ids: Optional list of section IDs to include (from user selection)

    Returns:
        Combined markdown with H1 title and H2 section headers
    """
    sections = plan.get("sections", [])

    # Filter to selected sections (or fallback to required only)
    if selected_section_ids:
        # Use user selection, preserve order from plan
        required_sections = [s for s in sections if s["id"] in selected_section_ids]
    else:
        # Fallback: filter out optional (backward compatibility)
        required_sections = [s for s in sections if not s.get("optional")]

    combined_parts = []

    # Add H1 title
    combined_parts.append(f"# {blog_title}\n")

    for section in required_sections:
        section_id = section.get("id", "")
        section_role = section.get("role", "")
        section_title = section.get("title")

        content = section_drafts.get(section_id, "")
        if not content:
            logger.warning(f"No content found for section {section_id}")
            continue

        # Hook section: no header, just content
        if section_role == "hook":
            combined_parts.append(content)
        else:
            # Other sections: add H2 header
            header = section_title or section_id.replace("_", " ").title()
            combined_parts.append(f"## {header}\n\n{content}")

    return "\n\n".join(combined_parts)


def _calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Calculate estimated reading time in minutes.

    Args:
        text: The text to measure
        words_per_minute: Reading speed (default 200 wpm)

    Returns:
        Reading time in minutes (minimum 1)
    """
    word_count = len(text.split())
    minutes = max(1, round(word_count / words_per_minute))
    return minutes


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _save_assembly_scratchpad(job_id: str, scratchpad: list[dict[str, Any]]) -> None:
    """Save final assembly scratchpad to disk."""
    if not job_id:
        return

    from pathlib import Path
    import json

    job_dir = Path.home() / ".blog_agent" / "jobs" / job_id
    scratchpad_dir = job_dir / "assembly_scratchpad"
    scratchpad_dir.mkdir(parents=True, exist_ok=True)

    scratchpad_file = scratchpad_dir / "iterations.json"
    with open(scratchpad_file, "w") as f:
        json.dump(scratchpad, f, indent=2, default=str)


def _load_assembly_scratchpad(job_id: str) -> list[dict[str, Any]]:
    """Load final assembly scratchpad from disk."""
    if not job_id:
        return []

    from pathlib import Path
    import json

    scratchpad_file = (
        Path.home() / ".blog_agent" / "jobs" / job_id / "assembly_scratchpad" / "iterations.json"
    )

    if not scratchpad_file.exists():
        return []

    with open(scratchpad_file) as f:
        return json.load(f)


def _build_assembly_scratchpad_entry(
    iteration: int,
    critic_result: Any,  # FinalCriticResult
    fixes_applied: list[str],
    prev_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build scratchpad entry for final assembly iteration."""
    from datetime import datetime

    # Calculate score changes from previous iteration
    # Convert FinalCriticScore Pydantic model to dict
    current_scores = critic_result.scores.model_dump() if hasattr(critic_result.scores, 'model_dump') else critic_result.scores.dict()
    if prev_scores:
        score_changes = {
            dim: current_scores.get(dim, 0) - prev_scores.get(dim, 0)
            for dim in current_scores.keys()
        }
    else:
        score_changes = {dim: 0 for dim in current_scores.keys()}

    overall_score = sum(current_scores.values()) / len(current_scores) if current_scores else 0

    return {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat(),
        "critic_scores": current_scores,
        "overall_score": overall_score,
        "passed": critic_result.overall_pass,
        "transition_fixes": critic_result.transition_fixes or [],
        "fixes_applied": fixes_applied,
        "score_changes": score_changes,
    }


async def final_assembly_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 4: Final Assembly Node.

    Combines all section drafts into a single markdown document,
    runs final critic with 7-dimension evaluation, and applies
    transition fixes if needed (max 2 iterations).

    Args:
        state: Current BlogAgentState containing:
            - plan: BlogPlan with sections
            - section_drafts: Dict of section_id -> content
            - job_id: Job identifier for checkpointing

    Returns:
        State update dict with:
        - combined_draft: Full blog markdown (pre-critic)
        - final_markdown: Refined blog markdown (post-critic)
        - final_review: FinalCriticResult as dict
        - metadata: Blog metadata with critic scores
        - current_phase: Updated to REVIEWING
        - metrics: Updated metrics dict
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.ASSEMBLING):
        logger.info("Skipping final_assembly_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "final_assembly"
    metrics, start_time = _init_node_metrics(state, node_name)
    critic_calls = 0
    fix_calls = 0
    logger.info("Starting final assembly")

    plan = state.get("plan", {})
    section_drafts = state.get("section_drafts", {})
    job_id = state.get("job_id", "")
    blog_title = plan.get("blog_title", state.get("title", "Untitled"))
    selected_section_ids = state.get("selected_section_ids", [])

    # Initialize scratchpad
    scratchpad = state.get("final_assembly_scratchpad", [])
    if not scratchpad:
        scratchpad = []

    if not section_drafts:
        logger.error("No section drafts to assemble")
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": "No section drafts to assemble",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    try:
        # Initialize key manager for LLM calls
        key_manager = KeyManager.from_env()

        # Combine sections into single document
        combined_draft = _combine_sections(
            section_drafts=section_drafts,
            plan=plan,
            blog_title=blog_title,
            selected_section_ids=selected_section_ids if selected_section_ids else None,
        )

        logger.info(f"Combined draft: {_count_words(combined_draft)} words")

        # Final critic loop (max 2 iterations)
        final_draft = combined_draft
        critic_result = None

        for iteration in range(MAX_FINAL_CRITIC_ITERATIONS):
            logger.info(f"Final critic iteration {iteration + 1}/{MAX_FINAL_CRITIC_ITERATIONS}")

            critic_result = await _final_critic(
                draft=final_draft,
                plan=plan,
                blog_title=blog_title,
                key_manager=key_manager,
                scratchpad=scratchpad,
            )
            critic_calls += 1

            # Build scratchpad entry (fixes_applied will be updated if we apply fixes)
            prev_scores = scratchpad[-1]["critic_scores"] if scratchpad else None
            scratchpad_entry = _build_assembly_scratchpad_entry(
                iteration=iteration,
                critic_result=critic_result,
                fixes_applied=[],
                prev_scores=prev_scores,
            )
            scratchpad.append(scratchpad_entry)

            # Save to disk after each iteration
            _save_assembly_scratchpad(job_id, scratchpad)

            if critic_result.overall_pass:
                logger.info("Final critic passed")
                break

            # Apply transition fixes if any and not last iteration
            if critic_result.transition_fixes and iteration < MAX_FINAL_CRITIC_ITERATIONS - 1:
                logger.info(f"Applying {len(critic_result.transition_fixes)} transition fixes")
                final_draft = await _apply_transition_fixes(
                    draft=final_draft,
                    fixes=critic_result.transition_fixes,
                    key_manager=key_manager,
                )
                fix_calls += 1

                # Update scratchpad entry with applied fixes
                scratchpad[-1]["fixes_applied"] = critic_result.transition_fixes
                _save_assembly_scratchpad(job_id, scratchpad)

        # Calculate metadata
        word_count = _count_words(final_draft)
        reading_time = _calculate_reading_time(final_draft)

        # Build metadata with critic scores
        metadata = {
            "blog_title": blog_title,
            "word_count": word_count,
            "reading_time_minutes": reading_time,
            "section_count": len([s for s in plan.get("sections", []) if not s.get("optional")]),
        }

        # Add critic scores to metadata
        if critic_result:
            scores_dict = critic_result.scores.model_dump()
            avg_score = sum(scores_dict.values()) / len(scores_dict)
            metadata["final_critic_scores"] = scores_dict
            metadata["final_critic_avg_score"] = round(avg_score, 2)
            metadata["final_critic_pass"] = critic_result.overall_pass

        logger.info(f"Final blog: {word_count} words, {reading_time} min read")

        # Prepare final review dict
        final_review = critic_result.model_dump() if critic_result else None

        # Save checkpoint
        if job_id:
            import json
            job_manager = JobManager()
            job_dir = job_manager.get_job_dir(job_id)

            # Ensure directories exist
            drafts_dir = job_dir / "drafts"
            drafts_dir.mkdir(parents=True, exist_ok=True)
            feedback_dir = job_dir / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)

            # Save combined draft (v1 - pre-critic)
            (drafts_dir / "v1.md").write_text(combined_draft)

            # Save refined draft (v2 - post-critic) if different
            if final_draft != combined_draft:
                (drafts_dir / "v2.md").write_text(final_draft)

            # Save final.md
            (job_dir / "final.md").write_text(final_draft)

            # Save final critic result
            if final_review:
                (feedback_dir / "final_critic.json").write_text(
                    json.dumps(final_review, indent=2)
                )

            # Save metadata
            (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            # Update state
            job_manager.save_state(
                job_id,
                {
                    "current_phase": Phase.REVIEWING.value,
                    "combined_draft": combined_draft,
                    "final_markdown": final_draft,
                    "final_review": final_review,
                    "metadata": metadata,
                },
            )

        # Record LLM metrics (critic uses lite model ~3000 in, 500 out; fixes use full ~2000 in, 500 out)
        if critic_calls > 0:
            _record_llm_call(metrics, node_name, LLM_MODEL_LITE, critic_calls * 3000, critic_calls * 500)
        if fix_calls > 0:
            _record_llm_call(metrics, node_name, LLM_MODEL_FULL, fix_calls * 2000, fix_calls * 500)

        return {
            "combined_draft": combined_draft,
            "final_markdown": final_draft,
            "final_review": final_review,
            "metadata": metadata,
            "current_phase": Phase.REVIEWING.value,
            "final_assembly_scratchpad": scratchpad,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }

    except Exception as e:
        logger.error(f"Final assembly failed: {e}")
        # Record any LLM calls before failure
        if critic_calls > 0:
            _record_llm_call(metrics, node_name, LLM_MODEL_LITE, critic_calls * 3000, critic_calls * 500)
        if fix_calls > 0:
            _record_llm_call(metrics, node_name, LLM_MODEL_FULL, fix_calls * 2000, fix_calls * 500)
        return {
            "current_phase": Phase.FAILED.value,
            "error_message": f"Final assembly failed: {e}",
            "final_assembly_scratchpad": scratchpad,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }


# =============================================================================
# Human Review Node (Phase 5)
# =============================================================================


async def human_review_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Simple human review with Rich prompt.

    Displays completion message and asks user to approve or quit.
    """
    # Skip if phase already past this node (resumption)
    if phase_is_past(state.get("current_phase", ""), Phase.REVIEWING):
        logger.info("Skipping human_review_node - phase already past")
        return _preserve_key_manager(state)

    node_name = "human_review"
    metrics, start_time = _init_node_metrics(state, node_name)

    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()
    job_id = state.get("job_id", "")
    final_review = state.get("final_review", {})

    # Display completion summary
    console.print(f"\n[bold green]✓ Blog Complete[/bold green]")
    console.print(f"Job: {job_id}")
    console.print(f"Output: ~/.blog_agent/jobs/{job_id}/final.md\n")

    # Show critic scores if available
    if final_review and "scores" in final_review:
        scores = final_review["scores"]
        console.print("[bold]Final Critic Scores:[/bold]")
        for dim, score in scores.items():
            console.print(f"  {dim}: {score}/10")
        console.print()

    # Prompt for approval
    approved = Confirm.ask("Approve and finalize?", default=True)

    if approved:
        return {
            "human_review_decision": "approve",
            "current_phase": Phase.DONE.value,
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }
    else:
        return {
            "human_review_decision": "quit",
            "metrics": _finalize_node_metrics(metrics, node_name, start_time),
        }