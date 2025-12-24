# Vertical Slice Implementation Plan

**Strategy**: Build thin slices across all layers first, then add depth. Each slice produces testable, working functionality.

**Testing**: Unit + Integration tests for every slice.

**Quality Standards**: Content evaluated against both internal criteria (8 dimensions from design.md) AND [Google's "Creating Helpful Content" guidelines](https://developers.google.com/search/docs/fundamentals/creating-helpful-content) (E-E-A-T: Experience, Expertise, Authoritativeness, Trustworthiness).

---

## Status Overview

| Round | Slices | Goal | Status |
|-------|--------|------|--------|
| Foundation | 0.1-0.4 | Tools, State, Discovery, Content Analysis | ✓ DONE |
| Pre-Foundation | Phases 4-5 | Planning, Research, Validation | ✓ DONE |
| Round 1 | 6.1-6.4 | Minimal End-to-End CLI | ✓ DONE |
| Round 2 | 6.5-6.8 | Quality Layer (15D Critic + Differentiation + Refine) | ○ PENDING |
| Round 3 | 6.9 | Title + SEO Generation | ○ PENDING |
| Round 4 | 6.10-6.12 | Polish Features (Mermaid, Citations, Fact Check) | ○ PENDING |
| Round 5 | 7.1-7.10 | Web UI with Reflex | ○ PENDING |

---

## Design Constraints & Infrastructure

### Token Budget (UPDATED with Content Landscape Analysis)
- **Total per blog**: ~36 LLM calls, ~115k tokens in, ~27.5k tokens out
- **Breakdown by phase**:
  - Topic Discovery: 1 call (Flash-Lite) - 500 in, 200 out
  - **NEW Content Landscape Analysis**: ~11 calls (Flash-Lite) - ~13,000 in, ~2,500 out
    - Article analysis: ~10 calls (1k in, 200 out each)
    - Strategy synthesis: 1 call (3k in, 500 out)
  - Planning (enhanced): 1 call (Flash-Lite) - 2,500 in, 1,000 out
  - Research Validation: 1 call (Flash-Lite) - 4,000 in, 500 out
  - Per Section (×6): ~3 calls (Flash) - ~13,500 in, ~3,400 out per section
    - Now evaluates **15 dimensions** instead of 10 (same token count, more comprehensive)
  - Final Assembly: ~2 calls (Flash) - 14,000 in, 3,000 out
- **Budget check**: 4 projects × 250 RPD = 1,000 requests/day ✓ (36 calls << 1,000 RPD)

### Checkpoint Directory Structure
```
~/.blog_agent/jobs/{job_id}/
├── state.json              # Current phase, progress, can_resume
├── input.json              # Original title + context
├── topic_context.json      # Discovery search results
├── content_strategy.json   # NEW: Content landscape analysis, unique angle, gaps to fill
├── plan.json               # Outline, search queries, metadata
├── research/
│   ├── cache/              # Raw fetched articles (for resume)
│   ├── validated/          # Post-quality-filter sources
│   └── sources.json        # URL → section mapping for citations
├── drafts/
│   ├── sections/           # Individual section drafts
│   ├── v1.md               # Combined draft v1
│   └── v2.md               # After refinement
├── feedback/
│   ├── section_{n}_critic.json  # Now includes 15D scores + differentiation flags
│   └── final_critic.json
├── human_inputs/           # User guidance/feedback saved
├── fact_check.md           # Claims to verify
├── images/                 # Rendered mermaid PNGs
└── final.md                # Approved output
```

### Quota Manager (KeyManager in tools.py)
- **Projects**: 4 Google Cloud API keys (GOOGLE_API_KEY_1 through 4)
- **Tracking**: RPD (requests per day) and TPM (tokens per minute) per project
- **Selection**: Pick project with most remaining RPD
- **On 429 error**: Switch to next project
- **All exhausted**: Pause job, notify human
- **Reset**: Counts reset at midnight Pacific Time
- **Logging**: Every request logged with usageMetadata

### Retry Strategies (from design.md RETRY MANAGER)
```yaml
gemini_api:
  max_retries: 3
  backoff: [1s, 2s, 4s]  # Exponential
  on_429: switch_project
  on_all_exhausted: pause_notify_human
  on_500: retry_same_project

web_search:
  max_retries: 2
  backoff: [2s, 5s]
  on_rate_limit: wait_60s_retry
  on_fail: try_alternate_query

web_fetch:
  max_retries: 2
  timeout: 30s
  on_fail: skip_source      # Don't fail whole job
  on_timeout: skip_source

mermaid_render:
  max_retries: 2
  fallback: save_raw_mermaid  # Human can render manually
```

### Google Quality Guidelines Integration + Content Differentiation

Content evaluated against [Google's "Creating Helpful Content" guidelines](https://developers.google.com/search/docs/fundamentals/creating-helpful-content) AND enhanced differentiation criteria to create **truly unique, bookmarkable content**.

**E-E-A-T Framework (Experience, Expertise, Authoritativeness, Trustworthiness):**
- **eeat_expertise** (scoring dimension): Does content demonstrate first-hand expertise and depth of knowledge?
- **people_first_value** (scoring dimension): Serves readers directly? Would you bookmark/share this?

**NEW: Content Differentiation Dimensions (5 additional scoring dimensions):**
1. **unique_insights** (1-10): Provides insights not found in existing content? Novel perspective?
2. **actionability** (1-10): Clear next steps? High action-to-theory ratio? Copy-paste ready code?
3. **specificity** (1-10): Concrete examples with real numbers? Specific tools/versions? Not vague?
4. **depth_appropriateness** (1-10): Right depth for topic? Covers edge cases? Explains 'why' not just 'how'?
5. **production_readiness** (1-10): Code has error handling? Security considerations? Real-world concerns?

**Total Scoring Dimensions: 15** (8 original + 2 E-E-A-T + 5 differentiation)

**Quality Red Flags to Detect:**

*Google Quality Flags:*
1. **written_for_word_count**: Content artificially padded to hit word count targets
2. **merely_summarizing_sources**: Just summarizing others' work without substantial additions
3. **lacks_first_hand_expertise**: Doesn't demonstrate clear experience with the topic
4. **surface_level_coverage**: Doesn't go beyond surface-level to offer substantial depth

*NEW Differentiation Flags:*
5. **rehashes_existing_content**: Says same things as top articles without new angle?
6. **generic_examples**: Uses foo/bar examples instead of real scenarios?
7. **missing_edge_cases**: Doesn't cover gotchas, limitations, or when NOT to use?
8. **vague_claims**: Says 'improves performance' without specific benchmarks?
9. **no_production_context**: Code examples lack error handling, logging, monitoring?

**Quality Criteria:**
- Provides original information, research, or analysis
- Goes beyond surface-level to offer substantial depth
- Adds substantial value beyond just summarizing sources
- Demonstrates first-hand expertise and depth of knowledge
- Is this something you'd bookmark, share with a friend, or recommend?
- Leaves readers satisfied they've learned enough
- NOT written primarily to hit word count targets
- NOT merely summarizing others' work without additions
- Shows genuine experience with the topic
- **NEW**: Provides unique angle based on content landscape analysis
- **NEW**: Includes concrete, specific examples with real numbers and versions
- **NEW**: Clear actionable next steps (not just theory)
- **NEW**: Covers edge cases and when NOT to use the solution
- **NEW**: Code examples are educational/conceptual (not necessarily production-tested)

**Integration Points:**
- **Content Landscape Analysis (Slice 0.4)**: Analyze top 10 articles, identify unique angle BEFORE planning
- **Section Critic (Slice 6.5)**: Evaluates 15 dimensions (8 original + 2 E-E-A-T + 5 differentiation)
- **Section Refiner (Slice 6.6)**: Addresses Google quality flags AND differentiation flags when refining
- **Writer Prompt Enhancement**: Strict actionability requirements (concrete examples, complete code, next steps, edge cases)
- **Web UI Review (Slice 7.6)**: Displays all quality and differentiation flags to human reviewer

---

## FOUNDATION: Core Infrastructure (Slices 0.1-0.3)

**Goal**: Set up tools, state management, and checkpoint system before implementing phases

---

### Slice 0.1: Tools & Utilities

**Status**: ✓ COMPLETE

**Files created**:
- `src/agent/tools.py` - Core utilities and API wrappers

**Implementation**:
```python
# Web search and content extraction
def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search via DuckDuckGo API, return titles/URLs/snippets."""

def fetch_url_content(url: str, timeout: int = 30) -> str:
    """Fetch and extract article content via trafilatura."""

def chunk_content(text: str, max_tokens: int = 4000) -> list[str]:
    """Split long articles by headings/paragraphs."""

# Quality checks
def check_originality(content: str, sources: list[dict]) -> list[dict]:
    """
    Plagiarism detection with difflib.SequenceMatcher.

    For each sentence in content:
      - Compare against all source chunks
      - Flag if similarity > 0.7 (70%)
      - Check N-gram overlap (3-gram, 4-gram)
      - Flag exact phrase matches > 8 words

    Returns: List of flagged sentences with similarity scores
    """

# Mermaid diagram rendering
async def render_mermaid(mermaid_code: str, output_path: str) -> bool:
    """
    Render mermaid diagram to PNG via kroki.io.

    1. Base64 encode mermaid code
    2. POST to https://kroki.io/mermaid/png/{encoded}
    3. Save response as PNG
    4. On failure: return False (caller keeps raw mermaid)
    """

# API key management
class KeyManager:
    """
    Manages 4 Google API keys with quota tracking.

    - Tracks RPD (requests/day) and TPM (tokens/min) per project
    - Selects key with most remaining quota
    - On 429: Switches to next project
    - Resets counters at midnight Pacific
    - Logs all requests with usageMetadata
    """
```

**Tests**:
- `tests/unit/test_tools.py`: Mock DuckDuckGo, trafilatura, kroki
- `tests/integration/test_tools_integration.py`: Real API calls

**Milestone**: All utility functions working and tested

---

### Slice 0.2: State & Checkpoints

**Status**: ✓ COMPLETE

**Files created**:
- `src/agent/state.py` - State schema, Pydantic models, Phase enum
- `src/agent/job_manager.py` - Checkpoint save/load (or in nodes.py)

**Implementation**:
```python
# state.py
from enum import Enum
from typing import TypedDict, Any
from pydantic import BaseModel, Field

class Phase(Enum):
    """Pipeline phases for state tracking."""
    TOPIC_DISCOVERY = "topic_discovery"
    PLANNING = "planning"
    RESEARCHING = "researching"
    VALIDATING_SOURCES = "validating_sources"
    WRITING = "writing"
    REVIEWING = "reviewing"
    ASSEMBLING = "assembling"
    FINAL_REVIEW = "final_review"
    DONE = "done"
    FAILED = "failed"

class BlogAgentState(TypedDict):
    """Complete state that flows through all nodes."""
    job_id: str
    title: str
    context: str
    target_length: str
    current_phase: str
    current_section_index: int
    topic_context: list[dict]
    plan: dict
    research_cache: dict
    validated_sources: dict
    section_drafts: dict
    fact_check_items: list[str]
    final_draft: str
    metadata: dict
    # ... (all state fields)

# Pydantic models for LLM outputs
class DiscoveryQueries(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=5)

class PlanSection(BaseModel):
    id: str
    title: str | None
    role: str
    search_queries: list[str]
    needs_code: bool
    needs_diagram: bool
    target_words: int

class BlogPlan(BaseModel):
    blog_title: str
    target_words: int
    sections: list[PlanSection]

# ... (all other Pydantic models)

# Checkpoint functions
def create_job(title: str, context: str, length: str) -> str:
    """
    Initialize job directory structure.

    Job ID: Slugified topic (e.g., "semantic-caching-for-llm-applications")
    Creates: ~/.blog_agent/jobs/{job_id}/ with subdirs
    Returns: job_id
    """

def save_state(job_id: str, state: BlogAgentState):
    """Save state.json checkpoint."""

def load_state(job_id: str) -> BlogAgentState:
    """Load state for resume."""
```

**Tests**:
- `tests/unit/test_checkpoints.py`: Create job, save/load round-trip
- `tests/unit/test_state.py`: Pydantic model validation

**Milestone**: State management and checkpointing working

---

### Slice 0.3: Topic Discovery (Phase 0.5)

**Status**: ✓ COMPLETE

**Files modified**:
- `src/agent/nodes.py` - Add topic_discovery_node

**Implementation**:
```python
async def topic_discovery_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 0.5: Topic Discovery Node.

    3-step process:
    1. Generate 3-5 search queries (LLM: Flash-Lite)
    2. Execute DuckDuckGo search for each query
    3. Compile top 15-20 unique snippets

    Saves: topic_context.json
    """

    # Step 1: Generate discovery queries
    prompt = f"""
    Generate 3-5 search queries to learn about this topic:

    Title: "{state['title']}"
    Context: "{state['context']}"

    Goals:
    - Understand what this topic is about
    - Find key subtopics and concepts
    - Discover recent developments (2024-2025)
    - Identify practical use cases

    Output JSON: {{"queries": ["...", "...", ...]}}
    """

    queries = await _call_llm(prompt, DiscoveryQueries, key_manager)

    # Step 2: Search for each query
    results = []
    for query in queries.queries:
        search_results = search_duckduckgo(query, max_results=5)
        results.extend(search_results)

    # Step 3: Deduplicate and compile
    unique_results = _deduplicate_by_url(results)[:20]

    topic_context = {
        "queries_used": queries.queries,
        "results": unique_results,
        "result_count": len(unique_results)
    }

    save_json(job_id, "topic_context.json", topic_context)

    return {
        "topic_context": unique_results,
        "current_phase": Phase.PLANNING.value
    }
```

**Tests**:
- `tests/unit/test_topic_discovery.py`: Query generation, deduplication
- `tests/integration/test_topic_discovery_integration.py`: Full discovery with real search

**Milestone**: Topic context gathered before planning

---

### Slice 0.4: Content Landscape Analysis (NEW - Pre-Planning Phase)

**Status**: ✓ COMPLETE

**Goal**: Analyze top 10 existing articles on the topic to identify content gaps and determine a unique angle BEFORE planning the blog structure.

**Files to create**:
- `src/agent/models.py` - NEW models for content analysis

**Files to modify**:
- `src/agent/state.py` - Add ContentStrategy to BlogAgentState
- `src/agent/nodes.py` - Add content_landscape_analysis_node
- `src/agent/graph.py` - Insert new node between topic_discovery and planning

**New Pydantic Models** (add to `src/agent/models.py` or `state.py`):
```python
class ContentGap(BaseModel):
    """Identified gap in existing content."""
    gap_type: str = Field(
        description="Type of gap: missing_topic | insufficient_depth | wrong_info | no_examples | missing_edge_cases"
    )
    description: str = Field(description="What's missing in existing articles")
    opportunity: str = Field(description="How we'll fill this gap uniquely")

class ExistingArticleSummary(BaseModel):
    """Summary of one analyzed article."""
    url: str
    title: str
    main_angle: str = Field(description="What angle does this article take?")
    strengths: list[str] = Field(description="What it does well")
    weaknesses: list[str] = Field(description="What's missing or weak")
    key_points_covered: list[str] = Field(description="Main points covered")

class ContentStrategy(BaseModel):
    """Strategy for differentiated blog content."""
    unique_angle: str = Field(
        description="Our differentiated perspective (chosen by LLM based on gaps). Examples: 'focus on production pitfalls', 'emphasize cost optimization', 'compare 3 implementations'"
    )
    target_persona: str = Field(
        description="Primary reader: junior_engineer | senior_architect | data_scientist | devops_engineer | etc."
    )
    reader_problem: str = Field(
        description="Specific problem they're solving (e.g., 'reduce LLM API costs by 50%')"
    )
    gaps_to_fill: list[ContentGap] = Field(
        min_length=1,
        max_length=5,
        description="Top 3-5 content gaps we'll address"
    )
    existing_content_summary: str = Field(
        description="1-2 sentence summary of what top articles already cover well"
    )
    analyzed_articles: list[ExistingArticleSummary] = Field(
        min_length=5,
        max_length=10,
        description="5-10 top articles analyzed"
    )
    differentiation_requirements: list[str] = Field(
        description="Specific requirements to ensure uniqueness (e.g., 'must include benchmarks', 'must cover edge cases X, Y, Z')"
    )
```

**Add to BlogAgentState**:
```python
class BlogAgentState(TypedDict):
    # ... existing fields ...
    content_strategy: dict | None  # NEW: ContentStrategy from landscape analysis
```

**Implementation** (add to `src/agent/nodes.py`):
```python
async def content_landscape_analysis_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 0.6: Content Landscape Analysis (NEW).

    Runs BEFORE planning to identify unique angle.

    Process:
    1. Use topic_context from discovery (top 15-20 search results)
    2. Select top 10 most relevant URLs based on:
       - Domain authority (prefer established tech blogs, official docs)
       - Title relevance to our topic
       - Recency (prefer 2023-2025 articles)
    3. Fetch full content for top 10 articles (via fetch_url_content)
    4. LLM analyzes each article (Flash-Lite):
       - Extract main angle, strengths, weaknesses, key points
       - Generate ExistingArticleSummary for each
    5. LLM synthesizes analysis (Flash-Lite):
       - Identify content gaps (what's missing across all articles)
       - Determine unique angle based on gaps
       - Select target persona (who needs this most)
       - Define reader problem we'll solve
       - List differentiation requirements
    6. Save ContentStrategy to state

    Saves: content_strategy.json
    """

    job_id = state.get("job_id", "")
    topic_context = state.get("topic_context", [])

    # Step 1: Select top 10 URLs from topic_context
    # Sort by relevance, domain authority, recency
    top_urls = _select_top_urls(topic_context, max_count=10)

    # Step 2: Fetch full content for all URLs (with retry/skip on failure)
    articles_content = []
    for url_data in top_urls:
        try:
            content = fetch_url_content(url_data['url'], timeout=30)
            if content:
                articles_content.append({
                    "url": url_data['url'],
                    "title": url_data.get('title', ''),
                    "content": content[:10000]  # Limit to 10k chars per article
                })
        except Exception as e:
            # Skip failed URLs, don't fail whole analysis
            logger.warning(f"Failed to fetch {url_data['url']}: {e}")
            continue

    # Step 3: Analyze each article with LLM (Flash-Lite)
    analyzed_articles = []
    for article in articles_content:
        analysis_prompt = f"""
        Analyze this article on "{state['title']}":

        URL: {article['url']}
        Title: {article['title']}
        Content (first 10k chars):
        {article['content']}

        Extract:
        1. main_angle: What unique perspective does this article take?
        2. strengths: What does it do well? (2-3 points)
        3. weaknesses: What's missing or weak? (2-3 points)
        4. key_points_covered: Main topics/sections covered (3-5 bullet points)

        Output JSON as ExistingArticleSummary model.
        """

        analysis = await _call_llm(
            analysis_prompt,
            ExistingArticleSummary,
            key_manager,
            model="flash-lite"
        )
        analyzed_articles.append(analysis)

    # Step 4: Synthesize content strategy (Flash-Lite)
    synthesis_prompt = f"""
    You are analyzing the content landscape for a blog on: "{state['title']}"

    Context provided by user:
    {state['context']}

    I've analyzed {len(analyzed_articles)} top articles on this topic.
    Here's what they cover:

    {_format_article_summaries(analyzed_articles)}

    Your task: Create a ContentStrategy that ensures our blog is UNIQUE and VALUABLE.

    Requirements:
    1. Identify 3-5 content gaps (what's missing, shallow, or wrong in existing articles)
    2. Choose a unique_angle that fills these gaps (e.g., "focus on production pitfalls others ignore", "provide concrete benchmarks", "compare 3 real implementations")
    3. Select target_persona (who needs this most: junior_engineer, senior_architect, etc.)
    4. Define reader_problem (specific problem they're solving)
    5. List differentiation_requirements (specific things our blog MUST include to stand out)

    Examples of good unique angles:
    - "Production-focused guide covering error handling, monitoring, and scaling (others focus only on basics)"
    - "Benchmark-driven comparison of 3 caching strategies with real numbers (others are purely theoretical)"
    - "Cost optimization angle: how to reduce LLM API bills by 60% (others don't mention costs)"
    - "Edge cases and limitations deep-dive (others only show happy path)"

    Output JSON as ContentStrategy model.
    """

    content_strategy = await _call_llm(
        synthesis_prompt,
        ContentStrategy,
        key_manager,
        model="flash-lite"
    )

    # Step 5: Save to checkpoint
    save_json(job_id, "content_strategy.json", content_strategy.dict())

    logger.info(f"Content strategy: {content_strategy.unique_angle}")
    logger.info(f"Gaps to fill: {len(content_strategy.gaps_to_fill)}")

    return {
        "content_strategy": content_strategy.dict(),
        "current_phase": Phase.PLANNING.value
    }

def _select_top_urls(topic_context: list[dict], max_count: int = 10) -> list[dict]:
    """
    Select top N URLs from topic context based on relevance and quality.

    Scoring criteria:
    - Domain authority (higher for .edu, official docs, established tech blogs)
    - Recency (prefer 2023-2025)
    - Title relevance to topic
    """
    # Simple implementation: take first max_count
    # TODO: Add scoring logic for domain authority, recency
    return topic_context[:max_count]

def _format_article_summaries(articles: list[ExistingArticleSummary]) -> str:
    """Format analyzed articles for synthesis prompt."""
    formatted = []
    for i, article in enumerate(articles, 1):
        formatted.append(f"""
Article {i}: {article.title}
URL: {article.url}
Angle: {article.main_angle}
Strengths: {', '.join(article.strengths)}
Weaknesses: {', '.join(article.weaknesses)}
Covers: {', '.join(article.key_points_covered)}
""")
    return "\n".join(formatted)
```

**Update graph.py** to include new node:
```python
def build_blog_agent_graph() -> StateGraph:
    graph = StateGraph(BlogAgentState)

    # Add nodes
    graph.add_node("topic_discovery", topic_discovery_node)
    graph.add_node("content_landscape_analysis", content_landscape_analysis_node)  # NEW
    graph.add_node("planning", planning_node)
    # ... rest of nodes

    # Set entry
    graph.set_entry_point("topic_discovery")

    # Linear edges
    graph.add_edge("topic_discovery", "content_landscape_analysis")  # NEW
    graph.add_edge("content_landscape_analysis", "planning")  # NEW
    # ... rest of edges
```

**Update planning_node** to use ContentStrategy:
```python
async def planning_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 1: Planning Node (ENHANCED with ContentStrategy).

    Now receives content_strategy from landscape analysis.
    Uses unique_angle and gaps_to_fill to generate differentiated plan.
    """

    content_strategy = state.get("content_strategy", {})

    planning_prompt = f"""
    Create a blog outline for: "{state['title']}"

    User context:
    {state['context']}

    CONTENT STRATEGY (from landscape analysis):
    Unique Angle: {content_strategy.get('unique_angle', 'N/A')}
    Target Reader: {content_strategy.get('target_persona', 'experienced engineer')}
    Reader Problem: {content_strategy.get('reader_problem', 'N/A')}

    Content Gaps to Fill:
    {_format_gaps(content_strategy.get('gaps_to_fill', []))}

    Differentiation Requirements:
    {_format_requirements(content_strategy.get('differentiation_requirements', []))}

    What top articles already cover:
    {content_strategy.get('existing_content_summary', 'N/A')}

    Your task: Create a blog outline that STANDS OUT by:
    1. Taking the unique angle identified above
    2. Filling the content gaps others missed
    3. Meeting all differentiation requirements
    4. Serving the target reader's specific problem

    Do NOT just rehash what existing articles already cover well.
    Focus on what's MISSING or WEAK in existing content.

    Output JSON as BlogPlan model.
    """

    # ... rest of planning logic
```

**Tests**:
- `tests/unit/test_content_landscape.py`:
  - Test ContentStrategy model validation
  - Test _select_top_urls (scoring logic)
  - Test _format_article_summaries
  - Mock LLM responses for article analysis and synthesis
- `tests/integration/test_content_landscape_integration.py`:
  - Real content fetching (top 5 URLs to limit API calls)
  - Real LLM analysis with Flash-Lite
  - Verify ContentStrategy generated with unique_angle and gaps

**Milestone**: Content landscape analyzed, unique angle identified, strategy saved to content_strategy.json

**Token Budget Impact**:
- Article analysis: ~10 LLM calls (Flash-Lite) - 1,000 tokens in, 200 out each = 10k in, 2k out
- Strategy synthesis: 1 LLM call (Flash-Lite) - 3,000 tokens in, 500 out
- **Total added**: ~11 calls, ~13k tokens in, ~2.5k tokens out
- **Updated pipeline total**: ~36 LLM calls, ~115k tokens in, ~27.5k tokens out (still within budget)

---

## PRE-FOUNDATION: Planning & Research (Phases 4-5)

**Status**: ✓ COMPLETE

These phases were implemented as part of the foundation:
- **Phase 4: Planning** - Generate blog outline with sections (planning_node)
- **Phase 5: Research** - Fetch and validate sources (research_node, validate_sources_node)

See IMPLEMENTATION_PLAN.md Phase 4 and Phase 5 for details.

---

## ROUND 1: Minimal End-to-End CLI (Slices 6.1-6.4)

**Goal**: Run `python -m src.agent start --title "..." --context "..."` and get final.md

---

### Slice 6.1: Minimal Section Writer

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/config.py` - Add STYLE_GUIDE, LLM_MODEL_FLASH
- `src/agent/nodes.py` - Add writer functions

**Add to config.py**:
```python
STYLE_GUIDE = """
Write in a direct, technical style for experienced engineers.
- Open with a clear problem statement, no warm-up.
- Be opinionated. Say "you need X" not "you might consider".
- Keep paragraphs short (2-4 sentences).
- Use bullet points only for listing items.
- Include specific tool names and real config examples.
- No fluff: "In today's world", "It's worth noting that".
- Address the reader as "you".
- Code: Python preferred, include imports, be runnable.
"""

LLM_MODEL_FLASH = "gemini-2.5-flash"  # For writing (higher quality)
```

**Add to nodes.py**:
```python
# Helper functions
def _build_writer_prompt(section, sources, previous_sections, style_guide) -> str:
    """Constructs writer prompt with sources and style guide."""

async def _write_section(section, sources, previous_text, key_manager) -> str:
    """Call Gemini Flash to write section content."""

# Node function
async def write_section_node(state: BlogAgentState) -> dict[str, Any]:
    """Writes ONE section per call (for graph loop)."""
```

**Behavior**:
- No critic, no refine - just write
- Save draft to `drafts/sections/{section_id}.md`
- Increment `current_section_index` for next iteration

**Tests**:
- `tests/unit/test_writing.py`: Prompt building, mock LLM response
- `tests/integration/test_writing_integration.py`: Write one section with real LLM

**Milestone**: Can write all sections (no quality checks yet)

---

### Slice 6.2: Basic Assembly

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/nodes.py` - Add assembly functions

**Add to nodes.py**:
```python
def _combine_sections(section_drafts: dict, plan: dict) -> str:
    """Concatenate sections with H1 title, H2 headers."""

async def final_assembly_node(state: BlogAgentState) -> dict[str, Any]:
    """Combines sections, saves final.md."""
```

**Behavior**:
- No final critic, no mermaid, no citations
- Just combine and save to `final.md`

**Tests**:
- `tests/unit/test_assembly.py`: Section combination, header hierarchy
- `tests/integration/test_assembly_integration.py`: Combines real sections

**Milestone**: Combined draft saved as final.md

---

### Slice 6.3: Graph + Routing

**Status**: ✓ COMPLETE

**Files to create**:
- `src/agent/graph.py` (NEW)

**Create graph.py**:
```python
from langgraph.graph import StateGraph, END
from .state import BlogAgentState
from .nodes import (
    topic_discovery_node, planning_node, research_node,
    validate_sources_node, write_section_node, final_assembly_node,
)

def section_router(state: BlogAgentState) -> str:
    """Route: 'write_next' if more sections, else 'all_complete'"""
    sections = state.get("plan", {}).get("sections", [])
    current_idx = state.get("current_section_index", 0)
    # Only count non-optional sections
    required = [s for s in sections if not s.get("optional")]
    if current_idx < len(required):
        return "write_next"
    return "all_complete"

def build_blog_agent_graph() -> StateGraph:
    """Build graph with section loop."""
    graph = StateGraph(BlogAgentState)

    # Add nodes
    graph.add_node("topic_discovery", topic_discovery_node)
    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("validate_sources", validate_sources_node)
    graph.add_node("write_section", write_section_node)
    graph.add_node("final_assembly", final_assembly_node)

    # Set entry
    graph.set_entry_point("topic_discovery")

    # Linear edges
    graph.add_edge("topic_discovery", "planning")
    graph.add_edge("planning", "research")
    graph.add_edge("research", "validate_sources")
    graph.add_edge("validate_sources", "write_section")

    # Section loop
    graph.add_conditional_edges(
        "write_section",
        section_router,
        {"write_next": "write_section", "all_complete": "final_assembly"}
    )

    # End after assembly (no human review yet)
    graph.add_edge("final_assembly", END)

    return graph.compile()
```

**Tests**:
- `tests/unit/test_graph.py`: Routing logic, graph structure
- `tests/integration/test_full_graph.py`: Full flow with real LLM

**Milestone**: Complete pipeline working (no critic, no human review)

---

### Slice 6.4: CLI Start Command

**Status**: ✓ COMPLETE

**Files to create**:
- `src/agent/__main__.py` (NEW)

**Create __main__.py**:
```python
import click
import asyncio
from .graph import build_blog_agent_graph
from .state import JobManager, Phase

@click.group()
def cli():
    """Blog Agent - AI-powered technical blog writer."""
    pass

@cli.command()
@click.option("--title", required=True, help="Blog title")
@click.option("--context", required=True, help="Context and notes")
@click.option("--length", type=click.Choice(["short", "medium", "long"]), default="medium")
def start(title, context, length):
    """Start a new blog generation job."""
    asyncio.run(_run_start(title, context, length))

async def _run_start(title, context, length):
    job_manager = JobManager()
    job_id = job_manager.create_job(title, context, length)

    initial_state = {
        "job_id": job_id,
        "title": title,
        "context": context,
        "target_length": length,
        "current_phase": Phase.TOPIC_DISCOVERY.value,
        "current_section_index": 0,
        "section_drafts": {},
    }

    graph = build_blog_agent_graph()
    result = await graph.ainvoke(initial_state)

    print(f"\n✓ Blog generated: ~/.blog_agent/jobs/{job_id}/final.md")

if __name__ == "__main__":
    cli()
```

**Tests**:
- `tests/unit/test_cli.py`: Argument parsing, validation
- `tests/integration/test_cli_integration.py`: End-to-end CLI run

**Milestone**:
```bash
python -m src.agent start --title "Semantic Caching" --context "Redis, GPTCache"
# Outputs: ~/.blog_agent/jobs/semantic-caching/final.md
```

---

## ROUND 2: Quality Layer (Slices 6.5-6.8)

**Goal**: Add section critic, refine loop, final critic, basic human approval

---

### Slice 6.5: Section Critic with Originality Check (15 Dimensions + Differentiation Flags)

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/state.py` - Expand CriticScore to 15 dimensions, add DifferentiationFlags
- `src/agent/nodes.py` - Add critic and originality check functions
- `src/agent/config.py` - Add ACTIONABILITY_REQUIREMENTS constant

**Add to state.py**:
```python
class CriticScore(BaseModel):
    """
    15 scoring dimensions (1-10 scale).

    Based on design.md + Google's "Creating Helpful Content" guidelines + Content Differentiation Strategy
    (https://developers.google.com/search/docs/fundamentals/creating-helpful-content)
    """
    # Original 8 dimensions
    technical_accuracy: int = Field(ge=1, le=10, description="Claims correct? No misleading statements?")
    completeness: int = Field(ge=1, le=10, description="Covers what title promises?")
    code_quality: int = Field(ge=1, le=10, description="Imports? Runnable? Well-explained?")
    clarity: int = Field(ge=1, le=10, description="Easy to follow? Terms explained?")
    voice: int = Field(ge=1, le=10, description="Matches style guide? No fluff? Opinionated?")
    originality: int = Field(ge=1, le=10, description="Adds value beyond sources? Not copy-paste?")
    length: int = Field(ge=1, le=10, description="Within 20% of target?")
    diagram_quality: int = Field(ge=1, le=10, default=10, description="(if mermaid) Correct? Clear?")

    # Google E-E-A-T quality dimensions (2)
    eeat_expertise: int = Field(
        ge=1, le=10,
        description="Demonstrates first-hand expertise and depth of knowledge? Shows experience with the topic?"
    )
    people_first_value: int = Field(
        ge=1, le=10,
        description="Serves readers directly (not just SEO)? Would you bookmark/share this? Leaves readers satisfied?"
    )

    # NEW: Content Differentiation Dimensions (5)
    unique_insights: int = Field(
        ge=1, le=10,
        description="Provides insights not found in existing content? Novel perspective? Not just rehashing?"
    )
    actionability: int = Field(
        ge=1, le=10,
        description="Clear next steps? High action-to-theory ratio? Copy-paste ready code? Tells reader what to DO?"
    )
    specificity: int = Field(
        ge=1, le=10,
        description="Concrete examples with real numbers? Specific tools/versions? Not vague claims?"
    )
    depth_appropriateness: int = Field(
        ge=1, le=10,
        description="Right depth for topic? Covers edge cases? Explains 'why' not just 'how'? Goes beyond surface?"
    )
    production_readiness: int = Field(
        ge=1, le=10,
        description="Code has error handling? Security considerations? Real-world concerns? (Note: code should be educational/conceptual, not necessarily production-tested)"
    )

class CriticIssue(BaseModel):
    """A single issue identified by the critic."""
    dimension: str
    location: str
    problem: str
    suggestion: str

class OriginalityFlag(BaseModel):
    """Flagged sentence from originality check."""
    sentence: str
    similar_to: str  # Source URL
    similarity: float  # 0.0 to 1.0

class GoogleQualityFlags(BaseModel):
    """
    Google quality red flags to check.
    From: https://developers.google.com/search/docs/fundamentals/creating-helpful-content
    """
    written_for_word_count: bool = Field(
        default=False,
        description="Is content artificially padded to hit word count?"
    )
    merely_summarizing_sources: bool = Field(
        default=False,
        description="Just summarizing others' work without substantial additions?"
    )
    lacks_first_hand_expertise: bool = Field(
        default=False,
        description="Doesn't demonstrate clear experience with the topic?"
    )
    surface_level_coverage: bool = Field(
        default=False,
        description="Doesn't go beyond surface-level to offer substantial depth?"
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Specific reasons for flagged items"
    )

class DifferentiationFlags(BaseModel):
    """
    NEW: Differentiation red flags to ensure content stands out.

    These flags detect generic, rehashed content that won't be bookmarked.
    Based on content landscape analysis and differentiation strategy.
    """
    rehashes_existing_content: bool = Field(
        default=False,
        description="Says same things as top articles without new angle? Just repeating what others wrote?"
    )
    generic_examples: bool = Field(
        default=False,
        description="Uses foo/bar examples instead of real scenarios? Generic code without context?"
    )
    missing_edge_cases: bool = Field(
        default=False,
        description="Doesn't cover gotchas, limitations, or when NOT to use? Only shows happy path?"
    )
    vague_claims: bool = Field(
        default=False,
        description="Says 'improves performance' without specific benchmarks? No concrete numbers?"
    )
    no_production_context: bool = Field(
        default=False,
        description="Code examples lack error handling, logging, monitoring? Not realistic?"
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Specific reasons for flagged items"
    )
```

**Add to nodes.py**:
```python
def _check_originality(content: str, sources: list[dict]) -> list[OriginalityFlag]:
    """
    Programmatic plagiarism check (NO LLM - uses difflib).

    From design.md STEP 3.2:
    1. Split content into sentences
    2. For each sentence, compare against all source chunks
    3. Flag if SequenceMatcher similarity > 0.7 (70%)
    4. Also check N-gram overlap (3-gram, 4-gram)
    5. Flag exact phrase matches > 8 words

    This runs BEFORE the critic to catch copy-paste issues.
    """
    from difflib import SequenceMatcher
    flags = []

    sentences = _split_into_sentences(content)

    for sentence in sentences:
        for source in sources:
            # Sentence-level similarity
            similarity = SequenceMatcher(None, sentence, source['content']).ratio()
            if similarity > 0.7:
                flags.append({
                    "sentence": sentence,
                    "similar_to": source['url'],
                    "similarity": similarity
                })

            # Check for exact phrase matches > 8 words
            # Check N-gram overlaps (3-gram, 4-gram)

    return flags

def _build_critic_prompt(section, content, originality_flags, content_strategy, target_words) -> str:
    """
    15 dimensions evaluation prompt + Google quality checks + Differentiation flags.

    Evaluates:
    1. Original 8 dimensions (technical_accuracy, completeness, code_quality, etc.)
    2. Google E-E-A-T (2): eeat_expertise, people_first_value
    3. NEW Differentiation (5): unique_insights, actionability, specificity, depth_appropriateness, production_readiness

    Google Quality Guidelines (https://developers.google.com/search/docs/fundamentals/creating-helpful-content):
    - Does content provide original information, research, or analysis?
    - Goes beyond surface-level to offer substantial depth?
    - Adds substantial value beyond just summarizing sources?
    - Demonstrates first-hand expertise and depth of knowledge?
    - Is this something you'd bookmark, share with a friend, or recommend?
    - Leaves readers satisfied they've learned enough?
    - NOT written primarily to hit word count targets
    - NOT merely summarizing others' work without additions
    - Shows genuine experience with the topic

    NEW Differentiation Criteria (from Content Strategy):
    - Provides unique insights not found in existing articles?
    - Clear actionable next steps (not just theory)?
    - Concrete examples with specific numbers, tools, versions?
    - Right depth: covers edge cases, explains 'why' not just 'how'?
    - Code includes error handling, security, real-world concerns? (educational/conceptual, not necessarily production-tested)
    - NOT rehashing what existing articles already cover?
    - NOT using generic foo/bar examples?
    - NOT making vague claims without benchmarks?
    - Covers when NOT to use, gotchas, limitations?

    Context from Content Landscape Analysis:
    - Unique Angle: {content_strategy.get('unique_angle')}
    - Gaps to Fill: {content_strategy.get('gaps_to_fill')}
    - Differentiation Requirements: {content_strategy.get('differentiation_requirements')}

    Includes originality flags from automated check for the critic to review.
    """

async def _critic_section(section, content, originality_flags, content_strategy, key_manager) -> SectionCriticResult:
    """
    Call Gemini Flash for critique with originality context and content strategy.

    Returns SectionCriticResult with:
    - 15 dimension scores (8 original + 2 E-E-A-T + 5 differentiation)
    - Google quality flags (written_for_word_count, merely_summarizing_sources, etc.)
    - NEW: DifferentiationFlags (rehashes_existing_content, generic_examples, etc.)
    - Issues list with suggestions
    - overall_pass (True if ALL 15 scores >= 8 AND no critical Google flags AND no differentiation flags)
    """
```

**Update SectionCriticResult model**:
```python
class SectionCriticResult(BaseModel):
    """Full critic evaluation of a section (ENHANCED with 15 dimensions + differentiation flags)."""
    scores: CriticScore  # Now 15 dimensions (8 original + 2 E-E-A-T + 5 differentiation)
    google_quality_flags: GoogleQualityFlags  # Google content quality red flags
    differentiation_flags: DifferentiationFlags  # NEW: Uniqueness and differentiation red flags
    overall_pass: bool = Field(
        description="True if ALL 15 scores >= 8 AND no critical Google quality flags AND no differentiation flags"
    )
    failure_type: str | None = Field(
        default=None,
        description="null | writing | research_gap | human"
    )
    issues: list[CriticIssue] = Field(default_factory=list)
    fact_check_needed: list[str] = Field(default_factory=list)
    missing_research: str | None = Field(default=None)
    praise: str = Field(default="", description="What's working well")
```

**Add ACTIONABILITY_REQUIREMENTS to config.py**:
```python
# From user requirements: strict actionability enforcement
ACTIONABILITY_REQUIREMENTS = """
CRITICAL: Every section MUST include (evaluated by actionability dimension):

1. **Concrete Example** with specific:
   - Tool versions (Python 3.11+, Redis 7.0+)
   - Actual numbers (reduced from 500ms to 50ms)
   - Real scenarios (e-commerce checkout, not generic "app")

2. **Complete Code** that:
   - Includes all imports
   - Is conceptually correct and educational
   - Shows realistic usage (not necessarily production-tested)
   - Has basic error handling (try/except where appropriate)

3. **Clear Next Steps**:
   - "Now you can..."
   - "Try this: [specific action]"
   - "To verify it works: [test command]"

4. **Edge Cases** (at least 2):
   - What breaks this?
   - When NOT to use this?
   - Common pitfalls?

5. **Performance Context** (if relevant):
   - How fast/slow is this?
   - What's the memory impact?
   - How does it scale?

FAIL actionability dimension if missing any of these.
"""
```

**Update write_section_node workflow**:
```python
# After write:
1. Run _check_originality(content, sources)  # Programmatic plagiarism check
2. Get content_strategy from state (from Slice 0.4)
3. Run _critic_section(section, content, originality_flags, content_strategy, key_manager)  # LLM with 15 dimensions + flags
4. Evaluate pass/fail:
   - ALL 15 scores >= 8 (8 original + 2 E-E-A-T + 5 differentiation)
   - NO critical Google quality flags (not merely_summarizing_sources, not surface_level_coverage, etc.)
   - NO differentiation flags (not rehashes_existing_content, not generic_examples, etc.)
```

**Tests**:
- `tests/unit/test_writing.py`:
  - Add originality check tests
  - Add critic tests with **15 dimensions** (8 original + 2 E-E-A-T + 5 differentiation)
  - Test Google quality flag detection
  - Test **DifferentiationFlags detection** (rehashes_existing_content, generic_examples, etc.)
  - Test overall_pass logic (all 15 scores + Google flags + differentiation flags)
  - Test ACTIONABILITY_REQUIREMENTS validation
- `tests/integration/test_writing_integration.py`:
  - Critic with real LLM evaluating Google guidelines AND differentiation criteria
  - Test sections that should fail Google quality checks
  - Test sections that should fail differentiation checks (generic examples, vague claims)
  - Test content_strategy integration in critic

**Milestone**: Originality check, Google-quality-aware, and differentiation-aware 15D critic run, saves to `feedback/`

---

### Slice 6.6: Section Refine Loop with Differentiation Flags + Retry Strategies

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/config.py` - Add retry constants and strategies, ACTIONABILITY_REQUIREMENTS
- `src/agent/nodes.py` - Add refine functions with differentiation flag handling
- `src/agent/tools.py` - Apply retry strategies to all external calls

**Add to config.py**:
```python
# Section-level retries
MAX_SECTION_RETRIES = 2
CRITIC_PASS_THRESHOLD = 8

# From design.md RETRY MANAGER - Comprehensive retry strategies
RETRY_STRATEGIES = {
    "gemini_api": {
        "max_retries": 3,
        "backoff_seconds": [1, 2, 4],  # Exponential
        "on_429": "switch_project",  # Handled by KeyManager
        "on_all_exhausted": "pause_notify_human",
        "on_500": "retry_same_project"
    },
    "web_search": {
        "max_retries": 2,
        "backoff_seconds": [2, 5],
        "on_rate_limit": "wait_60s_retry",
        "on_fail": "try_alternate_query"
    },
    "web_fetch": {
        "max_retries": 2,
        "timeout_seconds": 30,
        "on_fail": "skip_source",  # Don't fail whole job
        "on_timeout": "skip_source"
    },
    "mermaid_render": {
        "max_retries": 2,
        "fallback": "save_raw_mermaid"  # Human can render manually
    }
}
```

**Add to nodes.py**:
```python
def _build_refiner_prompt(section, content, issues, google_flags, differentiation_flags, content_strategy) -> str:
    """
    Fix specific issues prompt (ENHANCED with differentiation flags).

    Addresses:
    - Critic issues from 15 dimensions (8 original + 2 E-E-A-T + 5 differentiation)
    - Google quality flags if any:
      - written_for_word_count: Remove fluff, focus on value
      - merely_summarizing_sources: Add original analysis, insights, examples
      - lacks_first_hand_expertise: Add practical experience, lessons learned
      - surface_level_coverage: Go deeper, add nuance, edge cases

    - NEW: Differentiation flags if any:
      - rehashes_existing_content: Rewrite with unique angle from content_strategy
      - generic_examples: Replace foo/bar with real scenarios (specific tools, versions, numbers)
      - missing_edge_cases: Add "When NOT to use", gotchas, limitations
      - vague_claims: Add concrete benchmarks, specific measurements
      - no_production_context: Add error handling, logging, monitoring examples (educational/conceptual)

    Context from Content Strategy:
    - Unique Angle: {content_strategy.get('unique_angle')}
    - Differentiation Requirements: {content_strategy.get('differentiation_requirements')}

    Instructs LLM to rewrite addressing ALL issues while preserving strengths.
    Emphasizes ACTIONABILITY_REQUIREMENTS from config.py.
    """

async def _refine_section(section, content, issues, google_flags, differentiation_flags, content_strategy, key_manager) -> str:
    """
    Call Gemini Flash for refinement.

    Addresses:
    - Traditional critic issues
    - Google quality red flags
    - NEW: Differentiation red flags
    - Uses content_strategy context to ensure uniqueness
    """

def _handle_failure_type(failure_type: str, retry_count: int) -> str:
    """
    Decision gate based on critic's failure_type.

    From design.md STEP 3.4:
    - "writing": Refine section and retry (includes Google quality issues)
    - "research_gap": Re-research with new queries, then rewrite
    - "human": Request human input
    - retry_count >= 2: Force human help
    """
    if retry_count >= MAX_SECTION_RETRIES:
        return "human_help"
    if failure_type == "writing":
        return "refine"
    if failure_type == "research_gap":
        return "re_research"
    return "human_help"
```

**Update write_section_node** with full retry loop (ENHANCED with differentiation flags):
```python
retry_count = 0
content_strategy = state.get("content_strategy", {})  # From Slice 0.4

while retry_count <= MAX_SECTION_RETRIES:
    # Step 1: Write (with ACTIONABILITY_REQUIREMENTS in prompt)
    content = await _write_section(section, sources, previous_text, content_strategy, key_manager)

    # Step 2: Originality check (programmatic)
    originality_flags = _check_originality(content, sources)

    # Step 3: Critic (15 dimensions + Google quality + Differentiation checks)
    critic_result = await _critic_section(
        section,
        content,
        originality_flags,
        content_strategy,  # NEW: Pass content strategy for differentiation evaluation
        key_manager
    )

    # Check overall_pass: ALL 15 scores >= 8 AND no critical Google flags AND no differentiation flags
    if critic_result.overall_pass:
        break

    # Step 4: Decision gate
    action = _handle_failure_type(critic_result.failure_type, retry_count)

    if action == "refine":
        # Pass issues + Google flags + NEW: differentiation flags + content strategy
        content = await _refine_section(
            section,
            content,
            critic_result.issues,
            critic_result.google_quality_flags,
            critic_result.differentiation_flags,  # NEW: Include differentiation feedback
            content_strategy,  # NEW: Context for uniqueness
            key_manager
        )
    elif action == "re_research":
        new_sources = await _re_research_section(section, key_manager)
        sources.extend(new_sources)
    else:
        # human_help needed
        state["needs_human_help"] = True
        break

    retry_count += 1
```

**Tests**:
- `tests/unit/test_writing.py`:
  - Refiner with Google quality flags AND **differentiation flags**
  - Decision gate, max retries, all failure types
  - Test refiner addresses Google quality issues
  - Test refiner addresses **differentiation issues** (generic examples → real scenarios, vague claims → concrete benchmarks)
  - Test content_strategy integration in writer and refiner
- `tests/integration/test_writing_integration.py`:
  - Full write→originality→critic(**15 dims** + Google + Differentiation)→refine loop
  - Test sections with Google quality issues get refined
  - Test sections with **differentiation issues** get refined (rehashed content → unique angle)
  - Test ACTIONABILITY_REQUIREMENTS enforcement

**Milestone**: Sections refined until quality gate passed (**15 dimensions** + Google quality + Differentiation), with comprehensive retry handling

---

### Slice 6.7: Final Critic

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/state.py` - Add final critic models
- `src/agent/nodes.py` - Add final critic functions

**Add to state.py**:
```python
class FinalCriticScore(BaseModel):
    """7 whole-blog dimensions (1-10)."""
    coherence: int = Field(ge=1, le=10)
    voice_consistency: int = Field(ge=1, le=10)
    no_redundancy: int = Field(ge=1, le=10)
    narrative_arc: int = Field(ge=1, le=10)
    hook_effectiveness: int = Field(ge=1, le=10)
    conclusion_strength: int = Field(ge=1, le=10)
    overall_polish: int = Field(ge=1, le=10)

class TransitionFix(BaseModel):
    """Fix needed between sections."""
    between: list[str]
    issue: str
    suggestion: str

class FinalCriticResult(BaseModel):
    """Full final critic evaluation."""
    scores: FinalCriticScore
    overall_pass: bool
    transition_fixes: list[TransitionFix] = Field(default_factory=list)
    reading_time_minutes: int
    word_count: int
```

**Add to nodes.py**:
```python
def _build_final_critic_prompt(draft: str) -> str:
    """7 dimensions for whole blog."""

async def _final_critic(draft, key_manager) -> FinalCriticResult:
    """Call Gemini Flash for final critique."""

async def _apply_transition_fixes(draft, fixes, key_manager) -> str:
    """Refine transitions between sections."""
```

**Update final_assembly_node** with critic loop (max 2 iterations)

**Tests**:
- `tests/unit/test_assembly.py`: Final critic prompt, scoring
- `tests/integration/test_assembly_integration.py`: Assembly with final critic

**Milestone**: Whole blog is critiqued and transitions are fixed

---

### Slice 6.8: Basic Human Review

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/nodes.py` - Add human_review_node
- `src/agent/graph.py` - Add review routing

**Add to nodes.py**:
```python
from rich.console import Console
from rich.prompt import Confirm

async def human_review_node(state: BlogAgentState) -> dict[str, Any]:
    """Simple human review with Rich prompt."""
    console = Console()
    job_id = state.get("job_id", "")

    console.print(f"\n[bold green]✓ Blog Complete[/bold green]")
    console.print(f"Job: {job_id}")
    console.print(f"Output: ~/.blog_agent/jobs/{job_id}/final.md\n")

    approved = Confirm.ask("Approve and finalize?", default=True)

    if approved:
        return {"human_review_decision": "approve", "current_phase": Phase.DONE.value}
    else:
        return {"human_review_decision": "quit"}
```

**Update graph.py**:
```python
def review_router(state: BlogAgentState) -> str:
    decision = state.get("human_review_decision", "")
    if decision == "approve":
        return "approved"
    return "quit"

# In build_blog_agent_graph():
graph.add_node("human_review", human_review_node)
graph.add_edge("final_assembly", "human_review")
graph.add_conditional_edges(
    "human_review",
    review_router,
    {"approved": END, "quit": END}
)
```

**Tests**:
- `tests/unit/test_graph.py`: Review router logic
- Manual testing for interactive flow

**Milestone**: User can approve or quit after generation

---

## ROUND 3: Full Human Experience (Slices 6.9-6.11)

**Goal**: Title suggestions, SEO, Rich UI

---

### Slice 6.9: Title Suggestions + SEO with Interactive Flow

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/state.py` - Add SEO models
- `src/agent/nodes.py` - Add title/SEO generation

**Add to state.py**:
```python
class SEOMetadata(BaseModel):
    meta_title: str = Field(description="SEO-optimized title (50-60 chars)")
    meta_description: str = Field(description="Search results description (150-160 chars)")
    excerpt: str = Field(description="Short summary for blog listings (~200 chars)")
    focus_keyword: str = Field(description="Primary SEO keyword")
    secondary_keywords: list[str] = Field(default_factory=list, description="2-3 secondary keywords")
    og_title: str = Field(description="Open Graph title for social sharing")
    og_description: str = Field(description="Open Graph description for social sharing")

class TitleSuggestion(BaseModel):
    title: str = Field(description="Engaging blog title (50-70 chars)")
    style: str = Field(description="Style: how-to | listicle | deep-dive | comparison | problem-solution")
    hook: str = Field(description="What makes this title compelling")
    seo_score: int = Field(ge=1, le=10, description="SEO effectiveness")
    engagement_score: int = Field(ge=1, le=10, description="Click-worthiness")

class TitleSuggestions(BaseModel):
    suggestions: list[TitleSuggestion] = Field(min_length=5, max_length=5)
    recommended_index: int = Field(ge=0, le=4, description="Index of recommended title")
    recommendation_reason: str = Field(description="Why this title is recommended")
```

**Add to nodes.py**:
```python
async def generate_title_suggestions(final_draft: str, original_title: str, key_manager) -> TitleSuggestions:
    """
    Generate 5 title options in different styles based on completed blog content.

    From IMPLEMENTATION_PLAN.md Section 8.8:
    - Style 1: How-to / Tutorial (e.g., "How to Build X with Y")
    - Style 2: Problem-Solution (e.g., "Solving X: A Practical Guide to Y")
    - Style 3: Deep-dive / Technical (e.g., "Understanding X: From Theory to Implementation")
    - Style 4: Listicle (e.g., "5 Ways to Improve X with Y")
    - Style 5: Comparison/Analysis (e.g., "X vs Y: Which One Should You Choose?")

    Each title includes:
    - The title (50-70 characters, SEO-friendly, includes main keyword)
    - Style type
    - Hook: What makes it compelling
    - SEO score (1-10): keyword relevance, search intent match, click potential
    - Engagement score (1-10): curiosity, value proposition, specificity

    Recommends the BEST title considering:
    - Balance of SEO (searchability) and engagement (click-worthiness)
    - Accurately represents the content
    - Clear value proposition for the reader
    - Appropriate for technical developer audience
    """

async def generate_seo_metadata(final_draft: str, selected_title: str, key_manager) -> SEOMetadata:
    """
    Generate complete SEO metadata after title selection.

    Generates:
    1. meta_title: SEO-optimized title (50-60 chars, include primary keyword)
    2. meta_description: For search results (150-160 chars, compelling, includes keyword)
    3. excerpt: Short summary for blog listings (1-2 sentences, ~200 chars, enticing)
    4. focus_keyword: Primary SEO keyword (the main search term)
    5. secondary_keywords: 2-3 related keywords
    6. og_title: Open Graph title for social sharing (can be slightly different from meta_title)
    7. og_description: Social sharing description (more casual, compelling for clicks)
    """
```

**Interactive Flow in human_review_node**:
```python
async def human_review_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 5: Human Review Node with Title Selection & SEO.
    """
    ui = BlogAgentUI(state["job_id"], state.get("flags", {}))

    # Step 1: Generate and display 5 title suggestions
    title_suggestions = await generate_title_suggestions(
        state["final_draft"],
        state["title"],
        key_manager
    )

    # Display title options with:
    # - All 5 suggestions with style, hook, scores
    # - Recommended title highlighted
    # - Options: [1-5], [k]eep original, [c]ustom
    selected_title = ui.show_title_suggestions(title_suggestions, state["title"])

    # Step 2: Generate SEO metadata based on selected title
    seo_metadata = await generate_seo_metadata(
        state["final_draft"],
        selected_title,
        key_manager
    )

    # Display SEO metadata with options:
    # - [a]ccept  [e]dit meta description  [r]egenerate  [s]kip SEO
    seo_metadata = ui.show_seo_metadata(seo_metadata)

    # Update state with selections
    state["title"] = selected_title
    state["metadata"]["seo"] = seo_metadata

    # Step 3: Display final review UI
    decision = ui.show_final_review(state)

    # ... rest of approval logic
    return state
```

**Tests**:
- `tests/unit/test_title_seo.py`:
  - Test generates exactly 5 title suggestions
  - Test each suggestion has all required fields
  - Test recommended_index is valid (0-4)
  - Test styles are diverse (no duplicates)
  - Test SEO metadata has all required fields
  - Test meta_title length (50-60 chars)
  - Test meta_description length (150-160 chars)
- `tests/integration/test_title_seo_integration.py`:
  - Title generation with real LLM
  - SEO metadata generation with real LLM
  - Interactive selection flow

**Milestone**: Title selection and SEO metadata saved to metadata.json with interactive flow

---

## ROUND 4: Polish Features (Slices 6.12-6.13)

**Goal**: Mermaid, citations, fact checking

---

### Slice 6.12: Mermaid + Citations

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/tools.py` - Add render_mermaid (if not exists)
- `src/agent/nodes.py` - Add mermaid/citation functions

**Add to nodes.py**:
```python
async def _render_all_mermaid(draft: str, job_dir: Path) -> tuple[str, int]:
    """Find and render all mermaid blocks via kroki.io."""

def _add_citations(draft: str, validated_sources: dict) -> str:
    """Add References section with all sources used."""
```

**Update final_assembly_node**:
1. Render mermaid diagrams → `images/`
2. Replace mermaid blocks with image references
3. Add References section (unless --no-citations)

**Tests**:
- `tests/unit/test_assembly.py`: Mermaid regex, citation formatting
- `tests/integration/test_assembly_integration.py`: Real mermaid rendering

**Milestone**: Diagrams rendered, citations added

---

### Slice 6.13: Fact Checking

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/state.py` - Add ClaimVerification model
- `src/agent/nodes.py` - Add fact check functions

**Add to state.py**:
```python
class ClaimVerification(BaseModel):
    claim: str
    status: str  # verified | unverified | conflicting
    sources: list[str] = Field(default_factory=list)
    reasoning: str
```

**Add to nodes.py**:
```python
async def _verify_claim(claim: str, key_manager) -> ClaimVerification:
    """Search + LLM evaluation for single claim."""

async def verify_all_claims(claims: list[str], key_manager) -> list[ClaimVerification]:
    """Concurrent verification of all claims."""

def _generate_fact_check_md(verifications: list, job_dir: Path):
    """Generate fact_check.md report."""
```

**Integrate into final_assembly_node**

**Tests**:
- `tests/unit/test_fact_check.py`: Verification logic
- `tests/integration/test_fact_check_integration.py`: Real fact checking

**Milestone**: fact_check.md generated with verified/unverified claims

---

## Implementation Checklist

| Slice | Description | Unit Tests | Integration Tests | Status |
|-------|-------------|------------|-------------------|--------|
| **Foundation** | **Core Infrastructure** | | | |
| 0.1 | Tools & Utilities | test_tools.py | test_tools_integration.py | ✓ |
| 0.2 | State & Checkpoints | test_checkpoints.py, test_state.py | - | ✓ |
| 0.3 | Topic Discovery (Phase 0.5) | test_topic_discovery.py | test_topic_discovery_integration.py | ✓ |
| **0.4** | **Content Landscape Analysis (NEW)** | **test_content_landscape.py** | **test_content_landscape_integration.py** | **✓** |
| **Pre-Foundation** | **Planning & Research (Phases 4-5)** | | | |
| - | Planning Node (enhanced with ContentStrategy) | test_planning.py | test_planning_integration.py | ✓ |
| - | Research + Validation Nodes | test_research.py, test_validation.py | test_research_integration.py | ✓ |
| **Round 1** | **Minimal End-to-End CLI** | | | |
| 6.1 | Minimal Section Writer | test_writing.py | test_writing_integration.py | ✓ |
| 6.2 | Basic Assembly | test_assembly.py | test_assembly_integration.py | ✓ |
| 6.3 | Graph + Routing | test_graph.py | test_full_graph.py | ✓ |
| 6.4 | CLI Start Command | test_cli.py | test_cli_integration.py | ✓ |
| **Round 2** | **Quality Layer (15D Critic + Differentiation + Refine)** | | | |
| 6.5 | Section Critic (**15D**) + Originality + Google + **Differentiation** | test_writing.py | test_writing_integration.py | ✓ |
| 6.6 | Section Refine Loop + Google + **Differentiation Fixes** | test_writing.py | test_writing_integration.py | ✓ |
| 6.7 | Final Critic | test_assembly.py | test_assembly_integration.py | ○ |
| 6.8 | Basic Human Review | test_graph.py | Manual | ○ |
| **Round 3** | **Title + SEO Generation** | | | |
| 6.9 | Title + SEO with Interactive Flow | test_title_seo.py | test_title_seo_integration.py | ○ |
| **Round 4** | **Polish Features** | | | |
| 6.12 | Mermaid + Citations | test_assembly.py | test_assembly_integration.py | ○ |
| 6.13 | Fact Checking | test_fact_check.py | test_fact_check_integration.py | ○ |
| **Round 5** | **Web UI with Reflex** | | | |
| 7.1 | Reflex Setup + Basic Dashboard | Manual | - | ○ |
| 7.2 | New Job Form | Manual | - | ○ |
| 7.3 | Job Detail Page (Static) | Manual | - | ○ |
| 7.4 | Pipeline Runner (Background) | test_pipeline_runner.py | test_pipeline_integration.py | ○ |
| 7.5 | Real-time Progress + API Quota | Manual | - | ○ |
| 7.6 | Section Review + Originality Flags | Manual | - | ○ |
| 7.7 | Final Review + Title + SEO | Manual | - | ○ |
| 7.8 | Job Output Viewer | Manual | - | ○ |
| 7.9 | Resume Failed Jobs | Manual | - | ○ |
| 7.10 | Version Comparison & Error Display | Manual | - | ○ |

---

## Quick Start Commands

**After Slice 6.4** (first CLI):
```bash
python -m src.agent start --title "Semantic Caching for LLMs" --context "Redis, GPTCache"
# Outputs: ~/.blog_agent/jobs/semantic-caching-for-llms/final.md
```

**After Slice 6.8** (with human review):
```bash
python -m src.agent start --title "..." --context "..."
# Prompts: "Approve and finalize? [Y/n]"
```

**After Round 5 (Web UI)**:
```bash
cd web_ui
reflex run
# Visit http://localhost:3000
```

---

## ROUND 5: Web UI with Reflex (Slices 7.1-7.10)

**Goal**: Replace CLI with web interface for real-time monitoring, interactive reviews, job management

---

### Slice 7.1: Reflex Setup + Basic Dashboard

**Status**: ○ PENDING

**Files to create**:
- `web_ui/app.py` - Main Reflex app
- `web_ui/pages/dashboard.py` - Dashboard page
- `web_ui/state.py` - Reflex state classes

**Implementation**:
1. Install Reflex: `pip install reflex python-multipart`
2. Create `web_ui/` directory structure
3. Initialize Reflex app with basic routing
4. Create dashboard page that reads jobs from `~/.blog_agent/jobs/`
5. Display job list in table (job_id, title, status, created_at)

**Tests**:
- Manual: Run `cd web_ui && reflex run`, verify dashboard shows at localhost:3000
- Manual: Verify existing jobs appear in table

**Milestone**: Basic web UI running, shows job list

---

### Slice 7.2: New Job Form

**Status**: ○ PENDING

**Files to create**:
- `web_ui/pages/new_job.py` - New job form page

**Implementation**:
1. Create form with inputs: title, context (textarea), length dropdown, checkboxes for options
2. Add form validation (required fields, max lengths)
3. On submit: Create job directory via JobManager, initialize state.json
4. Redirect to dashboard after job creation
5. Display success toast notification

**Tests**:
- Manual: Fill form, submit, verify job created in `~/.blog_agent/jobs/`
- Manual: Test validation (empty title, etc.)

**Milestone**: Can create new jobs from web form

---

### Slice 7.3: Job Detail Page (Static)

**Status**: ○ PENDING

**Files to create**:
- `web_ui/pages/job_detail.py` - Job detail page
- `web_ui/services/job_manager.py` - Wrapper for job operations

**Implementation**:
1. Create job detail page that loads job state from disk
2. Display: job_id, title, status, current_phase
3. Show plan sections with status indicators (✓ ○ ×)
4. Add "Start" button for new jobs, "Resume" for incomplete
5. Navigation back to dashboard

**Tests**:
- Manual: Click job from dashboard, view details
- Manual: Verify correct phase and section status display

**Milestone**: Can view job details (not yet real-time)

---

### Slice 7.4: Pipeline Runner (Background)

**Status**: ○ PENDING

**Files to create**:
- `web_ui/services/pipeline_runner.py` - Run LangGraph in background

**Implementation**:
1. Create `run_pipeline_in_background(job_id)` function
2. Import and call `build_blog_agent_graph()` from `src/agent/graph.py`
3. Run graph with `asyncio.create_task()` (non-blocking)
4. Save checkpoints after each node completion
5. Handle errors, update job status on completion/failure

**Tests**:
- Unit: Mock LangGraph, test pipeline runner starts/completes
- Integration: Run real pipeline in background, verify completion

**Milestone**: Pipeline runs in background from web UI

---

### Slice 7.5: Real-time Progress Updates with API Quota Display

**Status**: ○ PENDING

**Files to modify**:
- `web_ui/pages/job_detail.py` - Add real-time polling
- `web_ui/services/pipeline_runner.py` - Send progress events

**Files to create**:
- `web_ui/components/progress_panel.py` - Progress display component
- `web_ui/components/activity_log.py` - Live activity stream

**Implementation** (from design.md Terminal UI lines 982-1013):

**1. Header Panel**:
- Job ID, tokens used, LLM calls made
- **PROMINENT API quota display**: "Project 2/4 (178/250 RPD remaining)"
- Update quota in real-time as requests are made

**2. Sections Progress List** (checkboxes with status):
- ✓ Completed sections with score [9.2] and word count (312 words)
- ◉ Current section being processed
- ○ Pending sections
- Show flags: [has diagram], [needs code]

**3. Current Action Panel**:
- What's happening now (e.g., "Writing Implementation Deep-dive")
- Progress bar for LLM generation (60% generating...)
- Target word count and flags (Code: yes, Diagram: yes)
- Using N validated sources

**4. Recent Activity Log**:
- Last 4-5 actions with timestamps (10:42:15)
- Color-coded by type (success, warning, info)
- Examples:
  - "Section 'Why Semantic Caching' passed critic (score: 9.0)"
  - "Generated mermaid diagram for architecture"
  - "Fetched 4 sources for 'Why Semantic Caching'"

**5. Polling Strategy**:
- Add Reflex background task that polls job state every 2 seconds
- Or: Implement WebSocket events from pipeline_runner to frontend
- Auto-refresh when job status changes

**Tests**:
- Manual: Start job, watch real-time updates on detail page
- Manual: Verify progress bars and activity log update live
- Manual: Verify API quota display updates correctly

**Milestone**: Real-time monitoring with comprehensive progress UI

---

### Slice 7.6: Section Review with Originality Flags and Critic Details

**Status**: ○ PENDING

**Files to create**:
- `web_ui/components/section_review.py` - Section review modal/page
- `web_ui/components/markdown_preview.py` - Markdown renderer with syntax highlighting

**Implementation** (from design.md Terminal UI lines 1019-1063):

**1. Section Review Header**:
- Section title (e.g., "Implementation Deep-dive")
- Overall score: 8.7 ✓ PASSED (or NEEDS REVISION)

**2. Scores Panel** (15 dimensions):
- Display as badges or progress bars (1-10 scale):
  - **Core Quality (8)**: Technical: 9 | Complete: 8 | Code: 9 | Clarity: 9 | Voice: 8 | Originality: 9 | Length: 9 | Diagram: N/A
  - **Google E-E-A-T (2)**: Expertise: 9 | People-First: 8
  - **NEW Differentiation (5)**: Unique Insights: 9 | Actionability: 8 | Specificity: 9 | Depth: 9 | Production: 8

**3. Content Preview**:
- Rendered markdown with syntax highlighting for code blocks
- Show first 15-20 lines with "[showing 15/52 lines]" indicator
- Button to view full section

**4. Critic Notes**:
- ✓ Praise: "Strong opening, runnable code example"
- → Minor Issues: "Consider adding error handling to the code snippet" (with location)
- ⚠ Originality Flags: If any sentences flagged by automated check, show them here
  - "Sentence X is 78% similar to source redis.io/docs"

**4b. Google Quality Flags**:
- Display if any Google quality red flags detected:
  - ⚠ Merely Summarizing Sources: "Content summarizes sources without substantial additions. Add original analysis, examples, or insights."
  - ⚠ Lacks First-Hand Expertise: "Doesn't demonstrate practical experience. Add lessons learned or real-world examples."
  - ⚠ Surface-Level Coverage: "Doesn't go deep enough. Add nuance, edge cases, or advanced considerations."
  - ⚠ Written for Word Count: "Content appears padded. Remove fluff, focus on value."

**4c. Differentiation Flags** (NEW):
- Display if any differentiation red flags detected:
  - ⚠ Rehashes Existing Content: "Says same things as top articles. Apply unique angle: [content_strategy.unique_angle]."
  - ⚠ Generic Examples: "Uses foo/bar examples. Replace with real scenarios (specific tools, versions, numbers)."
  - ⚠ Missing Edge Cases: "Only shows happy path. Add: when NOT to use, gotchas, limitations."
  - ⚠ Vague Claims: "Makes claims without evidence. Add concrete benchmarks, specific measurements."
  - ⚠ No Production Context: "Code lacks real-world concerns. Add error handling, logging, monitoring (educational examples)."

**5. Fact-Check Items**:
- List claims that need human verification:
  - "HNSW provides O(log n) search - verify this claim"

**6. Action Buttons**:
- [Enter] approve & continue
- [v]iew full section
- [e]dit in textarea
- [f]eedback & rewrite (enter guidance, goes back to refine)
- [s]kip section
- [q]uit & save state

**7. Workflow**:
1. Detect when `write_section_node` completes and state has `needs_review` flag
2. Display modal/page with all above elements
3. On approve: Update state with `section_approved`, trigger pipeline to continue
4. On feedback: Save feedback to `human_inputs/section_{id}_feedback.txt`, re-run refine
5. On skip: Mark section as skipped, continue to next

**Tests**:
- Manual: Run job with `--review-sections`, verify modal appears with all elements
- Manual: Verify **15 dimension scores** displayed correctly (8 core + 2 E-E-A-T + 5 differentiation)
- Manual: Approve section, verify pipeline continues
- Manual: Provide feedback, verify section is refined
- Manual: Verify originality flags are displayed correctly
- Manual: Verify Google quality flags displayed if present
- Manual: Verify **differentiation flags** displayed if present (rehashed content, generic examples, etc.)

**Milestone**: Comprehensive section review with **15D scores**, originality flags, Google quality flags, **differentiation flags**, and fact-check items

---

### Slice 7.7: Final Review with Title Selection and SEO

**Status**: ○ PENDING

**Files to create**:
- `web_ui/components/final_review.py` - Final review page
- `web_ui/components/title_selector.py` - Title selection UI
- `web_ui/components/seo_editor.py` - SEO metadata editor

**Implementation** (from design.md lines 939-974 + IMPLEMENTATION_PLAN.md lines 762-844):

**1. Title Suggestions Panel** (display first, before final review):
- Header: "TITLE SUGGESTIONS - Based on your completed blog content, here are 5 title options:"
- Display all 5 suggestions in cards/list:
  - **[HOW-TO]** "How to Build Semantic Caching for LLM APIs with Redis"
    - Hook: Direct, actionable, mentions specific tool
    - SEO: 8/10 | Engagement: 7/10
  - **[PROBLEM-SOLUTION]** "Cutting LLM API Costs by 60%: A Guide to Semantic Caching"
    - Hook: Leads with concrete benefit
    - SEO: 7/10 | Engagement: 9/10
  - ... (3 more options)
- ★ RECOMMENDED: #3 with reason (highlighted)
- Selection UI: Radio buttons [1-5], "Keep Original", "Enter Custom"

**2. SEO Metadata Panel** (after title selection):
- Display generated SEO metadata:
  - Meta Title (57 chars): "Semantic Caching for LLMs: Vector Search to Production | Dev Guide"
  - Meta Description (158 chars): "Learn how to implement semantic caching..."
  - Excerpt (195 chars): "Semantic caching stores query embeddings..."
  - Focus Keyword: "semantic caching LLM"
  - Secondary Keywords: ["vector search", "Redis cache", "LLM optimization"]
  - Open Graph: Title and Description for social sharing
- Action buttons:
  - [a]ccept
  - [e]dit meta description (open textarea)
  - [r]egenerate (call LLM again)
  - [s]kip SEO

**3. Final Review Panel**:
- Header: "✓ BLOG COMPLETE"
- Job stats: Title, Words: 1,487 | Reading time: 6 min | Sections: 6
- LLM usage: LLM calls: 24 | Tokens: 101k | Duration: 22 min

**4. Quality Scores Panel** (7 dimensions for whole blog):
- Coherence: 9 | Voice: 9 | No Redundancy: 9 | Narrative Arc: 8 | Hook: 9 | Conclusion: 9 | Overall Polish: 9

**5. Fact Check Warning** (if any):
- ⚠ FACT CHECK REQUIRED (see fact_check.md):
  - "HNSW provides O(log n) search complexity"
  - "GPTCache reduces latency by 80%"

**6. Files List**:
- → final.md (ready to publish)
- → images/diagram_0.png
- → fact_check.md
- → metadata.json

**7. Action Buttons**:
- [a]pprove - finalize and mark complete
- [v]iew final.md in modal/new tab
- [f]act_check - view fact_check.md
- [e]dit - open in textarea editor
- [r]equest changes - enter feedback, re-run assembly
- [q]uit - save state for later resume

**8. Workflow**:
1. Generate 5 title suggestions (LLM call)
2. User selects title
3. Generate SEO metadata based on selected title (LLM call)
4. User reviews/edits SEO metadata
5. Display final review with all panels
6. On approve: Save final.md with selected title and metadata, mark job complete

**Tests**:
- Manual: Complete job to final review, verify all panels display
- Manual: Select title, verify SEO generated
- Manual: Edit SEO metadata, verify saves
- Manual: Approve, verify final.md saved with correct title and metadata
- Manual: Verify fact-check warnings appear correctly

**Milestone**: Comprehensive final review with title selection, SEO, and quality scores

---

### Slice 7.8: Job Output Viewer

**Status**: ○ PENDING

**Files to create**:
- `web_ui/pages/job_output.py` - Output viewer page

**Implementation**:
1. Create page to view completed job outputs
2. Display final.md (rendered markdown)
3. Show metadata.json (formatted JSON)
4. Display images inline (if any mermaid diagrams)
5. Add download button for final.md

**Tests**:
- Manual: Click completed job, view output
- Manual: Download final.md, verify correct file

**Milestone**: Can view and download job outputs

---

### Slice 7.9: Resume Failed Jobs

**Status**: ○ PENDING

**Files to modify**:
- `web_ui/services/pipeline_runner.py` - Add resume capability
- `web_ui/pages/dashboard.py` - Add "Resume" button for failed jobs

**Implementation**:
1. Add `resume_pipeline(job_id)` function that loads checkpoint
2. Continue graph execution from last completed node
3. Display "Resume" button on dashboard for failed/incomplete jobs
4. Handle API quota errors gracefully (show error message)
5. Allow resuming next day when quota resets

**Tests**:
- Manual: Stop job mid-execution, resume, verify continues correctly
- Manual: Simulate API quota error, verify can resume

**Milestone**: Can resume interrupted/failed jobs

---

### Slice 7.10: Version Comparison & Error Display

**Status**: ○ PENDING

**Files to create**:
- `web_ui/components/version_diff.py` - Diff viewer
- `web_ui/components/error_panel.py` - Error display

**Implementation**:
1. Load all draft versions (v1.md, v2.md, final.md) for a job
2. Add dropdown to select two versions to compare
3. Display side-by-side diff (use `difflib`, highlight changes)
4. Create error panel component for displaying errors
5. Show errors on job detail page with stack trace (collapsible)

**Tests**:
- Manual: Compare v1 vs v2, verify diff display correct
- Manual: Trigger error, verify error panel shows details

**Milestone**: Version comparison and error handling complete

---

## Notes

- Each slice should be completable in ~30-60 minutes
- Run tests after each slice before moving to next
- Commit after each slice passes tests
- Update status (○ → ◉ → ✓) as you progress
