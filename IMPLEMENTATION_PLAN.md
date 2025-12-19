# Blog Agent Implementation Plan

## Overview
Build a multi-phase AI blog writing agent using **Python + LangGraph** with **Google Gemini** as the LLM provider. The system follows the design in `design.md` with 5 phases for generating technical blog posts.

---

## Module Structure (5 Files under `src/agent/`)

| File | Purpose |
|------|---------|
| `state.py` | State schema (TypedDict), Pydantic models for LLM outputs, Phase enum |
| `graph.py` | LangGraph StateGraph definition, nodes registration, conditional edges |
| `nodes.py` | All phase implementations (topic_discovery, planning, research, write, etc.) |
| `tools.py` | DuckDuckGo search, trafilatura fetch, originality checker, mermaid renderer |
| `ui.py` | Rich-based Terminal UI with progress display, human review interface |

Additional files:
- `src/agent/__init__.py` - Module exports
- `tests/unit/` - Fast unit tests with mocked dependencies
- `tests/integration/` - Real API tests (needs API keys)

---

## Incremental Build Order

### Phase 1: Foundation (Setup + Tools)
**Goal**: Project structure, dependencies, and utility functions

1. Create `src/agent/` directory structure
2. Install dependencies: `langgraph`, `duckduckgo-search`, `trafilatura`, `httpx`, `rich`
3. Implement `tools.py`:
   - `search_duckduckgo(query, max_results)` - Web search via DuckDuckGo
   - `fetch_url_content(url)` - Content extraction via trafilatura
   - `chunk_content(text, max_tokens)` - Split long articles
   - `check_originality(content, sources)` - Plagiarism check with difflib
   - `render_mermaid(code, output_path)` - Diagram rendering via kroki.io

**Integration Test**: `test_tools_integration.py`
- Test DuckDuckGo returns results
- Test trafilatura extracts content from a real URL
- Test mermaid rendering produces PNG

---

### Phase 2: State + Checkpoints
**Goal**: Define data flow and persistence

1. Implement `state.py`:
   - `Phase` enum (topic_discovery, planning, researching, etc.)
   - `BlogAgentState` TypedDict (all state fields)
   - Pydantic models: `DiscoveryQueries`, `BlogPlan`, `PlanSection`, `SectionDraft`, `CriticResult`

2. Implement checkpoint functions (in `nodes.py` or separate):
   - `save_state(job_id, state)` - Save to `~/.blog_agent/jobs/{job_id}/`
   - `load_state(job_id)` - Load for resume
   - `create_job(title, context)` - Initialize job directory
   - **Job ID format**: Slugified topic name (e.g., "semantic-caching-for-llm-applications")

**Unit Test**: `test_checkpoints.py`
- Test create job creates correct directory structure
- Test save/load round-trip preserves state

---

### Phase 3: Topic Discovery (Phase 0.5) - COMPLETED
**Goal**: Generate search queries and gather topic context

1. Implement `topic_discovery_node(state)` in `nodes.py`:
   - Call Gemini Flash-Lite to generate 3-5 search queries
   - Execute DuckDuckGo search for each query
   - Deduplicate results, keep top 15-20 snippets
   - Return updated state with `topic_context`

**Integration Test**: `test_topic_discovery.py`
- Test generates valid queries (mocked LLM or real API)
- Test aggregates search results correctly

**Implementation Notes**:
- Model used: `gemini-2.5-flash-lite`
- Uses `langchain-google-genai` with structured output
- Concurrent search execution with `asyncio.gather()`
- API key rotation on 429 errors via KeyManager

---

### Phase 4: Planning (Phase 1) - COMPLETED
**Goal**: Generate blog outline with sections

1. Implement `planning_node(state)` in `nodes.py`:
   - Build prompt with topic context snippets
   - Call Gemini Flash-Lite for structured BlogPlan output
   - Parse sections with: id, title, role, search_queries, target_words

**Integration Test**: `test_planning.py`
- Test generates plan with 5-6 sections
- Test each section has required fields

**Implementation Notes**:
- Model used: `gemini-2.5-flash-lite`
- Uses `langchain-google-genai` with structured output (BlogPlan Pydantic model)
- Helper functions: `_format_topic_context_snippets`, `_build_planning_prompt`, `_generate_blog_plan`
- TARGET_WORDS_MAP: {"short": 800, "medium": 1500, "long": 2500}
- API key rotation on 429 errors via KeyManager
- 26 unit tests + 7 integration tests all passing

---

### Phase 5: Research (Phase 2 + 2.5) - COMPLETED
**Goal**: Fetch and validate sources for each section

1. Implement `research_node(state)`:
   - For each section, search using its queries
   - Fetch content from top URLs
   - Cache in state.research_cache

2. Implement `validate_sources_node(state)`:
   - Call Gemini Flash-Lite to filter sources
   - Keep sources where `use=true`
   - Ensure min 4 sources per section

**Integration Test**: `test_research.py`
- Test fetches content from multiple URLs
- Test validation filters out low-quality sources

**Implementation Notes**:
- Model used: `gemini-2.5-flash-lite` for validation
- Helper functions: `_hash_url`, `_research_section`, `_build_validation_prompt`, `_validate_section_sources`
- Research cache keyed by MD5 hash of URL (12 chars)
- Validates all sections including optional deep_dive sections
- Sources include quality (high/medium/low) and reason fields
- API key rotation on 429 errors via KeyManager
- 34 unit tests (research + validation) + 8 integration tests all passing

---

### Phase 6: Writing Loop (Phase 3)
**Goal**: Write, critique, and refine each section

#### 6.1 New Pydantic Models (add to state.py)

```python
class SectionWriteResult(BaseModel):
    """Output from section writer LLM."""
    content: str = Field(description="Markdown content for the section")
    sources_used: list[str] = Field(default_factory=list, description="URLs of sources used")
    claims_to_verify: list[str] = Field(default_factory=list, description="Claims needing fact-check")

class CriticIssue(BaseModel):
    """A single issue identified by the critic."""
    dimension: str = Field(description="Which scoring dimension this affects")
    location: str = Field(description="Where in the section (paragraph, line)")
    problem: str = Field(description="Description of the issue")
    suggestion: str = Field(description="How to fix it")

class SectionCriticResult(BaseModel):
    """Full critic evaluation of a section."""
    scores: CriticScore  # 8 dimensions, each 1-10
    overall_pass: bool = Field(description="Whether all scores >= 8")
    failure_type: str | None = Field(default=None, description="null | writing | research_gap | human")
    issues: list[CriticIssue] = Field(default_factory=list)
    fact_check_needed: list[str] = Field(default_factory=list)
    missing_research: str | None = Field(default=None, description="Topics needing more research")
    praise: str = Field(default="", description="What's working well")
```

**8 Scoring Dimensions (update CriticScore):**
- `technical_accuracy` - Claims correct? Misleading statements?
- `completeness` - Covers what title promises?
- `code_quality` - Imports? Runnable? Well-explained?
- `clarity` - Easy to follow? Terms explained?
- `voice` - Matches style guide? No fluff? Opinionated?
- `originality` - Flagged sentences need rewriting?
- `length` - Within 20% of target?
- `diagram_quality` - (if mermaid present) Correct? Clear?

#### 6.2 Style Guide Constant (add to config.py)

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
- YAML for configuration examples.
"""

MAX_SECTION_RETRIES = 2
CRITIC_PASS_THRESHOLD = 8
```

#### 6.3 Helper Functions (add to nodes.py)

- `_build_writer_prompt(section, sources, previous_sections, style_guide)` - Constructs writer prompt
- `_write_section(section, sources, previous_text, key_manager)` - Call Gemini Flash
- `_build_critic_prompt(section, content, originality_flags)` - Constructs critic prompt
- `_critic_section(section, content, originality_flags, key_manager)` - Call Gemini Flash
- `_build_refiner_prompt(section, content, issues, human_feedback)` - Constructs refiner prompt
- `_refine_section(section, content, issues, human_feedback, key_manager)` - Call Gemini Flash

#### 6.4 Sub-steps (write → originality → critic → refine loop)

**Step 3.1: Write Section** (Model: Gemini Flash)
- Input: Section definition, validated sources, previous sections text, style guide
- Output: `SectionWriteResult` with content, sources_used, claims_to_verify
- Mermaid format: Include `\`\`\`mermaid` blocks if `needs_diagram=true`

**Step 3.2: Originality Check** (Programmatic - no LLM)
- Use `check_originality()` from tools.py
- Compare each sentence against source content
- Flag sentences with >70% similarity (SequenceMatcher)
- Check 4-gram overlaps and exact phrase matches >8 words

**Step 3.3: Section Critic** (Model: Gemini Flash)
- Input: Section content, target word count, originality flags
- Output: `SectionCriticResult` with 8 scores, issues, fact_check_needed
- Pass threshold: All scores >= 8

**Step 3.4: Decision Gate**
```python
def _handle_failure_type(failure_type: str, section_id: str, retry_count: int) -> str:
    """
    Handle critic failure:
    - "writing": Refine section and retry
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

**Step 3.5: Refine Section** (Model: Gemini Flash)
- Input: Original content, issues from critic, human feedback (if any)
- Output: Refined content addressing ALL issues
- Preserve what's working (noted in praise)

#### 6.5 write_section_node Implementation

```python
async def write_section_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 3: Write Section Node.

    Loop: write -> originality -> critic -> refine (max 2 retries)
    """
    sections = state["plan"]["sections"]
    current_idx = state.get("current_section_index", 0)
    section = sections[current_idx]

    # Get sources for this section
    sources = state["validated_sources"].get(section["id"], [])

    # Get previous sections for context
    previous_text = _get_previous_sections_text(state["section_drafts"], sections[:current_idx])

    retry_count = 0
    while retry_count <= MAX_SECTION_RETRIES:
        # Step 3.1: Write
        result = await _write_section(section, sources, previous_text, key_manager)

        # Step 3.2: Originality check
        originality_flags = check_originality(result.content, sources)

        # Step 3.3: Critic
        critic_result = await _critic_section(section, result.content, originality_flags, key_manager)

        if critic_result.overall_pass:
            # Save draft and move to next section
            state["section_drafts"][section["id"]] = result.content
            state["fact_check_items"].extend(critic_result.fact_check_needed)
            save_section_draft(job_id, section["id"], result.content)
            save_critic_feedback(job_id, section["id"], critic_result)
            break

        # Step 3.4: Decision gate
        action = _handle_failure_type(critic_result.failure_type, section["id"], retry_count)

        if action == "refine":
            # Step 3.5: Refine
            result.content = await _refine_section(section, result.content,
                                                    critic_result.issues, None, key_manager)
        elif action == "re_research":
            # Re-research and rewrite
            new_sources = await _re_research_section(section, key_manager)
            sources.extend(new_sources)
        else:
            # Human help needed
            state["needs_human_help"] = True
            state["human_help_section"] = section["id"]
            break

        retry_count += 1

    # Move to next section or complete
    state["current_section_index"] = current_idx + 1
    return state
```

#### 6.6 Checkpoints
- Save `drafts/sections/{section_id}.md` after each section passes
- Save `feedback/section_{id}_critic.json` with critic results
- Accumulate `fact_check_needed` into `state.fact_check_items`

**Unit Tests** (`tests/unit/test_writing.py`):
- Test writer prompt includes style guide and previous sections
- Test critic returns valid scores for 8 dimensions
- Test decision gate routes correctly for each failure_type
- Test refiner addresses specific issues
- Test max 2 retries before human help
- Test originality check flags similar sentences

**Integration Tests** (`tests/integration/test_writing_integration.py`):
- Test writes section with correct word count (±20%)
- Test full write→critic→refine loop
- Test checkpoint saves after each section

---

### Phase 7: Final Assembly (Phase 4)
**Goal**: Combine sections, run final critic, render diagrams, generate outputs

#### 7.1 New Pydantic Models (add to state.py)

```python
class TransitionFix(BaseModel):
    """Fix needed for transition between sections."""
    between: list[str] = Field(description="[section_a_id, section_b_id]")
    issue: str = Field(description="What's wrong with the transition")
    suggestion: str = Field(description="How to fix it")

class FinalCriticScore(BaseModel):
    """Final whole-blog critic scores (1-10)."""
    coherence: int = Field(ge=1, le=10, description="Do sections flow into each other?")
    voice_consistency: int = Field(ge=1, le=10, description="Same tone throughout?")
    no_redundancy: int = Field(ge=1, le=10, description="Any repeated points across sections?")
    narrative_arc: int = Field(ge=1, le=10, description="Problem→Solution journey clear?")
    hook_effectiveness: int = Field(ge=1, le=10, description="Does opening grab attention?")
    conclusion_strength: int = Field(ge=1, le=10, description="Actionable takeaways?")
    overall_polish: int = Field(ge=1, le=10, description="Ready to publish?")

class FinalCriticResult(BaseModel):
    """Full final critic evaluation."""
    scores: FinalCriticScore
    overall_pass: bool
    transition_fixes: list[TransitionFix] = Field(default_factory=list)
    fact_check_final: list[str] = Field(default_factory=list)
    meta_description: str = Field(description="SEO meta description (160 chars)")
    reading_time_minutes: int
    word_count: int

class BlogMetadata(BaseModel):
    """Metadata for the generated blog."""
    job_id: str
    title: str
    word_count: int
    reading_time_minutes: int
    created_at: str
    completed_at: str
    total_duration_minutes: int
    token_usage: dict[str, Any]
    llm_calls: int
    sources_used: int
    sections: int
    diagrams_generated: int
    human_interventions: int
    # SEO Fields
    seo: SEOMetadata

class SEOMetadata(BaseModel):
    """SEO metadata for the blog."""
    meta_title: str = Field(description="SEO-optimized title (50-60 chars)")
    meta_description: str = Field(description="SEO description for search results (150-160 chars)")
    excerpt: str = Field(description="Short summary for blog listings (1-2 sentences, ~200 chars)")
    focus_keyword: str = Field(description="Primary SEO keyword")
    secondary_keywords: list[str] = Field(default_factory=list, description="2-3 secondary keywords")
    og_title: str = Field(description="Open Graph title for social sharing")
    og_description: str = Field(description="Open Graph description for social sharing")

class TitleSuggestion(BaseModel):
    """A single title suggestion with reasoning."""
    title: str = Field(description="Engaging blog title (50-70 chars)")
    style: str = Field(description="Style: how-to | listicle | deep-dive | comparison | problem-solution")
    hook: str = Field(description="What makes this title compelling")
    seo_score: int = Field(ge=1, le=10, description="SEO effectiveness")
    engagement_score: int = Field(ge=1, le=10, description="Click-worthiness")

class TitleSuggestions(BaseModel):
    """5 title suggestions with recommendations."""
    suggestions: list[TitleSuggestion] = Field(min_length=5, max_length=5)
    recommended_index: int = Field(ge=0, le=4, description="Index of recommended title")
    recommendation_reason: str = Field(description="Why this title is recommended")
```

#### 7.2 Helper Functions (add to nodes.py)

- `_combine_sections(section_drafts, plan)` - Concatenate with H1 title, H2 headers
- `_build_final_critic_prompt(complete_draft)` - Prompt for 7 whole-blog dimensions
- `_final_critic(draft, key_manager)` - Call Gemini Flash for final critique
- `_apply_transition_fixes(draft, fixes, key_manager)` - Refine transitions
- `_render_all_mermaid(draft, job_dir)` - Find and render all mermaid blocks
- `_add_citations(draft, validated_sources)` - Add References section
- `_generate_fact_check_md(fact_check_items, verified_results)` - Generate fact_check.md
- `_generate_metadata(state, start_time)` - Generate metadata.json

#### 7.3 Sub-steps

**Step 4.1: Combine Sections** (Programmatic)
- Concatenate all approved sections in order
- Add main title as H1
- Add section titles as H2 (skip for hook if no title)
- Save as `drafts/v1.md`

**Step 4.2: Final Critic** (Model: Gemini Flash) - COMPLETE BLOG REVIEW
- Input: Complete draft, all section metadata
- Output: `FinalCriticResult` with 7 scores, transition_fixes, meta_description
- Pass threshold: All scores >= 8
- Individual sections already passed quality checks - now evaluate the WHOLE

**Step 4.3: Final Refine** (Model: Gemini Flash)
- If `overall_pass = false`: Apply transition fixes
- Re-run final critic (max 2 iterations)
- Save as `drafts/v2.md` (or v3.md)

**Step 4.4: Render Mermaid Diagrams** (API: kroki.io)
```python
# Find all mermaid blocks in draft
mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', draft, re.DOTALL)

for i, diagram in enumerate(mermaid_blocks):
    success = await render_mermaid(diagram, f"images/diagram_{i}.png")
    if success:
        # Replace mermaid block with image reference
        draft = draft.replace(f"```mermaid\n{diagram}\n```",
                             f"![diagram](images/diagram_{i}.png)")
    # else: keep raw mermaid as fallback
```

**Step 4.5: Add Citations** (Programmatic - skip if `--no-citations`)
```markdown
## References

1. [Redis Vector Search Documentation](https://redis.io/docs/...)
2. [GPTCache: Semantic Caching for LLMs](https://github.com/...)
```

**Step 4.6: Generate fact_check.md**
- Collect all `fact_check_items` from section critics
- Run automated verification (see Phase 11)
- Generate formatted report

**Step 4.7: Generate metadata.json**
```json
{
  "job_id": "2024-12-19_semantic-caching",
  "title": "Semantic Caching for LLM Applications",
  "meta_description": "Learn how semantic caching reduces...",
  "word_count": 1487,
  "reading_time_minutes": 6,
  "created_at": "2024-12-19T10:30:00Z",
  "completed_at": "2024-12-19T10:52:00Z",
  "total_duration_minutes": 22,
  "token_usage": {"total_in": 89000, "total_out": 12000, "by_phase": {...}},
  "llm_calls": 24,
  "sources_used": 12,
  "sections": 6,
  "diagrams_generated": 1,
  "human_interventions": 0
}
```

#### 7.4 final_assembly_node Implementation

```python
async def final_assembly_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 4: Final Assembly Node.

    Steps: combine → final critic → refine → render → citations → metadata
    """
    # Step 4.1: Combine sections
    combined_draft = _combine_sections(state["section_drafts"], state["plan"])
    save_draft(job_id, "v1.md", combined_draft)

    # Step 4.2 & 4.3: Final critic with refine loop
    iteration = 0
    while iteration < MAX_FINAL_ITERATIONS:
        critic_result = await _final_critic(combined_draft, key_manager)
        save_final_critic(job_id, critic_result)

        if critic_result.overall_pass:
            break

        # Apply transition fixes
        combined_draft = await _apply_transition_fixes(
            combined_draft, critic_result.transition_fixes, key_manager)
        iteration += 1
        save_draft(job_id, f"v{iteration + 1}.md", combined_draft)

    # Step 4.4: Render mermaid diagrams
    combined_draft, diagram_count = await _render_all_mermaid(combined_draft, job_dir)

    # Step 4.5: Add citations (unless --no-citations)
    if not state.get("no_citations"):
        combined_draft = _add_citations(combined_draft, state["validated_sources"])

    # Step 4.6: Generate fact_check.md
    await _generate_fact_check_md(state["fact_check_items"], job_dir)

    # Step 4.7: Generate metadata
    metadata = _generate_metadata(state, critic_result, diagram_count)
    save_metadata(job_id, metadata)

    # Save final draft
    save_draft(job_id, "final.md", combined_draft)
    state["final_draft"] = combined_draft
    state["final_critic_result"] = critic_result
    state["metadata"] = metadata

    return state
```

#### 7.5 Checkpoints
- `drafts/v{n}.md` after each refinement
- `feedback/final_critic.json` with scores and fixes
- `images/` with rendered mermaid PNGs
- `fact_check.md` with verification results
- `metadata.json` with job stats
- `final.md` with approved output

**Unit Tests** (`tests/unit/test_assembly.py`):
- Test section combination with correct header hierarchy
- Test final critic returns 7 scores
- Test mermaid extraction regex
- Test citations generation format
- Test metadata generation with all fields

**Integration Tests** (`tests/integration/test_assembly_integration.py`):
- Test full assembly with real mermaid rendering
- Test final critic loop with transition fixes
- Test citations are valid URLs

---

### Phase 8: Human Review (Phase 5)
**Goal**: Interactive terminal UI for approval with Rich

#### 8.1 New File: ui.py

```python
"""
UI module - Rich terminal interface for human review.

Components:
- BlogAgentUI: Main UI class with layout panels
- Progress display during execution
- Section review interface (--review-sections)
- Final review interface
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
```

#### 8.2 BlogAgentUI Class

```python
class BlogAgentUI:
    """Rich-based terminal UI for the blog agent."""

    def __init__(self, job_id: str, flags: dict):
        self.console = Console()
        self.job_id = job_id
        self.flags = flags
        self.activity_log = []

    def show_progress(self, state: BlogAgentState):
        """Display main progress view during execution."""
        pass

    def show_section_review(self, section: PlanSection, content: str,
                           critic_result: SectionCriticResult) -> str:
        """Section review interface (--review-sections)."""
        pass

    def show_final_review(self, state: BlogAgentState) -> str:
        """Final review interface before completion."""
        pass

    def log_activity(self, message: str):
        """Add to activity log with timestamp."""
        pass
```

#### 8.3 UI Components

**Header Panel:**
- Job ID, tokens used, LLM calls made
- Current project quota (e.g., "Project 2/4 (178/250 RPD)")

**Sections List:**
- ✓ Completed sections with score and word count
- ◉ Current section being processed
- ○ Pending sections

**Current Action Panel:**
- What's happening now (e.g., "Writing Implementation Deep-dive")
- Progress bar for LLM generation
- Target word count and flags (Code: yes, Diagram: yes)

**Recent Activity Log:**
- Last 4-5 actions with timestamps
- Color-coded by type (success, warning, info)

**Footer:**
- Available keyboard shortcuts

#### 8.4 Interactive Prompts

**Section Review** (when `--review-sections`):
```
Actions:
[Enter] approve & continue
[v]iew full section
[e]dit in $EDITOR
[f]eedback & rewrite (enter guidance)
[s]kip section
[q]uit & save state
```

**Final Review:**
```
Actions:
[a]pprove - finalize and save
[v]iew final.md in pager
[f]act_check - view fact_check.md
[e]dit - open in $EDITOR
[r]equest changes - enter feedback, re-run assembly
[q]uit - save state for later resume
```

#### 8.5 human_review_node Implementation

```python
async def human_review_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 5: Human Review Node.

    Presents final draft to human for approval.
    """
    ui = BlogAgentUI(state["job_id"], state.get("flags", {}))

    # Display final review UI
    decision = ui.show_final_review(state)

    if decision == "approve":
        state["human_review_decision"] = "approve"
        state["current_phase"] = Phase.DONE
        # Copy final.md to output location if configured
    elif decision == "edit":
        # User made edits in $EDITOR
        state["human_review_decision"] = "edit"
        # Reload draft from file
        state["final_draft"] = load_draft(state["job_id"], "final.md")
    elif decision == "feedback":
        # User provided feedback for re-assembly
        feedback = ui.get_feedback_input()
        state["human_feedback"] = feedback
        state["human_review_decision"] = "edit"
    elif decision == "quit":
        state["human_review_decision"] = "quit"
        # State already saved via checkpoints

    return state
```

#### 8.6 Review Mode Flags

| Flag | Behavior |
|------|----------|
| `--review-sections` | Pause after each section for approval |
| `--review-final` | Pause only before final output (default) |
| `--review-all` | Pause at every decision point |

#### 8.8 Title Suggestions & SEO Metadata (During Human Review)

**Goal**: Generate 5 engaging title options and complete SEO metadata during final review

When the blog content is ready, generate title suggestions based on the actual content:

```python
async def generate_title_suggestions(final_draft: str, original_title: str, key_manager) -> TitleSuggestions:
    """Generate 5 title suggestions based on completed blog content."""
    prompt = f"""
Based on this completed blog post, generate 5 engaging title options.

Original working title: "{original_title}"

Blog content (first 3000 chars):
{final_draft[:3000]}

Generate titles in these 5 styles:
1. How-to / Tutorial style (e.g., "How to Build X with Y")
2. Problem-Solution style (e.g., "Solving X: A Practical Guide to Y")
3. Deep-dive / Technical style (e.g., "Understanding X: From Theory to Implementation")
4. Listicle style (e.g., "5 Ways to Improve X with Y")
5. Comparison/Analysis style (e.g., "X vs Y: Which One Should You Choose?")

For each title provide:
- The title (50-70 characters, SEO-friendly, includes main keyword)
- Style type
- Hook: What makes it compelling
- SEO score (1-10): keyword relevance, search intent match, click potential
- Engagement score (1-10): curiosity, value proposition, specificity

Then recommend the BEST title considering:
- Balance of SEO (searchability) and engagement (click-worthiness)
- Accurately represents the content
- Clear value proposition for the reader
- Appropriate for technical developer audience
"""
    return await _call_llm(prompt, TitleSuggestions, key_manager)


async def generate_seo_metadata(final_draft: str, selected_title: str, key_manager) -> SEOMetadata:
    """Generate complete SEO metadata after title selection."""
    prompt = f"""
Generate SEO metadata for this blog post.

Title: "{selected_title}"

Blog content (first 4000 chars):
{final_draft[:4000]}

Generate:
1. meta_title: SEO-optimized title (50-60 chars, include primary keyword)
2. meta_description: For search results (150-160 chars, compelling, includes keyword)
3. excerpt: Short summary for blog listings (1-2 sentences, ~200 chars, enticing)
4. focus_keyword: Primary SEO keyword (the main search term)
5. secondary_keywords: 2-3 related keywords
6. og_title: Open Graph title for social sharing (can be slightly different from meta_title)
7. og_description: Social sharing description (more casual, compelling for clicks)
"""
    return await _call_llm(prompt, SEOMetadata, key_manager)
```

**Title Suggestion UI Display:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TITLE SUGGESTIONS                                                          │
│                                                                              │
│  Based on your completed blog content, here are 5 title options:            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. [HOW-TO] "How to Build Semantic Caching for LLM APIs with Redis"       │
│     Hook: Direct, actionable, mentions specific tool                        │
│     SEO: 8/10 | Engagement: 7/10                                            │
│                                                                              │
│  2. [PROBLEM-SOLUTION] "Cutting LLM API Costs by 60%: A Guide to           │
│                         Semantic Caching"                                   │
│     Hook: Leads with concrete benefit (60% cost reduction)                  │
│     SEO: 7/10 | Engagement: 9/10                                            │
│                                                                              │
│  3. [DEEP-DIVE] "Semantic Caching for LLMs: From Vector Search to          │
│                  Production"                                                │
│     Hook: Signals comprehensive coverage                                    │
│     SEO: 9/10 | Engagement: 7/10                                            │
│                                                                              │
│  4. [LISTICLE] "5 Strategies to Reduce LLM Latency with Semantic           │
│                 Caching"                                                    │
│     Hook: Numbered list, specific benefit                                   │
│     SEO: 6/10 | Engagement: 8/10                                            │
│                                                                              │
│  5. [COMPARISON] "GPTCache vs Redis VSS: Choosing the Right Semantic       │
│                   Cache"                                                    │
│     Hook: Direct comparison, decision-focused                               │
│     SEO: 8/10 | Engagement: 8/10                                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ★ RECOMMENDED: #3 "Semantic Caching for LLMs: From Vector Search to       │
│                     Production"                                             │
│                                                                              │
│  Reason: Best SEO score (9/10) while maintaining solid engagement.          │
│          Keywords match high-intent developer searches. Accurately          │
│          represents your deep-dive content with code examples.              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Select [1-5], [k]eep original, or [c]ustom:                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**SEO Metadata Display (after title selection):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SEO METADATA GENERATED                                                      │
│                                                                              │
│  Selected Title: "Semantic Caching for LLMs: From Vector Search to          │
│                   Production"                                               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Meta Title (57 chars):                                                     │
│  "Semantic Caching for LLMs: Vector Search to Production | Dev Guide"       │
│                                                                              │
│  Meta Description (158 chars):                                              │
│  "Learn how to implement semantic caching for LLM applications using        │
│   Redis vector search. Reduce API costs by 40-60% with this step-by-step   │
│   guide."                                                                   │
│                                                                              │
│  Excerpt (195 chars):                                                       │
│  "Semantic caching stores query embeddings to serve similar requests        │
│   without hitting the LLM. This guide shows you how to build one with      │
│   Redis that cuts costs by 40-60%."                                        │
│                                                                              │
│  Focus Keyword: "semantic caching LLM"                                      │
│  Secondary Keywords: "vector search", "Redis cache", "LLM optimization"    │
│                                                                              │
│  Open Graph:                                                                │
│  • OG Title: "Build Semantic Caching for Your LLM Apps"                    │
│  • OG Description: "Cut your LLM API costs by 40-60% with semantic         │
│                     caching. Here's how to build one with Redis."          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [a]ccept  [e]dit meta description  [r]egenerate  [s]kip SEO               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Updated human_review_node with Title & SEO:**

```python
async def human_review_node(state: BlogAgentState) -> dict[str, Any]:
    """
    Phase 5: Human Review Node with Title Selection & SEO.
    """
    ui = BlogAgentUI(state["job_id"], state.get("flags", {}))

    # Step 1: Generate and display title suggestions
    title_suggestions = await generate_title_suggestions(
        state["final_draft"],
        state["title"],
        key_manager
    )
    selected_title = ui.show_title_suggestions(title_suggestions, state["title"])

    # Step 2: Generate SEO metadata based on selected title
    seo_metadata = await generate_seo_metadata(
        state["final_draft"],
        selected_title,
        key_manager
    )
    seo_metadata = ui.show_seo_metadata(seo_metadata)  # Allow edits

    # Update state with selections
    state["title"] = selected_title
    state["metadata"]["seo"] = seo_metadata

    # Step 3: Display final review UI
    decision = ui.show_final_review(state)

    # ... rest of approval logic
    return state
```

**SEO Fields in metadata.json:**

```json
{
  "job_id": "2024-12-19_semantic-caching",
  "title": "Semantic Caching for LLMs: From Vector Search to Production",
  "seo": {
    "meta_title": "Semantic Caching for LLMs: Vector Search to Production | Dev Guide",
    "meta_description": "Learn how to implement semantic caching for LLM applications using Redis vector search. Reduce API costs by 40-60% with this step-by-step guide.",
    "excerpt": "Semantic caching stores query embeddings to serve similar requests without hitting the LLM. This guide shows you how to build one with Redis that cuts costs by 40-60%.",
    "focus_keyword": "semantic caching LLM",
    "secondary_keywords": ["vector search", "Redis cache", "LLM optimization"],
    "og_title": "Build Semantic Caching for Your LLM Apps",
    "og_description": "Cut your LLM API costs by 40-60% with semantic caching. Here's how to build one with Redis."
  },
  "word_count": 1487,
  "reading_time_minutes": 6
}
```

**Unit Tests** (`tests/unit/test_title_seo.py`):
- Test generates exactly 5 title suggestions
- Test each suggestion has all required fields
- Test recommended_index is valid (0-4)
- Test styles are diverse (no duplicates)
- Test SEO metadata has all required fields
- Test meta_title length (50-60 chars)
- Test meta_description length (150-160 chars)

**Integration Tests** (`tests/integration/test_title_seo_integration.py`):
- Test title generation with real LLM
- Test SEO metadata generation with real LLM
- Test interactive selection flow

#### 8.7 human_inputs/ Persistence

Save user feedback for resume:
```
human_inputs/
├── section_{id}_feedback.txt
└── final_feedback.txt
```

**Unit Tests** (`tests/unit/test_ui.py`):
- Test UI component rendering
- Test keyboard shortcut handling
- Test state transitions from decisions

**Manual Testing:**
- Visual verification of Rich layouts
- Interactive prompt flow testing

---

### Phase 9: Graph Assembly
**Goal**: Wire everything together with LangGraph

#### 9.1 New File: graph.py

```python
"""
Graph module - LangGraph StateGraph definition.

Defines the complete blog agent pipeline with conditional routing.
"""

from langgraph.graph import StateGraph, END
from .state import BlogAgentState, Phase
from .nodes import (
    topic_discovery_node,
    planning_node,
    research_node,
    validate_sources_node,
    write_section_node,
    final_assembly_node,
    human_review_node,
)
```

#### 9.2 Routing Functions

```python
def section_router(state: BlogAgentState) -> str:
    """
    Route after write_section:
    - "write_next": More sections to write
    - "all_complete": All sections done, go to assembly
    """
    sections = state.get("plan", {}).get("sections", [])
    section_drafts = state.get("section_drafts", {})

    # Count required (non-optional) sections
    required_sections = [s for s in sections if not s.get("optional")]

    if len(section_drafts) >= len(required_sections):
        return "all_complete"
    return "write_next"


def review_router(state: BlogAgentState) -> str:
    """
    Route after human_review:
    - "approved": Done, end graph
    - "edit": Back to assembly for changes
    - "rejected": End with rejection
    - "quit": End (save for resume)
    """
    decision = state.get("human_review_decision", "")

    if decision == "approve":
        return "approved"
    elif decision == "edit":
        return "edit"
    elif decision == "quit":
        return "quit"
    return "rejected"
```

#### 9.3 Graph Structure Diagram

```
┌─────────────────┐
│ topic_discovery │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    planning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    research     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│validate_sources │
└────────┬────────┘
         │
         ▼
┌─────────────────┐◄──────┐
│  write_section  │       │ (loop: write_next)
└────────┬────────┘───────┘
         │ (all_complete)
         ▼
┌─────────────────┐◄──────┐
│ final_assembly  │       │ (edit)
└────────┬────────┘       │
         │                │
         ▼                │
┌─────────────────┐───────┘
│  human_review   │
└────────┬────────┘
         │ (approved/rejected/quit)
         ▼
       [END]
```

#### 9.4 build_blog_agent_graph Implementation

```python
def build_blog_agent_graph() -> StateGraph:
    """Build and compile the blog agent graph."""
    graph = StateGraph(BlogAgentState)

    # Add all nodes
    graph.add_node("topic_discovery", topic_discovery_node)
    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("validate_sources", validate_sources_node)
    graph.add_node("write_section", write_section_node)
    graph.add_node("final_assembly", final_assembly_node)
    graph.add_node("human_review", human_review_node)

    # Set entry point
    graph.set_entry_point("topic_discovery")

    # Linear edges (phases 0.5 → 2.5)
    graph.add_edge("topic_discovery", "planning")
    graph.add_edge("planning", "research")
    graph.add_edge("research", "validate_sources")
    graph.add_edge("validate_sources", "write_section")

    # Conditional: section writing loop
    graph.add_conditional_edges(
        "write_section",
        section_router,
        {
            "write_next": "write_section",
            "all_complete": "final_assembly"
        }
    )

    # Linear edge to human review
    graph.add_edge("final_assembly", "human_review")

    # Conditional: human decision
    graph.add_conditional_edges(
        "human_review",
        review_router,
        {
            "approved": END,
            "edit": "final_assembly",
            "rejected": END,
            "quit": END
        }
    )

    return graph.compile()
```

#### 9.5 Graph Execution with Checkpointing

```python
async def run_blog_agent(title: str, context: str, length: str, flags: dict):
    """Execute the blog agent with checkpointing."""
    graph = build_blog_agent_graph()

    # Initialize state
    initial_state = {
        "job_id": slugify(title),
        "title": title,
        "context": context,
        "target_length": length,
        "flags": flags,
        "current_phase": Phase.TOPIC_DISCOVERY,
        # ... other initial fields
    }

    # Create job directory
    job_manager = JobManager()
    job_manager.create_job(initial_state["job_id"], initial_state)

    # Run graph with streaming
    async for event in graph.astream(initial_state):
        # Update UI with progress
        ui.show_progress(event)
        # Save checkpoint after each node
        job_manager.save_state(event)

    return event
```

**Unit Tests** (`tests/unit/test_graph.py`):
- Test graph builds without error
- Test section_router returns correct values
- Test review_router returns correct values
- Test all nodes are registered

**Integration Tests** (`tests/integration/test_full_graph.py`):
- Test end-to-end flow with mocked LLM
- Test resume from checkpoint at each phase
- Test section loop executes correct number of times
- Test human review routing

---

### Phase 10: CLI Interface
**Goal**: Click-based CLI with all commands and flags

#### 10.1 CLI Commands

```python
import click

@click.group()
def cli():
    """Blog Agent - AI-powered technical blog writer."""
    pass


@cli.command()
@click.option("--title", required=True, help="Blog title")
@click.option("--context", required=True, help="Context and notes")
@click.option("--length", type=click.Choice(["short", "medium", "long"]), default="medium")
@click.option("--review-sections", is_flag=True, help="Pause after each section")
@click.option("--review-final", is_flag=True, help="Pause only before final output")
@click.option("--review-all", is_flag=True, help="Pause at every decision point")
@click.option("--no-citations", is_flag=True, help="Skip references section")
@click.option("--no-hook", is_flag=True, help="Skip hook generation")
def start(title, context, length, review_sections, review_final, review_all, no_citations, no_hook):
    """Start a new blog generation job."""
    pass


@cli.command()
@click.argument("job_id")
@click.option("--review-sections", is_flag=True)
def resume(job_id, review_sections):
    """Resume an interrupted job."""
    pass


@cli.command()
@click.option("--status", type=click.Choice(["complete", "incomplete"]))
def jobs(status):
    """List all jobs."""
    pass


@cli.command()
def quota():
    """View API quota across all projects."""
    pass


@cli.command()
@click.argument("job_id")
def show(job_id):
    """View details of a specific job."""
    pass


@cli.command()
@click.argument("job_id")
@click.option("--format", "fmt", type=click.Choice(["md", "html"]), default="md")
def export(job_id, fmt):
    """Export final blog to specified format."""
    pass
```

#### 10.2 Entry Point (__main__.py)

```python
"""Entry point for python -m src.agent"""

from .ui import cli

if __name__ == "__main__":
    cli()
```

#### 10.3 Command Examples

```bash
# Start new blog
python -m src.agent start \
  --title "Semantic Caching for LLM Applications" \
  --context "Saw GPTCache on Twitter. Redis vector search." \
  --length medium \
  --review-sections

# Start with all defaults (--review-final is default)
python -m src.agent start --title "..." --context "..."

# Resume interrupted job
python -m src.agent resume semantic-caching-for-llm-applications
python -m src.agent resume <job_id> --review-sections  # Change review mode

# List jobs
python -m src.agent jobs                     # All jobs
python -m src.agent jobs --status incomplete
python -m src.agent jobs --status complete

# View quota across all projects
python -m src.agent quota

# View specific job details
python -m src.agent show <job_id>

# Export final blog
python -m src.agent export <job_id> --format md
python -m src.agent export <job_id> --format html
```

#### 10.4 Flag Behaviors

| Flag | Behavior |
|------|----------|
| `--review-sections` | Pause after each section for approval |
| `--review-final` | Pause only before final output (default) |
| `--review-all` | Pause at every decision point |
| `--no-citations` | Skip References section at end |
| `--no-hook` | Skip hook generation (start with problem section) |
| `--length short` | Target ~800 words |
| `--length medium` | Target ~1500 words (default) |
| `--length long` | Target ~2500 words |

**Unit Tests** (`tests/unit/test_cli.py`):
- Test start command validates required options
- Test resume loads job state
- Test jobs filters by status
- Test export generates correct format

**Integration Tests** (`tests/integration/test_cli_integration.py`):
- Test full start→complete flow via CLI
- Test resume continues from correct phase

---

## API Key Setup (4 Keys for Fallback)

As per the design, we'll use 4 separate Google API keys for quota management:

1. **Get API Keys from Google AI Studio**:
   - Go to https://aistudio.google.com/apikey
   - Create 4 separate API keys (or from 4 different GCP projects)

2. **Create `.env` file** in project root:
   ```env
   GOOGLE_API_KEY_1=your_first_key
   GOOGLE_API_KEY_2=your_second_key
   GOOGLE_API_KEY_3=your_third_key
   GOOGLE_API_KEY_4=your_fourth_key
   ```

3. **Simple KeyManager** (will be implemented in `tools.py`):
   - Tracks usage per key (requests per day)
   - Rotates to next key on 429 (rate limit) errors
   - Selects key with most remaining quota

---

## Infrastructure Components

### Retry Manager

Retry strategies for all external API calls:

```python
RETRY_CONFIG = {
    "gemini_api": {
        "max_retries": 3,
        "backoff": [1, 2, 4],  # seconds (exponential)
        "on_429": "switch_project",
        "on_500": "retry_same_project",
        "on_all_exhausted": "pause_notify_human"
    },
    "web_search": {
        "max_retries": 2,
        "backoff": [2, 5],
        "on_rate_limit": "wait_60s_retry",
        "on_fail": "try_alternate_query"
    },
    "web_fetch": {
        "max_retries": 2,
        "timeout": 30,
        "on_fail": "skip_source",
        "on_timeout": "skip_source"
    },
    "mermaid_render": {
        "max_retries": 2,
        "fallback": "save_raw_mermaid"  # Human can render manually
    }
}
```

### Enhanced Checkpoint Structure

```
~/.blog_agent/
├── config.yaml                    # API keys, default settings
├── usage/
│   └── {date}.json                # Daily token usage per project
└── jobs/
    └── {job_id}/
        ├── state.json             # Current phase, progress, can_resume
        ├── input.json             # Original title + context
        ├── topic_context.json     # Discovery search results
        ├── plan.json              # Outline, search queries, metadata
        ├── research/
        │   ├── cache/             # Raw fetched articles (for resume)
        │   ├── validated/         # Post-quality-filter sources
        │   └── sources.json       # URL → section mapping for citations
        ├── drafts/
        │   ├── sections/          # Individual section drafts
        │   ├── v1.md              # Combined draft v1
        │   └── v2.md              # After refinement
        ├── feedback/
        │   ├── section_{n}_critic.json
        │   └── final_critic.json
        ├── human_inputs/          # User guidance/feedback saved
        ├── fact_check.md          # Claims to verify
        ├── images/                # Rendered mermaid PNGs
        └── final.md               # Approved output
```

---

## Dependencies to Install

```bash
pip install langgraph duckduckgo-search trafilatura httpx rich click python-dotenv
```

Or add to `pyproject.toml`:
```toml
dependencies = [
    "langgraph>=0.2.0",
    "duckduckgo-search>=6.0.0",
    "trafilatura>=1.8.0",
    "httpx>=0.27.0",
    "rich>=13.7.0",
    "click>=8.1.0",
    "langchain-google-genai>=2.0.0",
    "pydantic>=2.5.0",
]
```

---

## Phase 11: Automated Fact Checking

**Goal**: Verify claims using LLM + Web Search

#### 11.1 Fact Checking Workflow

For each claim in `fact_check_items`:
1. Search DuckDuckGo for the claim
2. Fetch content from top 3 sources
3. LLM evaluates claim against sources
4. Mark as: `verified` | `unverified` | `conflicting`

#### 11.2 Implementation

```python
class ClaimVerification(BaseModel):
    """Result of verifying a single claim."""
    claim: str
    status: str = Field(description="verified | unverified | conflicting")
    sources: list[str] = Field(default_factory=list, description="URLs that support/refute")
    reasoning: str = Field(description="Brief explanation")


async def _verify_claim(claim: str, key_manager: KeyManager) -> ClaimVerification:
    """
    Verify a single claim using web search + LLM.
    """
    # Step 1: Search for claim
    results = await search_duckduckgo(claim, max_results=5)

    # Step 2: Fetch content from top sources
    sources = []
    for r in results[:3]:
        content = await fetch_url_content(r["url"])
        if content["success"]:
            sources.append({
                "url": r["url"],
                "title": r.get("title", ""),
                "content": content["content"][:2000]  # Limit context
            })

    if not sources:
        return ClaimVerification(
            claim=claim,
            status="unverified",
            sources=[],
            reasoning="No sources found to verify claim"
        )

    # Step 3: LLM evaluation
    prompt = f"""Evaluate whether this claim is accurate based on the sources.

Claim: "{claim}"

Sources:
{_format_verification_sources(sources)}

Respond with:
- status: "verified" (sources support claim), "unverified" (no supporting evidence),
          or "conflicting" (sources disagree)
- reasoning: Brief explanation (1-2 sentences)
"""

    result = await _call_llm(prompt, ClaimVerification, key_manager)
    result.sources = [s["url"] for s in sources]
    return result


async def verify_all_claims(fact_check_items: list[str], key_manager: KeyManager) -> list[ClaimVerification]:
    """Verify all claims concurrently."""
    tasks = [_verify_claim(claim, key_manager) for claim in fact_check_items]
    return await asyncio.gather(*tasks)
```

#### 11.3 fact_check.md Format

```markdown
# Fact Check Report

Generated: 2024-12-19T10:52:00Z
Claims verified: 5

## ✓ Verified Claims

- [x] **"HNSW provides O(log n) search complexity"**
  - Sources: [Redis Docs](https://redis.io/docs/...), [Pinecone Blog](https://...)
  - Reasoning: Multiple authoritative sources confirm HNSW's logarithmic search time.

- [x] **"Semantic caching can reduce API costs by 40-60%"**
  - Sources: [GPTCache Paper](https://...), [Medium Article](https://...)
  - Reasoning: Performance benchmarks from GPTCache show 40-60% cache hit rates.

## ✗ Unverified Claims

- [ ] **"GPTCache reduces latency by 80%"**
  - Sources: None found
  - Reasoning: No reliable benchmark data found to support this specific number.

## ⚠ Conflicting Information

- [?] **"Redis VSS supports up to 10M vectors per index"**
  - Sources: [Redis Docs](https://...), [Stack Overflow](https://...)
  - Reasoning: Official docs mention no hard limit; community reports vary by hardware.

---

**Action Required**: Please verify unverified and conflicting claims before publication.
```

**Unit Tests** (`tests/unit/test_fact_check.py`):
- Test claim verification with mocked search/LLM
- Test fact_check.md generation format
- Test handling of no sources found

**Integration Tests** (`tests/integration/test_fact_check_integration.py`):
- Test real claim verification with API
- Test concurrent verification

---

## Extra Ideas (Future Enhancements)

### Parallel Section Writing
- Identify independent sections (sections with `role: "deep_dive"`)
- Use `asyncio.gather()` to write 2-3 sections concurrently
- Reduces total generation time by ~30-40%
- Constraint: Sections must not depend on each other's content

```python
async def write_parallel_sections(sections: list, sources: dict, key_manager):
    """Write multiple independent sections concurrently."""
    # Group sections by dependency
    independent = [s for s in sections if s["role"] in ["deep_dive", "production"]]
    sequential = [s for s in sections if s["role"] not in ["deep_dive", "production"]]

    # Write sequential sections first (hook, problem, why)
    for section in sequential:
        await write_single_section(section, sources, key_manager)

    # Write independent sections in parallel
    await asyncio.gather(*[
        write_single_section(s, sources, key_manager) for s in independent
    ])
```

### Readability Scoring
- Calculate Flesch-Kincaid readability score
- Add as 9th dimension to section critic
- Target: Grade 10-12 level for technical blogs
- Flag sections that are too complex (Grade 16+) or too simple (Grade 8-)

```python
def calculate_readability(text: str) -> float:
    """Calculate Flesch-Kincaid Grade Level."""
    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())
    syllables = count_syllables(text)

    if sentences == 0 or words == 0:
        return 0

    return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
```

### Social Media Snippets
- Auto-generate after final assembly
- Add to metadata.json

```python
class SocialSnippets(BaseModel):
    """Auto-generated social media content."""
    twitter_thread: list[str] = Field(description="5 tweets, each ≤280 chars")
    linkedin_post: str = Field(description="LinkedIn summary, ~300 words")
    one_liner: str = Field(description="Single sentence hook")


async def generate_social_snippets(final_draft: str, key_manager) -> SocialSnippets:
    """Generate social media snippets from final blog."""
    prompt = f"""Generate social media content for this blog post:

{final_draft[:3000]}

Create:
1. Twitter thread (5 tweets, each under 280 characters, use emojis sparingly)
2. LinkedIn post (~300 words, professional tone)
3. One-liner hook for sharing
"""
    return await _call_llm(prompt, SocialSnippets, key_manager)
```

### Version Diffing
- Save each draft version (v1, v2, v3)
- Generate diff between versions using difflib
- Show in terminal UI during refinement
- Help user understand what changed

```python
def show_version_diff(v1: str, v2: str) -> str:
    """Generate readable diff between versions."""
    differ = difflib.unified_diff(
        v1.splitlines(keepends=True),
        v2.splitlines(keepends=True),
        fromfile='v1.md',
        tofile='v2.md'
    )
    return ''.join(differ)
```

---

## Token Budget Table

Estimated token usage for a medium-length blog (~1500 words, 6 sections):

| Phase | LLM Calls | Model | Tokens In | Tokens Out |
|-------|-----------|-------|-----------|------------|
| **Topic Discovery** | 1 | Flash-Lite | 500 | 200 |
| **Planning** | 1 | Flash-Lite | 2,500 | 1,000 |
| **Research Validation** | 1 | Flash-Lite | 4,000 | 500 |
| **Per Section (×6):** | | | | |
| - Write | 1 | Flash | 6,000 | 1,500 |
| - Critic | 1 | Flash | 2,500 | 500 |
| - Refine (50% need) | 0.5 | Flash | 3,000 | 1,000 |
| **Section subtotal (×6)** | ~18 | | ~81,000 | ~20,400 |
| **Final Assembly:** | | | | |
| - Final Critic | 1 | Flash | 8,000 | 1,000 |
| - Final Refine | 1 | Flash | 6,000 | 2,000 |
| - Fact Check (~5 claims) | 5 | Flash-Lite | 5,000 | 1,000 |
| **TOTAL** | **~30** | | **~107,000** | **~26,100** |

**Budget Analysis:**
- 4 projects × 250 RPD = **1,000 requests/day** → Using ~30 ✓
- 4 projects × 250k TPM = **1M tokens/min** → Using ~133k total ✓
- **Plenty of headroom** for retries, longer blogs, or multiple jobs per day

---

## Terminal UI Mockups

### Main Progress View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BLOG AGENT v1.0                                    Tokens: 45.2k | 24 calls│
│  Job: semantic-caching              Project: 2/4 (178/250 RPD remaining)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ● Phase: WRITING                          [--review-sections ON]           │
│                                                                              │
│  Sections:                                                                  │
│  ├── ✓ Hook                          [9.2] 98 words                        │
│  ├── ✓ Problem Statement             [8.8] 215 words                       │
│  ├── ✓ Why Semantic Caching          [9.0] 312 words  [has diagram]        │
│  ├── ◉ Implementation Deep-dive      [WRITING...]                          │
│  ├── ○ Production Considerations                                            │
│  └── ○ Conclusion                                                           │
│                                                                              │
│  ┌─ Current Action ─────────────────────────────────────────────────────┐  │
│  │ Writing section "Implementation Deep-dive"                            │  │
│  │ Using 4 validated sources | Target: 500 words | Code: yes            │  │
│  │                                                                       │  │
│  │ ████████████░░░░░░░░ 60% generating...                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Recent Activity ────────────────────────────────────────────────────┐  │
│  │ 10:42:15  Section "Why Semantic Caching" passed critic (score: 9.0)  │  │
│  │ 10:41:03  Generated mermaid diagram for architecture                 │  │
│  │ 10:40:22  Section "Why Semantic Caching" written (312 words)         │  │
│  │ 10:39:45  Fetched 4 sources for "Why Semantic Caching"               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [p]ause  [f]eedback  [v]iew draft  [s]kip section  [q]uit & save          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Section Review View (--review-sections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BLOG AGENT v1.0                                    Tokens: 52.1k | 28 calls│
│  Job: semantic-caching              Project: 2/4 (174/250 RPD remaining)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ◉ SECTION REVIEW: "Implementation Deep-dive"                               │
│                                                                              │
│  ┌─ Scores ─────────────────────────────────────────────────────────────┐  │
│  │ Technical: 9 │ Complete: 8 │ Code: 9 │ Clarity: 9 │ Voice: 8 │ Orig: 9│  │
│  │                                                                       │  │
│  │ Overall: 8.7 ✓ PASSED                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Preview ────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ## Building Semantic Cache with Redis                               │  │
│  │                                                                       │  │
│  │  Semantic caching stores query embeddings alongside cached responses.│  │
│  │  When a new query arrives, you compute its embedding and search for  │  │
│  │  similar vectors. If similarity exceeds your threshold, return the   │  │
│  │  cached response without hitting the LLM.                            │  │
│  │                                                                       │  │
│  │  Here's a minimal implementation using Redis Stack:                  │  │
│  │                                                                       │  │
│  │  ```python                                                           │  │
│  │  import redis                                                        │  │
│  │  from redis.commands.search.query import Query                       │  │
│  │  ...                                                                 │  │
│  │  ```                                                                 │  │
│  │                                          [showing 15/52 lines]       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Critic Notes ───────────────────────────────────────────────────────┐  │
│  │ ✓ Strong: Clear explanation, runnable code example                   │  │
│  │ → Minor: Consider adding error handling to the code snippet          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Enter] approve & continue   [v]iew full   [e]dit   [f]eedback & rewrite  │
│  [s]kip section               [q]uit & save                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Final Review View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✓ BLOG COMPLETE                                                            │
│                                                                              │
│  Title: Semantic Caching for LLM Applications                              │
│  Words: 1,487 | Reading time: 6 min | Sections: 6                          │
│  LLM calls: 24 | Tokens: 101k | Duration: 22 min                           │
│                                                                              │
│  ┌─ Quality Scores ─────────────────────────────────────────────────────┐  │
│  │ Coherence: 9 │ Voice: 9 │ Narrative: 8 │ Hook: 9 │ Conclusion: 8     │  │
│  │                                                                       │  │
│  │ Overall: 8.6 ✓ PASSED                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ⚠ FACT CHECK REQUIRED (see fact_check.md):                                │
│    • ✓ "HNSW provides O(log n) search complexity" - VERIFIED               │
│    • ✗ "GPTCache reduces latency by 80%" - UNVERIFIED                      │
│    • ? "Redis VSS supports 10M vectors" - CONFLICTING                       │
│                                                                              │
│  Generated Files:                                                           │
│    → final.md (ready to publish)                                           │
│    → images/diagram_0.png                                                  │
│    → fact_check.md                                                         │
│    → metadata.json                                                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [v] view final.md   [f] fact_check.md   [e] edit   [a] approve            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files to Reference

- [design.md](design.md) - Full prompts and phase specifications

---

## Directory Structure After Implementation

```
blog_data_analyst/
├── design.md                    # Design specification
├── IMPLEMENTATION_PLAN.md       # This file
├── CLAUDE.md                    # Claude Code guidelines
├── pyproject.toml               # Dependencies
├── src/
│   └── agent/
│       ├── __init__.py          # Module exports
│       ├── state.py             # State schema + Pydantic models
│       ├── graph.py             # LangGraph StateGraph definition
│       ├── nodes.py             # All phase implementations
│       ├── tools.py             # Search, fetch, render utilities
│       ├── config.py            # Constants, retry config, style guide
│       ├── key_manager.py       # API key rotation
│       ├── ui.py                # Rich Terminal UI + CLI
│       └── __main__.py          # Entry point for python -m src.agent
├── tests/
│   ├── unit/                    # Fast, isolated tests (mocked dependencies)
│   │   ├── test_state.py
│   │   ├── test_tools.py
│   │   ├── test_topic_discovery.py
│   │   ├── test_planning.py
│   │   ├── test_research.py
│   │   ├── test_validate_sources.py
│   │   ├── test_writing.py      # NEW
│   │   ├── test_assembly.py     # NEW
│   │   ├── test_graph.py        # NEW
│   │   ├── test_ui.py           # NEW
│   │   ├── test_cli.py          # NEW
│   │   ├── test_fact_check.py   # NEW
│   │   ├── test_title_seo.py    # NEW - Title suggestions & SEO metadata
│   │   └── test_key_manager.py
│   └── integration/             # Real API tests (slower, needs keys)
│       ├── test_topic_discovery_integration.py
│       ├── test_planning_integration.py
│       ├── test_research_integration.py
│       ├── test_writing_integration.py     # NEW
│       ├── test_assembly_integration.py    # NEW
│       ├── test_full_graph.py              # NEW
│       ├── test_cli_integration.py         # NEW
│       ├── test_fact_check_integration.py  # NEW
│       └── test_title_seo_integration.py   # NEW - Title & SEO with real LLM
└── ~/.blog_agent/               # Runtime data (created at runtime)
    ├── config.yaml
    ├── usage/
    │   └── {date}.json          # Daily token usage per project
    └── jobs/
        └── {job_id}/
            ├── state.json
            ├── input.json
            ├── topic_context.json
            ├── plan.json
            ├── research/
            │   ├── cache/
            │   ├── validated/
            │   └── sources.json
            ├── drafts/
            │   ├── sections/
            │   ├── v1.md
            │   └── v2.md
            ├── feedback/
            │   ├── section_{n}_critic.json
            │   └── final_critic.json
            ├── human_inputs/
            ├── fact_check.md
            ├── images/
            ├── metadata.json
            └── final.md
```

---

## Vertical Slice Implementation Plan

**Strategy**: Build thin slices across all layers first, then add depth. Each slice produces testable, working functionality.

---

### ✓ COMPLETED: Foundation (Phases 1-5)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✓ DONE | Foundation - directory structure, tools.py |
| Phase 2 | ✓ DONE | State + Checkpoints - BlogAgentState, JobManager |
| Phase 3 | ✓ DONE | Topic Discovery - search queries, web context |
| Phase 4 | ✓ DONE | Planning - blog outline with sections |
| Phase 5 | ✓ DONE | Research + Validation - fetch and filter sources |

---

### ROUND 1: Minimal End-to-End CLI (Slices 6.1-6.4)

**Goal**: Run `python -m src.agent start --title "..." --context "..."` and get final.md

---

#### Slice 6.1: Minimal Section Writer

**Files**: `config.py`, `nodes.py`

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
- `_build_writer_prompt(section, sources, previous_sections, style_guide)` - Constructs writer prompt
- `_write_section(section, sources, previous_text, key_manager)` - Call Gemini Flash
- `write_section_node(state)` - Writes ONE section per call (for graph loop)

**Behavior**:
- No critic, no refine - just write
- Save draft to `drafts/sections/{section_id}.md`
- Update `current_section_index` for next iteration

**Tests**:
- `tests/unit/test_writing.py`: Prompt building, mock LLM response
- `tests/integration/test_writing_integration.py`: Write one section with real LLM

**Output**: Can write all sections (no quality checks yet)

---

#### Slice 6.2: Basic Assembly

**Files**: `nodes.py`

**Add to nodes.py**:
- `_combine_sections(section_drafts, plan)` - Concatenate with H1 title, H2 headers
- `final_assembly_node(state)` - Combines sections, saves final.md

**Behavior**:
- No final critic, no mermaid, no citations
- Just combine and save to `final.md`

**Tests**:
- `tests/unit/test_assembly.py`: Section combination, header hierarchy
- `tests/integration/test_assembly_integration.py`: Combines real sections

**Output**: Combined draft saved as final.md

---

#### Slice 6.3: Graph + Routing

**Files**: `graph.py` (NEW)

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

**Output**: Complete pipeline working (no critic, no human review)

---

#### Slice 6.4: CLI Start Command

**Files**: `__main__.py` (NEW), `ui.py` (basic)

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

**Output**: `python -m src.agent start --title "Semantic Caching" --context "..."` works!

---

### ROUND 2: Quality Layer (Slices 6.5-6.8)

**Goal**: Add section critic, refine loop, final critic, basic human approval

---

#### Slice 6.5: Section Critic

**Files**: `state.py`, `nodes.py`

**Update state.py** - Expand CriticScore to 8 dimensions:
```python
class CriticScore(BaseModel):
    """8 scoring dimensions (1-10 scale)."""
    technical_accuracy: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    code_quality: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    voice: int = Field(ge=1, le=10)
    originality: int = Field(ge=1, le=10)
    length: int = Field(ge=1, le=10)
    diagram_quality: int = Field(ge=1, le=10, default=10)  # 10 if no diagram

class CriticIssue(BaseModel):
    """A single issue identified by the critic."""
    dimension: str
    location: str
    problem: str
    suggestion: str
```

**Add to nodes.py**:
- `_build_critic_prompt(section, content, target_words)` - 8 dimensions prompt
- `_critic_section(section, content, key_manager)` - Call Gemini Flash
- Update `write_section_node` to call critic after write

**Behavior**:
- Critic runs but doesn't block (evaluate only)
- Save critic results to `feedback/section_{id}_critic.json`
- Log pass/fail status

**Tests**:
- `tests/unit/test_writing.py`: Add critic tests
- `tests/integration/test_writing_integration.py`: Critic with real LLM

---

#### Slice 6.6: Section Refine Loop

**Files**: `nodes.py`, `config.py`

**Add to config.py**:
```python
MAX_SECTION_RETRIES = 2
CRITIC_PASS_THRESHOLD = 8
```

**Add to nodes.py**:
- `_build_refiner_prompt(section, content, issues)` - Fix specific issues
- `_refine_section(section, content, issues, key_manager)` - Call Gemini Flash
- `_handle_failure(failure_type, retry_count)` - Basic decision gate

**Update write_section_node**:
```python
retry_count = 0
while retry_count <= MAX_SECTION_RETRIES:
    content = await _write_section(...)
    critic_result = await _critic_section(...)

    if critic_result.overall_pass:
        break

    # Refine and retry
    content = await _refine_section(section, content, critic_result.issues, key_manager)
    retry_count += 1
```

**Tests**:
- `tests/unit/test_writing.py`: Refiner, decision gate, max retries
- `tests/integration/test_writing_integration.py`: Full write→critic→refine loop

---

#### Slice 6.7: Final Critic

**Files**: `state.py`, `nodes.py`

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
    between: list[str]  # [section_a_id, section_b_id]
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
- `_build_final_critic_prompt(draft)` - 7 dimensions for whole blog
- `_final_critic(draft, key_manager)` - Call Gemini Flash
- `_apply_transition_fixes(draft, fixes, key_manager)` - Refine transitions

**Update final_assembly_node** with critic loop (max 2 iterations)

**Tests**:
- `tests/unit/test_assembly.py`: Final critic prompt, scoring
- `tests/integration/test_assembly_integration.py`: Assembly with final critic

---

#### Slice 6.8: Basic Human Review

**Files**: `nodes.py`, `graph.py`

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

# Add to build_blog_agent_graph():
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

---

### ROUND 3: Full Human Experience (Slices 6.9-6.11)

**Goal**: Title suggestions, SEO, Rich UI

---

#### Slice 6.9: Title Suggestions + SEO

**Files**: `state.py`, `nodes.py`

**Add to state.py**:
```python
class SEOMetadata(BaseModel):
    meta_title: str = Field(description="50-60 chars")
    meta_description: str = Field(description="150-160 chars")
    excerpt: str = Field(description="~200 chars")
    focus_keyword: str
    secondary_keywords: list[str] = Field(default_factory=list)
    og_title: str
    og_description: str

class TitleSuggestion(BaseModel):
    title: str
    style: str  # how-to | listicle | deep-dive | comparison | problem-solution
    hook: str
    seo_score: int = Field(ge=1, le=10)
    engagement_score: int = Field(ge=1, le=10)

class TitleSuggestions(BaseModel):
    suggestions: list[TitleSuggestion] = Field(min_length=5, max_length=5)
    recommended_index: int = Field(ge=0, le=4)
    recommendation_reason: str
```

**Add to nodes.py**:
- `generate_title_suggestions(final_draft, original_title, key_manager)`
- `generate_seo_metadata(final_draft, selected_title, key_manager)`

**Integrate into human_review_node**:
1. Generate title suggestions
2. Display and let user select
3. Generate SEO metadata
4. Save to metadata.json

**Tests**:
- `tests/unit/test_title_seo.py`: Title/SEO generation
- `tests/integration/test_title_seo_integration.py`: With real LLM

---

#### Slice 6.10: Rich Progress UI

**Files**: `ui.py` (expand)

**Create BlogAgentUI class**:
```python
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live

class BlogAgentUI:
    def __init__(self, job_id: str):
        self.console = Console()
        self.job_id = job_id
        self.activity_log = []

    def show_progress(self, state: BlogAgentState):
        """Display main progress view during execution."""
        # Header with job info
        # Sections list with status (✓ ◉ ○)
        # Current action panel
        # Activity log

    def log_activity(self, message: str):
        """Add timestamped activity."""
```

**Wire into graph execution** - Update CLI to use UI during run

**Tests**:
- `tests/unit/test_ui.py`: Component rendering (mock console)
- Manual visual testing

---

#### Slice 6.11: Rich Review Interfaces

**Files**: `ui.py`, `nodes.py`

**Add to BlogAgentUI**:
- `show_section_review(section, content, critic_result)` - For --review-sections
- `show_final_review(state)` - Final review with scores
- `show_title_suggestions(suggestions)` - Title selection interface
- `show_seo_metadata(metadata)` - SEO display

**Actions**:
- `[Enter]` approve & continue
- `[v]` view full content
- `[e]` edit in $EDITOR
- `[f]` feedback & rewrite
- `[q]` quit & save

**Add --review-sections flag to CLI**

**Tests**:
- `tests/unit/test_ui.py`: UI state transitions
- Manual visual testing

---

### ROUND 4: Polish Features (Slices 6.12-6.15)

**Goal**: Mermaid, citations, fact checking, full CLI

---

#### Slice 6.12: Mermaid + Citations

**Files**: `nodes.py`, `tools.py`

**Add to tools.py** (if not already):
```python
async def render_mermaid(code: str, output_path: str) -> bool:
    """Render mermaid diagram via kroki.io."""
```

**Add to nodes.py**:
- `_render_all_mermaid(draft, job_dir)` - Find and render all mermaid blocks
- `_add_citations(draft, validated_sources)` - Add References section

**Update final_assembly_node**:
1. Render mermaid diagrams → `images/`
2. Replace mermaid blocks with image references
3. Add References section (unless --no-citations)

**Tests**:
- `tests/unit/test_assembly.py`: Mermaid regex, citation formatting
- `tests/integration/test_assembly_integration.py`: Real mermaid rendering

---

#### Slice 6.13: Fact Checking

**Files**: `state.py`, `nodes.py`

**Add to state.py**:
```python
class ClaimVerification(BaseModel):
    claim: str
    status: str  # verified | unverified | conflicting
    sources: list[str] = Field(default_factory=list)
    reasoning: str
```

**Add to nodes.py**:
- `_verify_claim(claim, key_manager)` - Search + LLM evaluation
- `verify_all_claims(claims, key_manager)` - Concurrent verification
- `_generate_fact_check_md(verifications, job_dir)` - Generate report

**Integrate into final_assembly_node**:
1. Collect fact_check_items from section critics
2. Verify claims concurrently
3. Generate fact_check.md

**Tests**:
- `tests/unit/test_fact_check.py`: Verification logic
- `tests/integration/test_fact_check_integration.py`: Real fact checking

---

#### Slice 6.14: CLI Completion

**Files**: `__main__.py`

**Add commands**:
```python
@cli.command()
@click.argument("job_id")
@click.option("--review-sections", is_flag=True)
def resume(job_id, review_sections):
    """Resume an interrupted job."""

@cli.command()
@click.option("--status", type=click.Choice(["complete", "incomplete"]))
def jobs(status):
    """List all jobs."""

@cli.command()
def quota():
    """View API quota across all projects."""

@cli.command()
@click.argument("job_id")
def show(job_id):
    """View details of a specific job."""

@cli.command()
@click.argument("job_id")
@click.option("--format", "fmt", type=click.Choice(["md", "html"]), default="md")
def export(job_id, fmt):
    """Export final blog to specified format."""
```

**Tests**:
- `tests/unit/test_cli.py`: All command validation
- `tests/integration/test_cli_integration.py`: Resume flow

---

#### Slice 6.15: Advanced Writing Features

**Files**: `nodes.py`, `tools.py`

**Add to tools.py**:
- `check_originality(content, sources)` - difflib similarity check

**Add to nodes.py**:
- Full decision gate: `writing` | `research_gap` | `human`
- `_re_research_section(section, key_manager)` - Fetch more sources for gaps

**Update write_section_node**:
1. After write, run originality check
2. Pass originality flags to critic
3. Decision gate routes to: refine, re-research, or request human help

**Tests**:
- `tests/unit/test_writing.py`: Originality check, decision routing
- `tests/integration/test_writing_integration.py`: Re-research flow

---

## Implementation Checklist

| Slice | Description | Unit Tests | Integration Tests |
|-------|-------------|------------|-------------------|
| 6.1 | Minimal Section Writer | test_writing.py | test_writing_integration.py |
| 6.2 | Basic Assembly | test_assembly.py | test_assembly_integration.py |
| 6.3 | Graph + Routing | test_graph.py | test_full_graph.py |
| 6.4 | CLI Start Command | test_cli.py | test_cli_integration.py |
| 6.5 | Section Critic | test_writing.py | test_writing_integration.py |
| 6.6 | Section Refine Loop | test_writing.py | test_writing_integration.py |
| 6.7 | Final Critic | test_assembly.py | test_assembly_integration.py |
| 6.8 | Basic Human Review | test_graph.py | Manual |
| 6.9 | Title + SEO | test_title_seo.py | test_title_seo_integration.py |
| 6.10 | Rich Progress UI | test_ui.py | Manual |
| 6.11 | Rich Review Interfaces | test_ui.py | Manual |
| 6.12 | Mermaid + Citations | test_assembly.py | test_assembly_integration.py |
| 6.13 | Fact Checking | test_fact_check.py | test_fact_check_integration.py |
| 6.14 | CLI Completion | test_cli.py | test_cli_integration.py |
| 6.15 | Advanced Writing | test_writing.py | test_writing_integration.py |

---

## Quick Start After Each Slice

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

**After Slice 6.11** (full UI):
```bash
python -m src.agent start --title "..." --context "..." --review-sections
# Rich UI with progress, section reviews, title selection
```

**After Slice 6.14** (full CLI):
```bash
python -m src.agent jobs --status incomplete
python -m src.agent resume semantic-caching-for-llms
python -m src.agent export semantic-caching-for-llms --format html
```
