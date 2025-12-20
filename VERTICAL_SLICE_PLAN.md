# Vertical Slice Implementation Plan

**Strategy**: Build thin slices across all layers first, then add depth. Each slice produces testable, working functionality.

**Testing**: Unit + Integration tests for every slice.

---

## Status Overview

| Round | Slices | Goal | Status |
|-------|--------|------|--------|
| Foundation | Phases 1-5 | Tools, State, Discovery, Planning, Research | ✓ DONE |
| Round 1 | 6.1-6.4 | Minimal End-to-End CLI | ✓ DONE |
| Round 2 | 6.5-6.8 | Quality Layer (Critic + Refine) | ○ PENDING |
| Round 3 | 6.9-6.11 | Full Human Experience (Rich UI) | ○ PENDING |
| Round 4 | 6.12-6.15 | Polish Features | ○ PENDING |

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

### Slice 6.5: Section Critic

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/state.py` - Expand CriticScore to 8 dimensions
- `src/agent/nodes.py` - Add critic functions

**Add to state.py**:
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
    diagram_quality: int = Field(ge=1, le=10, default=10)

class CriticIssue(BaseModel):
    """A single issue identified by the critic."""
    dimension: str
    location: str
    problem: str
    suggestion: str
```

**Add to nodes.py**:
```python
def _build_critic_prompt(section, content, target_words) -> str:
    """8 dimensions evaluation prompt."""

async def _critic_section(section, content, key_manager) -> SectionCriticResult:
    """Call Gemini Flash for critique."""
```

**Update write_section_node** to call critic after write (evaluate only, no refine yet)

**Tests**:
- `tests/unit/test_writing.py`: Add critic tests
- `tests/integration/test_writing_integration.py`: Critic with real LLM

**Milestone**: Critic runs and logs pass/fail, saves to `feedback/`

---

### Slice 6.6: Section Refine Loop

**Status**: ✓ COMPLETE

**Files to modify**:
- `src/agent/config.py` - Add retry constants
- `src/agent/nodes.py` - Add refine functions

**Add to config.py**:
```python
MAX_SECTION_RETRIES = 2
CRITIC_PASS_THRESHOLD = 8
```

**Add to nodes.py**:
```python
def _build_refiner_prompt(section, content, issues) -> str:
    """Fix specific issues prompt."""

async def _refine_section(section, content, issues, key_manager) -> str:
    """Call Gemini Flash for refinement."""

def _handle_failure(failure_type: str, retry_count: int) -> str:
    """Basic decision gate: refine or give up."""
```

**Update write_section_node** with retry loop:
```python
retry_count = 0
while retry_count <= MAX_SECTION_RETRIES:
    content = await _write_section(...)
    critic_result = await _critic_section(...)

    if critic_result.overall_pass:
        break

    content = await _refine_section(section, content, critic_result.issues, key_manager)
    retry_count += 1
```

**Tests**:
- `tests/unit/test_writing.py`: Refiner, decision gate, max retries
- `tests/integration/test_writing_integration.py`: Full write→critic→refine loop

**Milestone**: Sections are refined until they pass quality gate

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

### Slice 6.9: Title Suggestions + SEO

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/state.py` - Add SEO models
- `src/agent/nodes.py` - Add title/SEO generation

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
```python
async def generate_title_suggestions(final_draft, original_title, key_manager):
    """Generate 5 title options with recommendations."""

async def generate_seo_metadata(final_draft, selected_title, key_manager):
    """Generate SEO metadata for selected title."""
```

**Integrate into human_review_node**

**Tests**:
- `tests/unit/test_title_seo.py`: Title/SEO generation
- `tests/integration/test_title_seo_integration.py`: With real LLM

**Milestone**: Title selection and SEO metadata saved to metadata.json

---

### Slice 6.10: Rich Progress UI

**Status**: ○ PENDING

**Files to create/modify**:
- `src/agent/ui.py` (expand)

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

**Wire into graph execution**

**Tests**:
- `tests/unit/test_ui.py`: Component rendering (mock console)
- Manual visual testing

**Milestone**: Beautiful terminal UI during generation

---

### Slice 6.11: Rich Review Interfaces

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/ui.py` - Add review interfaces
- `src/agent/nodes.py` - Integrate UI
- `src/agent/__main__.py` - Add --review-sections flag

**Add to BlogAgentUI**:
```python
def show_section_review(self, section, content, critic_result):
    """Section review interface for --review-sections."""

def show_final_review(self, state):
    """Final review with quality scores."""

def show_title_suggestions(self, suggestions):
    """Title selection interface."""

def show_seo_metadata(self, metadata):
    """SEO display interface."""
```

**Actions**:
- `[Enter]` approve & continue
- `[v]` view full content
- `[e]` edit in $EDITOR
- `[f]` feedback & rewrite
- `[q]` quit & save

**Tests**:
- `tests/unit/test_ui.py`: UI state transitions
- Manual visual testing

**Milestone**: Full interactive review experience

---

## ROUND 4: Polish Features (Slices 6.12-6.15)

**Goal**: Mermaid, citations, fact checking, full CLI

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

### Slice 6.14: CLI Completion

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/__main__.py` - Add all commands

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

**Milestone**: Full CLI with all commands working

---

### Slice 6.15: Advanced Writing Features

**Status**: ○ PENDING

**Files to modify**:
- `src/agent/tools.py` - Add originality check
- `src/agent/nodes.py` - Add decision gate, re-research

**Add to tools.py**:
```python
def check_originality(content: str, sources: list[dict]) -> list[dict]:
    """Check for similarity with sources using difflib."""
```

**Add to nodes.py**:
```python
def _decision_gate(failure_type: str, retry_count: int) -> str:
    """Full decision gate: writing | research_gap | human"""

async def _re_research_section(section, key_manager) -> list[dict]:
    """Fetch more sources when research gap detected."""
```

**Update write_section_node**:
1. After write, run originality check
2. Pass originality flags to critic
3. Decision gate routes to: refine, re-research, or human help

**Tests**:
- `tests/unit/test_writing.py`: Originality check, decision routing
- `tests/integration/test_writing_integration.py`: Re-research flow

**Milestone**: Robust handling of quality issues

---

## Implementation Checklist

| Slice | Description | Unit Tests | Integration Tests | Status |
|-------|-------------|------------|-------------------|--------|
| 6.1 | Minimal Section Writer | test_writing.py | test_writing_integration.py | ✓ |
| 6.2 | Basic Assembly | test_assembly.py | test_assembly_integration.py | ✓ |
| 6.3 | Graph + Routing | test_graph.py | test_full_graph.py | ✓ |
| 6.4 | CLI Start Command | test_cli.py | test_cli_integration.py | ✓ |
| 6.5 | Section Critic | test_writing.py | test_writing_integration.py | ✓ |
| 6.6 | Section Refine Loop | test_writing.py | test_writing_integration.py | ✓ |
| 6.7 | Final Critic | test_assembly.py | test_assembly_integration.py | ○ |
| 6.8 | Basic Human Review | test_graph.py | Manual | ○ |
| 6.9 | Title + SEO | test_title_seo.py | test_title_seo_integration.py | ○ |
| 6.10 | Rich Progress UI | test_ui.py | Manual | ○ |
| 6.11 | Rich Review Interfaces | test_ui.py | Manual | ○ |
| 6.12 | Mermaid + Citations | test_assembly.py | test_assembly_integration.py | ○ |
| 6.13 | Fact Checking | test_fact_check.py | test_fact_check_integration.py | ○ |
| 6.14 | CLI Completion | test_cli.py | test_cli_integration.py | ○ |
| 6.15 | Advanced Writing | test_writing.py | test_writing_integration.py | ○ |

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

---

## Notes

- Each slice should be completable in ~30-60 minutes
- Run tests after each slice before moving to next
- Commit after each slice passes tests
- Update status (○ → ◉ → ✓) as you progress
