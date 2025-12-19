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

### Phase 4: Planning (Phase 1)
**Goal**: Generate blog outline with sections

1. Implement `planning_node(state)` in `nodes.py`:
   - Build prompt with topic context snippets
   - Call Gemini Flash-Lite for structured BlogPlan output
   - Parse sections with: id, title, role, search_queries, target_words

**Integration Test**: `test_planning.py`
- Test generates plan with 5-6 sections
- Test each section has required fields

---

### Phase 5: Research (Phase 2 + 2.5)
**Goal**: Fetch and validate sources for each section

1. Implement `research_node(state)`:
   - For each section, search using its queries
   - Fetch content from top URLs
   - Cache in state.research_cache

2. Implement `validate_sources_node(state)`:
   - Call Gemini Flash-Lite to filter sources
   - Keep sources where `use=true`
   - Ensure min 2 sources per section

**Integration Test**: `test_research.py`
- Test fetches content from multiple URLs
- Test validation filters out low-quality sources

---

### Phase 6: Writing Loop (Phase 3)
**Goal**: Write, critique, and refine each section

1. Implement `write_section_node(state)`:
   - Write section using Gemini Flash with sources + style guide
   - Run originality check (programmatic)
   - Call critic (Gemini Flash) for quality scores
   - If scores < 8: refine and retry (max 2 retries)
   - Save draft, move to next section

2. Implement helper functions:
   - `write_section(section, sources, previous_text)`
   - `critic_section(section, content, originality_flags)`
   - `refine_section(content, critic_feedback)`

**Integration Test**: `test_writing.py`
- Test writes section with correct word count
- Test critic returns valid scores
- Test refine improves flagged issues

---

### Phase 7: Assembly (Phase 4)
**Goal**: Combine sections into final blog

1. Implement `final_assembly_node(state)`:
   - Concatenate all sections with H1 title, H2 section headers
   - Run final critic for coherence/voice consistency
   - Render mermaid diagrams to PNG
   - Add References section with source URLs
   - Generate metadata.json

**Integration Test**: `test_assembly.py`
- Test combines sections correctly
- Test diagrams render to images/
- Test citations section is generated

---

### Phase 8: Human Review (Phase 5)
**Goal**: Interactive terminal UI for approval

1. Implement `human_review_node(state)`:
   - Display draft preview, scores, fact-check items
   - Prompt for: [a]pprove, [v]iew, [e]dit, [q]uit
   - Return decision in state

2. Implement `ui.py` with Rich:
   - `BlogAgentUI` class with layout panels
   - Progress display during execution
   - Human review interface

**Integration Test**: `test_ui.py` (manual/visual testing)

---

### Phase 9: Graph Assembly
**Goal**: Wire everything together with LangGraph

1. Implement `graph.py`:
   ```python
   def build_blog_agent_graph():
       graph = StateGraph(BlogAgentState)

       # Add nodes
       graph.add_node("topic_discovery", topic_discovery_node)
       graph.add_node("planning", planning_node)
       graph.add_node("research", research_node)
       graph.add_node("validate_sources", validate_sources_node)
       graph.add_node("write_section", write_section_node)
       graph.add_node("final_assembly", final_assembly_node)
       graph.add_node("human_review", human_review_node)

       # Add edges
       graph.set_entry_point("topic_discovery")
       graph.add_edge("topic_discovery", "planning")
       graph.add_edge("planning", "research")
       graph.add_edge("research", "validate_sources")
       graph.add_edge("validate_sources", "write_section")

       # Conditional: loop for sections
       graph.add_conditional_edges(
           "write_section",
           section_router,  # returns "write_next" or "all_complete"
           {"write_next": "write_section", "all_complete": "final_assembly"}
       )

       graph.add_edge("final_assembly", "human_review")

       # Conditional: human decision
       graph.add_conditional_edges(
           "human_review",
           review_router,  # returns "approved", "edit", or "rejected"
           {"approved": END, "edit": "final_assembly", "rejected": END}
       )

       return graph.compile()
   ```

**Integration Test**: `test_full_graph.py`
- Test end-to-end flow with mocked LLM
- Test resume from checkpoint

---

### Phase 10: CLI Interface
**Goal**: Command-line interface with Rich UI

1. Add CLI commands (in `ui.py` or `__main__.py`):
   - `python -m src.agent start --title "..." --context "..." --length medium`
   - `python -m src.agent resume <job_id>`
   - `python -m src.agent jobs [--status complete|incomplete]`

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

## Key Files to Reference

- [design.md](design.md) - Full prompts and phase specifications

---

## Directory Structure After Implementation

```
blog_data_analyst/
├── design.md                    # Design specification
├── IMPLEMENTATION_PLAN.md       # This file
├── pyproject.toml               # Dependencies
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── state.py             # State schema + Pydantic models
│       ├── graph.py             # LangGraph definition
│       ├── nodes.py             # All phase implementations
│       ├── tools.py             # Search, fetch, render utilities
│       └── ui.py                # Rich Terminal UI + CLI
├── tests/
│   ├── unit/                    # Fast, isolated tests (mocked dependencies)
│   │   ├── test_state.py
│   │   ├── test_tools.py
│   │   └── test_checkpoints.py
│   └── integration/             # Real API tests (slower, needs keys)
│       ├── test_topic_discovery.py
│       ├── test_planning.py
│       ├── test_research.py
│       ├── test_writing.py
│       ├── test_assembly.py
│       └── test_full_graph.py
└── ~/.blog_agent/               # Runtime data (created at runtime)
    ├── config.yaml
    └── jobs/
        └── {job_id}/            # e.g., "semantic-caching-for-llm-applications"
            ├── state.json
            ├── plan.json
            ├── drafts/
            └── images/
```

---

## Next Steps

1. Start with **Phase 1: Foundation** - create directory structure and implement `tools.py`
2. After each phase, run integration tests before moving to next
3. Build incrementally, testing as we go
