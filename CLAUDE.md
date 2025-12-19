# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Guidelines

1. **Think like a professional developer** - Understand the problem thoroughly, evaluate all available options, and present options to the designer/owner of this application before implementing.

2. **Don't assume anything** - Analyze the existing code and ask clarifying questions with your recommendations. When uncertain, ask rather than guess.

3. **Always test your code** - Write unit tests and integration tests for all new functionality. No code is complete without tests.

4. **Do the right thing** - No shortcuts. Follow best practices, write clean code, and maintain the established patterns in the codebase.

5. **Keep plans concise** - Plans should be short and to the point. No fluff. Skip code details and use bulleted points to explain how each task will be accomplished.

## Project Overview

Blog Agent - An AI-powered technical blog writer that generates publication-ready markdown from a title and context. Uses Python + LangGraph with Google Gemini (Flash/Flash-Lite models).

## Environment Setup

**IMPORTANT:** This project uses a Python virtual environment. All commands must be run with the venv activated or using the venv Python directly.

```bash
# Create virtual environment (first time only)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (after activation)
pip install langgraph duckduckgo-search trafilatura httpx rich click python-dotenv langchain-google-genai pydantic

# Or install from pyproject.toml if available
pip install -e .
```

The virtual environment is located at `.venv/` in the project root.

## Commands

**All commands below assume the virtual environment is activated, or prefix with `.venv/bin/`**

```bash
# Run the agent
python -m src.agent start --title "Topic Title" --context "Notes and context" --length medium

# Resume interrupted job
python -m src.agent resume <job_id>

# List jobs
python -m src.agent jobs [--status complete|incomplete]

# Run tests (use PYTHONPATH=. to ensure src module is found)
PYTHONPATH=. pytest tests/unit/                    # Fast unit tests (mocked)
PYTHONPATH=. pytest tests/integration/             # Integration tests (needs API keys)
PYTHONPATH=. pytest tests/unit/test_tools.py -v    # Single test file
PYTHONPATH=. pytest -k "test_search"               # Run tests matching pattern

# Alternative: run tests using venv directly without activation
PYTHONPATH=. .venv/bin/pytest tests/unit/
```

## Architecture

### LangGraph State Machine

The agent is a LangGraph StateGraph with 7 nodes connected in a pipeline:

```
topic_discovery → planning → research → validate_sources → write_section → final_assembly → human_review
                                                              ↑______|  (loops per section)
```

**Conditional Routing:**
- `write_section` loops back to itself until all sections complete
- `human_review` can route back to `final_assembly` for edits

### Pipeline Phases

| Phase | Node | Model | Purpose |
|-------|------|-------|---------|
| 0.5 | `topic_discovery` | Flash-Lite | Generate search queries, gather web context |
| 1 | `planning` | Flash-Lite | Create blog outline with sections |
| 2 | `research` | - | Fetch content from URLs via trafilatura |
| 2.5 | `validate_sources` | Flash-Lite | Filter sources by quality/relevance |
| 3 | `write_section` | Flash | Write, critique, refine each section (loop) |
| 4 | `final_assembly` | Flash | Combine sections, render diagrams, add citations |
| 5 | `human_review` | - | Interactive approval via terminal UI |

### Module Structure (src/agent/)

- **state.py** - `BlogAgentState` TypedDict, Pydantic models for LLM outputs, `Phase` enum
- **graph.py** - LangGraph StateGraph definition, `build_blog_agent_graph()`, routing functions
- **nodes.py** - Node implementations: `topic_discovery_node`, `planning_node`, `write_section_node`, etc.
- **tools.py** - Utilities: `search_duckduckgo`, `fetch_url_content`, `check_originality`, `render_mermaid`, KeyManager
- **ui.py** - Rich terminal UI, CLI commands, human review interface

### State Flow

State is a TypedDict that flows through all nodes. Key fields:
- `job_id` - Slugified topic name (e.g., "semantic-caching-for-llm-applications")
- `current_phase` - Phase enum value
- `plan` - Blog outline with sections
- `section_drafts` - Dict of section_id → markdown content
- `validated_sources` - Dict of section_id → list of source objects

### Checkpointing

Jobs persist to `~/.blog_agent/jobs/{job_id}/` with:
- `state.json` - Current phase and progress
- `plan.json` - Blog outline
- `drafts/sections/` - Individual section drafts
- `images/` - Rendered mermaid diagrams

## API Keys

Uses 4 Google Gemini API keys for quota management. Create `.env`:

```env
GOOGLE_API_KEY_1=...
GOOGLE_API_KEY_2=...
GOOGLE_API_KEY_3=...
GOOGLE_API_KEY_4=...
```

KeyManager rotates keys on 429 errors and tracks usage per key.

## Key Design Documents

- **design.md** - Full architecture spec with prompts, state machine, terminal UI mockups
- **IMPLEMENTATION_PLAN.md** - Incremental build phases with test strategy
