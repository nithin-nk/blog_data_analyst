# Blog Agent

> AI-powered technical blog writer that creates unique, bookmarkable content

An intelligent blog writing system that analyzes existing content, identifies gaps, and generates publication-ready technical blog posts with comprehensive quality checks.

## What It Does

Blog Agent takes a title and context notes as input and generates a complete, publication-ready technical blog post. Unlike generic AI writing tools, it:

**🎯 Content Differentiation Engine**
- Analyzes the top 10 existing articles on your topic
- Identifies content gaps and unique angles
- Creates differentiated content that stands out from the crowd
- Ensures your blog is bookmarkable and shares a unique perspective

**✅ 15-Dimension Quality System**
- **8 core dimensions**: Technical accuracy, code quality, clarity, voice consistency, originality, completeness, length, diagram quality
- **2 Google E-E-A-T dimensions**: Expertise demonstration, people-first value
- **5 differentiation dimensions**: Unique insights, actionability, specificity, depth appropriateness, production readiness

**📤 Complete Outputs**
- Publication-ready markdown
- Mermaid diagrams (auto-rendered to PNG)
- Citations and references
- Fact-check report for human verification
- Execution metrics and token usage stats

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd blog_data_analyst

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Setup API Keys

```bash
# Copy example env file
cp .env.example .env

# Add your Google Gemini API keys (1-5 keys for quota management)
# Edit .env and add:
GOOGLE_API_KEY_1=your_key_here
GOOGLE_API_KEY_2=your_key_here  # Optional: more keys = higher quota
GOOGLE_API_KEY_3=your_key_here  # Optional
GOOGLE_API_KEY_4=your_key_here  # Optional
GOOGLE_API_KEY_5=your_key_here  # Optional
```

**Pro tip:** Use keys from different Google Cloud projects (even under the same account) to multiply your quota. Each project gets 20 requests/day per model for free tier.

### Generate Your First Blog

```bash
python -m src.agent start \
  --title "Semantic Caching for LLM Applications" \
  --context "Redis, GPTCache, vector similarity search" \
  --length medium
```

Output:
```
🚀 Blog Agent
Title: Semantic Caching for LLM Applications
Length: medium

  🔍 Topic Discovery complete
  🌐 Content Landscape Analysis complete
  📋 Planning complete
  🔬 Researching complete
  ✓ Validating Sources complete
  ✍️  Writing sections...
    ✓ Hook
    ✓ The Problem: Why Traditional Caching Fails for LLMs
    ✓ What Semantic Caching Does Differently
    ✓ Building Semantic Cache with Redis
    ✓ Production Considerations
    ✓ Conclusion
  📦 Assembling complete
  👀 Final review complete

✅ Blog generated successfully!
Output: ~/.blog_agent/jobs/semantic-caching-for-llm-applications/final.md

Stats:
  📝 Words: 1,487
  ⏱️  Reading time: 6 min
  📑 Sections: 6
```

## CLI Commands

### Start New Blog

```bash
python -m src.agent start \
  --title "Your Blog Title" \
  --context "Notes, references, ideas..." \
  [--length short|medium|long] \
  [--auto-select]
```

Options:
- `--title`: Blog title (required)
- `--context`: Context notes, references, ideas for the blog (required)
- `--length`: Target blog length - `short` (~800 words), `medium` (~1500 words), `long` (~2500 words). Default: `medium`
- `--auto-select`: Skip section selection and include all sections automatically

### Resume Interrupted Job

```bash
python -m src.agent resume <job_id>
```

Jobs are automatically saved at each phase. Resume anytime if interrupted by Ctrl+C or errors.

### List All Jobs

```bash
# List all jobs
python -m src.agent jobs

# Filter by status
python -m src.agent jobs --status complete
python -m src.agent jobs --status incomplete
```

### Show Job Details

```bash
python -m src.agent show <job_id>
```

Displays job title, phase, sections, and output location.

## Features

### 🎯 Content Differentiation Engine

The system doesn't just generate content - it creates **unique, bookmarkable blogs** that stand out:

1. **Landscape Analysis**: Fetches and analyzes the top 10 existing articles on your topic
2. **Gap Identification**: Identifies what's missing, shallow, or wrong in existing content
3. **Unique Angle**: Determines a differentiated perspective based on content gaps
4. **Differentiation Requirements**: Ensures your blog meets specific uniqueness criteria

Example differentiation strategies:
- "Production-focused guide covering error handling, monitoring, and scaling (others focus only on basics)"
- "Benchmark-driven comparison with real numbers (others are purely theoretical)"
- "Cost optimization angle: reduce LLM API bills by 60% (others don't mention costs)"

### ✅ 15-Dimension Quality System

Every section is evaluated against 15 quality dimensions:

**Core Quality (8 dimensions):**
- Technical accuracy
- Completeness
- Code quality (imports, runnability, explanation)
- Clarity
- Voice consistency
- Originality (plagiarism detection)
- Length appropriateness
- Diagram quality

**Google E-E-A-T (2 dimensions):**
- First-hand expertise demonstration
- People-first value (bookmarkable, shareable)

**Differentiation (5 dimensions):**
- Unique insights not found elsewhere
- Actionability (clear next steps, copy-paste code)
- Specificity (concrete examples, real numbers)
- Depth appropriateness (edge cases, "why" not just "how")
- Production readiness (error handling, real-world concerns)

Sections scoring below 8/10 on any dimension are automatically refined.

### 🔬 Multi-Source Research

- **Web Search**: DuckDuckGo API for real-time research
- **Content Extraction**: Trafilatura for clean article extraction
- **Source Validation**: LLM-based quality filtering for relevance and freshness
- **Originality Checking**: Automated plagiarism detection using SequenceMatcher

### ✍️ Intelligent Writing Pipeline

LangGraph-based state machine with write-critique-refine loops:

1. **Write** section using validated sources and style guide
2. **Originality check** using programmatic plagiarism detection
3. **Critique** using 15-dimension quality evaluation
4. **Refine** if any dimension scores < 8/10 (max 2 retries)
5. Move to next section

### 📦 Complete Outputs

Every job produces:
- `final.md` - Publication-ready markdown
- `metadata.json` - Stats, token usage, reading time
- `fact_check.md` - Claims flagged for human verification
- `images/` - Rendered mermaid diagrams (PNG)
- `plan.json` - Blog outline and section plan
- `content_strategy.json` - Unique angle and gaps analysis
- `drafts/` - Section drafts and versions

## Output Structure

Jobs are saved to `~/.blog_agent/jobs/<job_id>/`:

```
<job_id>/
├── final.md                    # Publication-ready blog
├── metadata.json               # Stats, token usage, reading time
├── fact_check.md              # Claims to verify
├── images/                     # Rendered mermaid diagrams
│   └── diagram_0.png
├── plan.json                   # Blog outline
├── content_strategy.json       # Unique angle and gaps analysis
├── research/
│   ├── cache/                  # Fetched articles
│   ├── validated/              # Quality-filtered sources
│   └── sources.json            # Source citations
└── drafts/
    ├── sections/               # Individual section drafts
    ├── v1.md                   # Combined draft v1
    └── v2.md                   # After refinement
```

## How It Works

### Pipeline Phases

1. **🔍 Topic Discovery** - Generates search queries and gathers web context
2. **🌐 Content Landscape Analysis** - Analyzes top 10 articles, identifies unique angle
3. **📋 Planning** - Creates outline with differentiation strategy
4. **🔬 Research** - Fetches and validates sources from the web
5. **✍️  Writing** - Generates sections with write-critique-refine loop
6. **📦 Assembly** - Combines sections, renders diagrams, adds citations
7. **👀 Review** - Final quality check and human approval

### Technology Stack

- **Python 3.11+**
- **LangGraph** - State machine orchestration
- **Google Gemini API** - Flash-Lite and Flash models
- **DuckDuckGo** - Web search
- **Trafilatura** - Content extraction
- **Rich** - Terminal UI
- **pytest** - Testing framework

## API Quota Management

The system supports up to 5 Google API keys for intelligent quota management:

- ✅ Automatically rotates between keys
- ✅ Tracks requests per day (RPD) per key
- ✅ Switches keys on 429 errors
- ✅ Pauses gracefully when all keys exhausted
- ✅ Resets daily at midnight Pacific Time

**Free tier limits:** 20 requests/day per model per project

**Pro tip:** Create multiple Google Cloud projects (even under the same Google account) and use one API key from each project to multiply your quota.

Example with 4 projects: 4 × 20 = 80 requests/day

## For Developers

### Project Structure

```
src/agent/
├── __main__.py         # CLI commands (start, resume, jobs, show)
├── graph.py            # LangGraph pipeline definition
├── nodes.py            # Pipeline node implementations
├── state.py            # State definitions, Pydantic models, Phase enum
├── config.py           # Configuration constants (STYLE_GUIDE, prompts)
├── tools.py            # Utility functions (search, fetch, originality check)
└── key_manager.py      # API key quota management

tests/
├── unit/               # Unit tests (fully mocked, fast)
└── integration/        # Integration tests (mocked LLM, real tools)
```

### Development Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
PYTHONPATH=. pytest tests/unit/           # Fast unit tests
PYTHONPATH=. pytest tests/unit/ -n auto   # Parallel execution (all CPU cores)
PYTHONPATH=. pytest tests/integration/    # Integration tests (needs API keys)

# Run specific test file
PYTHONPATH=. pytest tests/unit/test_writing.py -v

# Run tests matching pattern
PYTHONPATH=. pytest -k "test_search"
```

### Testing Strategy

**Unit Tests** (`tests/unit/`)
- Fully mocked - All external dependencies (LLM, web requests) are mocked
- Fast execution - Run in < 5 seconds total
- Deterministic - Same input always produces same output
- No API keys required - Can run offline

**Integration Tests** (`tests/integration/`)
- Mocked LLM responses - Use predefined LLM outputs
- Real tools - Web search (DuckDuckGo) and content fetching run against real URLs
- Avoid API quota - Mock LLM calls to prevent hitting free tier limits
- Test flows - Verify node transitions, state updates, error handling

**Vertical Slice Approach**
- Build thin slices across all layers first, then add depth
- Each slice produces testable, working functionality

### Contributing

1. Read the documentation:
   - [design.md](design.md) - Complete architecture specification
   - [CLAUDE.md](CLAUDE.md) - Guidance for AI assistants

2. Follow existing patterns:
   - Match the code style in the codebase
   - Use Pydantic models for LLM structured outputs
   - Add type hints to all functions
   - Write docstrings for public functions

3. Write tests:
   - Unit tests for all new functionality
   - Integration tests for end-to-end flows
   - Run tests before submitting PRs

4. Keep it simple:
   - No over-engineering
   - No features beyond what's requested
   - No unnecessary abstractions

## Documentation

- **[design.md](design.md)** - Complete technical specification, prompts, state machine diagrams
- **[CLAUDE.md](CLAUDE.md)** - Guidance for AI assistants working with this codebase

## Examples

### Generate a blog on semantic caching

```bash
python -m src.agent start \
  --title "Semantic Caching for LLM Applications" \
  --context "Saw GPTCache on Twitter. Redis vector search. Reduce latency and cost." \
  --length medium
```

### Output Stats Example

```
✅ Blog generated successfully!
Output: ~/.blog_agent/jobs/semantic-caching-for-llm-applications/final.md

Stats:
  📝 Words: 1,487
  ⏱️  Reading time: 6 min
  📑 Sections: 6

📊 Execution Metrics
┌──────────────────────┬──────────┬───────────┬──────────────┬───────────┐
│ Node                 │ Duration │ API Calls │ Tokens (i/o) │ Est. Cost │
├──────────────────────┼──────────┼───────────┼──────────────┼───────────┤
│ Topic Discovery      │ 8.2s     │ 1         │ 500 / 200    │ $0.0003   │
│ Content Landscape    │ 45.1s    │ 11        │ 13k / 2.5k   │ $0.0042   │
│ Planning             │ 12.3s    │ 1         │ 2.5k / 1k    │ $0.0008   │
│ Research             │ 18.7s    │ -         │ -            │ -         │
│ Validate Sources     │ 6.1s     │ 1         │ 4k / 500     │ $0.0011   │
│ Write Section        │ 3.2m     │ 18        │ 81k / 20.4k  │ $0.0215   │
│ Final Assembly       │ 35.2s    │ 2         │ 14k / 3k     │ $0.0035   │
│ Total                │ 4.5m     │ 33        │ 111k / 27k   │ $0.0303   │
└──────────────────────┴──────────┴───────────┴──────────────┴───────────┘

🔑 API Key Usage:
  GOOGLE_API_KEY_1: 15 calls, 235 remaining
  GOOGLE_API_KEY_2: 18 calls, 232 remaining
```

### Resume an interrupted job

```bash
# Job was interrupted (Ctrl+C, error, or API quota exhausted)
⚠️  Interrupted. Job saved for resume.
Resume with: python -m src.agent resume semantic-caching-for-llm-applications

# Resume later
python -m src.agent resume semantic-caching-for-llm-applications
```

### List jobs and check status

```bash
# List all jobs
python -m src.agent jobs

Jobs (5 total)

  ✓ semantic-caching-for-llm-applications
      Semantic Caching for LLM Applications
      Phase: done

  ○ redis-performance-optimization
      Redis Performance Optimization Tips
      Phase: writing

# Filter by status
python -m src.agent jobs --status incomplete
```

## Troubleshooting

### API Key Errors

```
Error: No API keys found
```
**Solution:** Copy `.env.example` to `.env` and add your Google Gemini API keys:
```bash
cp .env.example .env
# Edit .env and add GOOGLE_API_KEY_1=your_key_here
```

### Rate Limit (429 errors)

```
Error: All API keys exhausted
```
**Solution:** Wait until next day when quota resets (midnight Pacific Time), or add more keys from different Google Cloud projects.

### Import Errors

```
Error: No module named 'src.agent'
```
**Solution:** Run with `PYTHONPATH=.` prefix:
```bash
PYTHONPATH=. python -m src.agent start --title "..." --context "..."
```

Or install the package:
```bash
pip install -e .
```

### Job Interrupted

```
⚠️  Interrupted. Job saved for resume.
```
**Solution:** Jobs are automatically checkpointed at each phase. Resume anytime:
```bash
python -m src.agent resume <job_id>
```

### Virtual Environment Not Activated

```
ModuleNotFoundError: No module named 'langgraph'
```
**Solution:** Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for state machine orchestration
- Powered by [Google Gemini](https://ai.google.dev/) API (Flash-Lite and Flash models)
- Based on Google's ["Creating Helpful Content" guidelines](https://developers.google.com/search/docs/fundamentals/creating-helpful-content)
- Web search via [DuckDuckGo API](https://duckduckgo.com/)
- Content extraction via [Trafilatura](https://github.com/adbar/trafilatura)
- Terminal UI via [Rich](https://github.com/Textualize/rich)

---

**Questions or Issues?** Check the [design.md](design.md) for detailed architecture or open an issue on GitHub.
