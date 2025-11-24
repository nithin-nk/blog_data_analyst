# 📋 Step-by-Step Implementation Plan for Blog Data Analyst AI Agent

Based on your README, here's a comprehensive implementation plan following clean architecture principles:

## **Phase 0: Project Setup & Architecture** 🏗️

### Step 0.1: Initialize Project Structure
```
blog_data_analyst/
├── .env.example                    # Environment variables template
├── .env                           # Actual secrets (git-ignored)
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata & tool configs
├── .venv/                         # Virtual environment
├── src/
│   ├── __init__.py
│   ├── main.py                    # Orchestrator/entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Environment-aware config
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── yaml_parser.py         # YAML input parser
│   ├── research/
│   │   ├── __init__.py
│   │   ├── search_agent.py        # browser-use integration
│   │   └── content_extractor.py   # crawl4ai integration
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── content_generator.py   # LLM content generation
│   │   ├── code_generator.py      # Code & mermaid diagrams
│   │   └── title_generator.py     # Titles, tags, descriptions
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── seo_optimizer.py       # SEO analysis & suggestions
│   │   └── quality_checker.py     # Dual LLM review
│   ├── refinement/
│   │   ├── __init__.py
│   │   └── blog_refiner.py        # Iterative improvement
│   ├── media/
│   │   ├── __init__.py
│   │   └── image_generator.py     # Banana/Nano API integration
│   ├── converters/
│   │   ├── __init__.py
│   │   └── md_to_html.py          # Markdown to HTML
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py        # I/O operations
│       └── logger.py              # Structured logging
├── tests/
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_research.py
│   ├── test_generation.py
│   ├── test_optimization.py
│   ├── test_converters.py
│   └── fixtures/
│       └── sample_input.yaml      # Test data
├── inputs/                        # User YAML files
├── outputs/                       # Generated blog posts
│   ├── drafts/
│   ├── final/
│   └── images/
└── README.md
```

### Step 0.2: Core Dependencies
- **LLM Framework**: `langchain` or `anthropic` SDK
- **Web Research**: `browser-use` (Google search)
- **Content Extraction**: `crawl4ai`
- **YAML Parsing**: `pyyaml`
- **SEO Analysis**: `textstat`, `spacy`
- **HTML Conversion**: `markdown`, `beautifulsoup4`
- **Image Generation**: API client for Banana
- **Testing**: `pytest`, `pytest-mock`
- **Code Quality**: `black`, `ruff`, `mypy`

---

## **Phase 1: Foundation Components** 🔧

### Step 1.1: Configuration & Environment Setup
**File**: `src/config/settings.py`
- Load environment variables (API keys for LLMs, Banana, etc.)
- Define environment modes (`dev`, `test`, `prod`)
- Set file paths, rate limits, retry configs
- **Test**: Verify config loading in all environments

### Step 1.2: YAML Input Parser
**File**: `src/parsers/yaml_parser.py`
- Parse YAML structure: `topic`, `outline` (list of questions)
- Detect special markers: `Mermaid:`, `Code:`
- Validate required fields
- Return structured data model (Pydantic or dataclass)
- **Test**: Parse valid/invalid YAML fixtures

### Step 1.3: Logging & File Handlers
**Files**: `src/utils/logger.py`, `src/utils/file_handler.py`
- Structured logging with verbosity to stdout
- Safe file I/O with error handling
- **Test**: Log output verification, file write/read tests

---

## **Phase 2: Research & Content Extraction** 🔍

### Step 2.1: Google Search Agent
**File**: `src/research/search_agent.py`
- Integrate `browser-use` library
- For each outline question, generate search queries
- Execute Google searches and extract top 5-10 URLs
- Rate limiting and error handling
- **Test**: Mock browser-use responses, verify URL extraction

### Step 2.2: Web Content Extractor
**File**: `src/research/content_extractor.py`
- Integrate `crawl4ai` library
- Extract clean text, headings, code blocks from URLs
- Handle extraction failures gracefully
- Store extracted content with source citations
- **Test**: Mock crawl4ai responses, test parser robustness

---

## **Phase 3: Content Generation** ✍️

### Step 3.1: Subtopic Content Generator
**File**: `src/generation/content_generator.py`
- For each outline question:
  - Combine research data from multiple sources
  - Use LLM to generate 300-500 word section
  - Include inline citations `[Source Name](URL)`
  - Maintain consistent tone and style
- **Test**: Mock LLM responses, verify citation format

### Step 3.2: Code & Diagram Generator
**File**: `src/generation/code_generator.py`
- Detect `Code:` and `Mermaid:` markers in outline
- Generate relevant code snippets with explanations
- Generate Mermaid diagram syntax
- Validate Mermaid syntax
- **Test**: Syntax validation for code and diagrams

### Step 3.3: Blog Post Combiner
**File**: `src/generation/content_generator.py` (method)
- Merge all subtopic sections into cohesive post
- Add transitions between sections
- Ensure narrative flow
- **Test**: Verify section ordering and transitions

### Step 3.4: Title, Tags & Meta Generator
**File**: `src/generation/title_generator.py`
- Generate 3-5 catchy title options (150-160 char meta)
- Extract 8-12 relevant tags/keywords
- Write compelling meta description (150-160 char)
- **Test**: Verify length constraints, keyword relevance

---

## **Phase 4: Optimization & Quality Control** 🎯

### Step 4.1: SEO Optimizer
**File**: `src/optimization/seo_optimizer.py`
- **Keyword Density**: Calculate and suggest optimal placement
- **Readability**: Flesch-Kincaid, Gunning Fog scores
- **Header Hierarchy**: Validate H1→H2→H3 structure
- **Internal/External Links**: Suggest relevant links
- **Meta Tags**: Validate title, description, keywords
- Return structured suggestions
- **Test**: Analyze sample blog posts, verify scoring

### Step 4.2: Dual LLM Quality Checker
**File**: `src/optimization/quality_checker.py`
- Use two different LLMs (e.g., Claude + GPT-4)
- Each rates blog 1-10 on:
  - Accuracy, Clarity, Engagement, Structure, Completeness
- Extract specific improvement suggestions
- Average scores from both LLMs
- **Test**: Mock LLM responses, verify scoring logic

---

## **Phase 5: Refinement Loop** 🔄

### Step 5.1: Blog Refiner
**File**: `src/refinement/blog_refiner.py`
- Accept feedback from quality checker
- Use LLM to apply specific improvements
- Track iteration count (max 3-5 iterations)
- Re-check quality after each iteration
- Break loop if score ≥ 8 or max iterations reached
- **Test**: Mock feedback loop, verify iteration limits

---

## **Phase 6: Media & Output Generation** 🎨

### Step 6.1: Image Generator
**File**: `src/media/image_generator.py`
- Generate prompt from blog topic and key themes
- Call Banana/Nano API for image generation
- Download and save image locally
- Return image path
- **Test**: Mock Banana API, verify image download

### Step 6.2: Markdown to HTML Converter
**File**: `src/converters/md_to_html.py`
- Convert markdown to HTML with proper formatting
- Apply CSS styling (include inline or external stylesheet)
- Embed code blocks with syntax highlighting
- Render Mermaid diagrams (use mermaid.js CDN)
- Insert generated image
- **Test**: Verify HTML structure, validate syntax highlighting

### Step 6.3: Output Saver
**File**: `src/utils/file_handler.py` (method)
- Save final HTML to `outputs/final/{topic-slug}.html`
- Save markdown draft to `outputs/drafts/{topic-slug}.md`
- Save metadata (scores, tags, etc.) to JSON
- **Test**: Verify file creation and paths

---

## **Phase 7: Orchestration & CLI** 🎼

### Step 7.1: Main Orchestrator
**File**: `src/main.py`
- CLI interface to accept YAML input path
- Execute pipeline steps sequentially:
  1. Parse YAML
  2. Research (search + extract)
  3. Generate content
  4. Generate code/diagrams
  5. Combine blog post
  6. Generate title/tags/meta
  7. Optimize SEO
  8. Quality check (with refinement loop)
  9. Generate image (if score ≥ 8)
  10. Convert to HTML
  11. Save output
- Display progress to stdout with verbose logging
- Handle errors gracefully with rollback
- **Test**: End-to-end integration test with fixture

### Step 7.2: CLI Enhancements
- Add flags: `--input`, `--output-dir`, `--max-iterations`, `--skip-image`
- Add `--dry-run` mode for testing
- Add `--verbose` for debug logging
- **Test**: Verify CLI argument parsing

---

## **Phase 8: Testing & Documentation** 🧪

### Step 8.1: Unit Tests
- Test each module independently with mocks
- Aim for 80%+ code coverage
- Run tests in `.venv` environment
- Use `pytest -v` for verbose output

### Step 8.2: Integration Tests
- End-to-end test with sample YAML input
- Verify complete pipeline execution
- Check output file quality

### Step 8.3: Documentation
- Write usage guide in README
- Document YAML format with examples
- Add docstrings to all functions
- Create example blog post

---

## **Implementation Order** 📅

**Week 1**: Phase 0 + Phase 1 (Setup, Config, Parsers)  
**Week 2**: Phase 2 (Research agents)  
**Week 3**: Phase 3 (Content generation)  
**Week 4**: Phase 4 + Phase 5 (Optimization, Quality, Refinement)  
**Week 5**: Phase 6 (Media, Conversion, Output)  
**Week 6**: Phase 7 (Orchestration, CLI)  
**Week 7**: Phase 8 (Testing, Documentation, Polish)  

---

## **Key Design Decisions** 🎯

1. **Modularity**: Each file ≤ 200-300 lines, single responsibility
2. **Testability**: All external APIs mocked in tests
3. **Error Handling**: Graceful degradation, retry logic
4. **Logging**: Verbose stdout for user visibility
5. **Environment Awareness**: Dev/test/prod configurations
6. **DRY Principle**: Shared utilities in `src/utils/`
7. **Type Safety**: Use type hints throughout

---

## **Next Steps** ▶️

Would you like me to:
1. **Start implementing Phase 0** (project structure & setup)?
2. **Create a detailed requirements.txt** with all dependencies?
3. **Generate sample YAML input** for testing?
4. **Begin with a specific module** (e.g., YAML parser)?

Let me know which phase you'd like to tackle first! 🚀