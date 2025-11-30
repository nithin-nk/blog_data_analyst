# 📋 Step-by-Step Implementation Plan for Blog Data Analyst AI Agent

Based on your README, here's a comprehensive implementation plan following clean architecture principles:

## **Pipeline Overview** 🔄

The system operates in two main phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE A: RESEARCH & OUTLINE GENERATION                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Input          LLM Question        Google Search      Content         │
│  (Title + Context) → Generation (10) →   (crawl4ai) →      Extraction →    │
│                                          (top 3 URLs each)  (crawl4ai)      │
│                                                                              │
│  → Subtopic Analysis → Outline Generation → LLM Quality Check →             │
│    (from content)      (with Code/Mermaid)   (score/10)                     │
│                                                                              │
│  → If score < 8: LLM feedback loop (max 3 iterations)                       │
│  → If score ≥ 8: Save YAML → Human Review → Approve/Edit                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE B: BLOG CONTENT GENERATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  YAML Outline → Content Generation → Code/Diagram Generation →              │
│  Blog Compilation → SEO Optimization → Dual LLM Quality Check →             │
│  Refinement Loop → Image Generation → HTML Conversion → Save Output         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **Phase 0: Project Setup & Architecture** 🏗️

### Step 0.1: Initialize Project Structure
```
blog_data_analyst/
├── .env.example                    # Environment variables template
├── .env                           # Actual secrets (git-ignored)
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
│   ├── planning/                  # NEW: Research & Outline Generation
│   │   ├── __init__.py
│   │   ├── question_generator.py  # LLM generates research questions
│   │   ├── outline_generator.py   # Generate blog outline from research
│   │   └── outline_reviewer.py    # LLM quality check for outline
│   ├── research/
│   │   ├── __init__.py
│   │   ├── search_agent.py        # Google Search via crawl4ai
│   │   └── content_extractor.py   # crawl4ai content extraction
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
│   ├── test_planning.py           # NEW: Tests for planning module
│   ├── test_research.py
│   ├── test_generation.py
│   ├── test_optimization.py
│   ├── test_converters.py
│   └── fixtures/
│       └── sample_input.yaml      # Test data
├── inputs/                        # Generated YAML files (for human review)
├── outputs/<blog_name>/           # Generated blog posts
│   ├── research/                  # NEW: Saved research data
│   ├── drafts/
│   ├── final/
│   └── images/
└── README.md
```

### Step 0.2: Core Dependencies
- **LLM Framework**: `langchain-google-genai`, `langchain-openai`
- **Web Research & Content Extraction**: `crawl4ai` (Google search + content extraction)
- **YAML Parsing**: `pyyaml`
- **SEO Analysis**: `textstat`, `spacy`
- **HTML Conversion**: `markdown`, `beautifulsoup4`
- **Image Generation**: API client for Banana/Nano
- **Testing**: `pytest`, `pytest-mock`, `pytest-asyncio`
- **Code Quality**: `black`, `ruff`, `mypy`

---

## **Phase 1: Foundation Components** 🔧

### Step 1.1: Configuration & Environment Setup ✅ COMPLETE
**File**: `src/config/settings.py`
- Load environment variables (API keys for LLMs, Banana, etc.)
- Define environment modes (`dev`, `test`, `prod`)
- Set file paths, rate limits, retry configs
- **Test**: Verify config loading in all environments

### Step 1.2: Logging & File Handlers ✅ COMPLETE
**Files**: `src/utils/logger.py`, `src/utils/file_handler.py`
- Structured logging with colors to stdout
- Safe file I/O with error handling
- Research data storage
- **Test**: Log output verification, file write/read tests

### Step 1.3: YAML Parser ✅ COMPLETE
**File**: `src/parsers/yaml_parser.py`
- Parse YAML structure: `topic`, `outline` (list of questions)
- Detect special markers: `Mermaid:`, `Code:`
- Validate required fields
- Return structured data model (Pydantic)
- **Test**: Parse valid/invalid YAML fixtures

---

## **Phase 2: Research Question Generation** 🔍 NEW

### Step 2.1: Question Generator ✅ COMPLETE
**File**: `src/planning/question_generator.py`
- Accept user input: **Title** + **Context** (optional description/constraints)
- Use LLM (Gemini 2.0 Flash) to generate **10 research questions** for Google search
- Questions are Google-optimized search queries (not full questions)
- Uses structured output (Pydantic) for consistent format
- Async implementation with sync wrapper
- Questions cover relevant categories:
  - Core concepts ("What is X?")
  - Why/rationale ("Why do we need X?")
  - How-to/implementation ("How to implement X?")
  - Architecture/design ("What is the architecture of X?")
  - Best practices ("Best practices for X")
  - Comparisons ("X vs Y")
  - Real-world examples ("Examples of X in production")
- Return list of search queries
- **Test**: Mock LLM, verify 10 diverse questions generated

### Step 2.2: Google Search Agent
**File**: `src/research/search_agent.py`
- Integrate `crawl4ai` library for Google Search
- For each of 10 questions, execute Google search using crawl4ai
- Extract **top 3 URLs** per question (30 total max)
- Deduplicate URLs across questions
- Rate limiting and error handling
- **Test**: Mock crawl4ai search responses, verify URL extraction

### Step 2.3: Web Content Extractor
**File**: `src/research/content_extractor.py`
- Integrate `crawl4ai` library
- Extract clean text, headings, code blocks from URLs
- Handle extraction failures gracefully
- **Save all research data** to `outputs/<blog>/research/`:
  - `research_questions.json` - Generated questions
  - `search_results.json` - URLs per question
  - `extracted_content/` - Content from each URL
- **Test**: Mock crawl4ai responses, test parser robustness

---

## **Phase 3: Outline Generation & Review** 📝 NEW

### Step 3.2: Outline Generator
**File**: `src/planning/outline_generator.py`
- Pass generated titles, snippets, headings, and source queries to the LLM.
- LLM generates a structured blog outline with:
  - Subheadings for each section
  - Summary of what each section should contain
  - Reference links for each section (from research data)
- Output YAML example:
  ```yaml
  topic: "How to build memory for AI agents using mem0"
  outline:
    - heading: "What is memory for AI agents?"
      summary: "Explain the concept of memory in AI agents, why it's important, and provide real-world context."
      references:
        - "https://example.com/article1"
        - "https://example.com/article2"
    - heading: "Architecture of mem0 memory system"
      summary: "Describe the architecture using diagrams and code snippets."
      references:
        - "https://example.com/diagram"
    ...
  metadata:
    target_audience: "AI developers"
    difficulty: "Intermediate"
    estimated_reading_time: "15 minutes"
  ```
- Ensure logical flow: Introduction → Core concepts → Implementation → Conclusion
- **Test**: Verify outline structure, reference inclusion, summary quality

### Step 3.3: Outline Quality Reviewer
**File**: `src/planning/outline_reviewer.py`
- Use LLM to review generated outline
- Score 1-10 on:
  - **Completeness**: Does it cover the topic thoroughly?
  - **Logical Flow**: Is the order sensible?
  - **Depth**: Are subtopics specific enough?
  - **Balance**: Mix of theory, code, visuals?
  - **Audience Fit**: Appropriate for target audience?
- If score < 8:
  - Extract specific feedback
  - Regenerate outline with improvements
  - Max 3 iterations
- If score ≥ 8:
  - Save YAML to `inputs/{topic-slug}.yaml`
  - Prompt for human review
- **Test**: Mock review responses, verify iteration logic

### Step 3.4: Human Review Interface
**File**: `src/planning/outline_reviewer.py` (method)
- Save generated YAML to file
- Display outline summary in terminal
- Prompt user: "Review the outline at `inputs/{topic}.yaml`"
- Wait for user confirmation:
  - User edits file if needed
  - User runs command to continue: `--approve` flag
- Load (potentially modified) YAML and proceed to Phase B
- **Test**: Verify file save/load, CLI prompts

---

## **Phase 4: Content Generation** ✍️

### Step 4.1: Subtopic Content Generator
**File**: `src/generation/content_generator.py`
- Load approved YAML outline
- For each outline question:
  - Retrieve relevant research data from saved files
  - Use LLM to generate 300-500 word section
  - Include inline citations `[Source Name](URL)`
  - Maintain consistent tone and style
- **Test**: Mock LLM responses, verify citation format

### Step 4.2: Code & Diagram Generator
**File**: `src/generation/code_generator.py`
- Detect `Code:` and `Mermaid:` markers in outline
- For `Code:` sections:
  - Analyze research content for code patterns
  - Generate relevant, working code snippets
  - Add explanatory comments
- For `Mermaid:` sections:
  - Generate Mermaid diagram syntax
  - Validate syntax before inclusion
- **Test**: Syntax validation for code and diagrams

### Step 4.3: Blog Post Combiner
**File**: `src/generation/content_generator.py` (method)
- Merge all subtopic sections into cohesive post
- Add transitions between sections
- Ensure narrative flow
- Insert code blocks and diagrams at marked positions
- **Test**: Verify section ordering and transitions

### Step 4.4: Title, Tags & Meta Generator
**File**: `src/generation/title_generator.py`
- Generate 3-5 catchy title options
- Extract 8-12 relevant tags/keywords
- Write compelling meta description (150-160 char)
- **Test**: Verify length constraints, keyword relevance

---

## **Phase 5: Optimization & Quality Control** 🎯

### Step 5.1: SEO Optimizer
**File**: `src/optimization/seo_optimizer.py`
- **Keyword Density**: Calculate and suggest optimal placement
- **Readability**: Flesch-Kincaid, Gunning Fog scores
- **Header Hierarchy**: Validate H1→H2→H3 structure
- **Internal/External Links**: Suggest relevant links
- **Meta Tags**: Validate title, description, keywords
- Return structured suggestions
- **Test**: Analyze sample blog posts, verify scoring

### Step 5.2: Dual LLM Quality Checker
**File**: `src/optimization/quality_checker.py`
- Use two different LLMs (Gemini + GPT-4)
- Each rates blog 1-10 on:
  - Accuracy, Clarity, Engagement, Structure, Completeness
- Extract specific improvement suggestions
- Average scores from both LLMs
- **Test**: Mock LLM responses, verify scoring logic

---

## **Phase 6: Refinement Loop** 🔄

### Step 6.1: Blog Refiner
**File**: `src/refinement/blog_refiner.py`
- Accept feedback from quality checker
- Use LLM to apply specific improvements
- Track iteration count (max 3 iterations)
- Re-check quality after each iteration
- Break loop if score ≥ 8 or max iterations reached
- **Test**: Mock feedback loop, verify iteration limits

---

## **Phase 7: Media & Output Generation** 🎨

### Step 7.1: Image Generator
**File**: `src/media/image_generator.py`
- Generate prompt from blog topic and key themes
- Call Banana/Nano API for image generation
- Download and save image locally
- Return image path
- **Test**: Mock Banana API, verify image download

### Step 7.2: Markdown to HTML Converter
**File**: `src/converters/md_to_html.py`
- Convert markdown to HTML with proper formatting
- Apply CSS styling (include inline or external stylesheet)
- Embed code blocks with syntax highlighting
- Render Mermaid diagrams (use mermaid.js CDN)
- Insert generated image
- **Test**: Verify HTML structure, validate syntax highlighting

### Step 7.3: Output Saver
**File**: `src/utils/file_handler.py` (method)
- Save final HTML to `outputs/<blog>/final/{topic-slug}.html`
- Save markdown draft to `outputs/<blog>/drafts/{topic-slug}.md`
- Save metadata (scores, tags, etc.) to JSON
- Keep research data for reference
- **Test**: Verify file creation and paths

---

## **Phase 8: Orchestration & CLI** 🎼

### Step 8.1: Main Orchestrator
**File**: `src/main.py`
- Two CLI modes:

**Mode 1: Full Pipeline (from topic)**
```bash
python -m src.main generate --topic "How to build memory for AI agents using mem0" \
  --context "Focus on the open source framework, include Python examples"
```
Pipeline:
1. Generate research questions (10)
2. Google search (top 3 URLs each)
3. Extract content (crawl4ai)
4. Save research data
5. Generate outline with Code/Mermaid markers
6. LLM review outline (iterate if < 8)
7. Save YAML, prompt for human review
8. *Wait for approval*
9. Generate blog content
10. Generate code/diagrams
11. Combine blog post
12. Generate title/tags/meta
13. Optimize SEO
14. Dual LLM quality check
15. Refinement loop (if needed)
16. Generate image (if score ≥ 8)
17. Convert to HTML
18. Save all outputs

**Mode 2: Continue from YAML (after human review)**
```bash
python -m src.main generate --approve inputs/mem0-memory.yaml
```
- Skip steps 1-8, start from approved YAML

### Step 8.2: CLI Flags
```bash
python -m src.main generate --help

Options:
  --topic TEXT              Blog topic/title [required for new]
  --context TEXT            Additional context/constraints
  --approve PATH            Path to approved YAML (skip research phase)
  --output-dir PATH         Output directory
  --max-outline-iterations  Max outline refinement iterations (default: 3)
  --max-content-iterations  Max content refinement iterations (default: 3)
  --skip-image              Skip image generation
  --dry-run                 Test mode without API calls
  -v, --verbose             Debug logging
```

### Step 8.3: Progress Display
- Rich progress bars for each phase
- Display research questions as generated
- Show outline preview before human review
- Display quality scores at each stage
- **Test**: End-to-end integration test

---

## **Phase 9: Testing & Documentation** 🧪

### Step 9.1: Unit Tests
- Test each module independently with mocks
- Aim for 80%+ code coverage
- Key test scenarios:
  - Question generation variety
  - URL deduplication
  - Content extraction failures
  - Outline marker placement
  - Review iteration limits
  - Human review workflow

### Step 9.2: Integration Tests
- End-to-end test with sample topic
- Verify complete pipeline execution
- Check output file quality and structure

### Step 9.3: Documentation
- Update README with new workflow
- Document CLI modes with examples
- Add docstrings to all functions
- Create example blog post

---

## **Implementation Order** 📅

**Week 1**: ✅ Phase 0 + Phase 1 (Setup, Config, Parsers, Utils)  
**Week 2**: Phase 2 (Question Generation, Search, Extraction)  
**Week 3**: Phase 3 (Outline Generation, Review, Human Interface)  
**Week 4**: Phase 4 (Content Generation)  
**Week 5**: Phase 5 + Phase 6 (Optimization, Quality, Refinement)  
**Week 6**: Phase 7 (Media, Conversion, Output)  
**Week 7**: Phase 8 (Orchestration, CLI)  
**Week 8**: Phase 9 (Testing, Documentation, Polish)  

---

## **Key Design Decisions** 🎯

1. **Two-Phase Pipeline**: Research/Planning → Content Generation
2. **Human-in-the-Loop**: User reviews outline before content generation
3. **LLM Quality Gates**: Score ≥ 8 required to proceed
4. **Research Persistence**: All research data saved for reference/citations
5. **Automatic Marker Detection**: LLM decides Code/Mermaid placement
6. **Graceful Degradation**: Handle API failures, extraction errors
7. **Iteration Limits**: Max 3 attempts for outline, max 3 for content
8. **Modularity**: Each phase can run independently
9. **Type Safety**: Pydantic models throughout

---

## **Data Flow Summary** 📊

```
User Input (Title + Context)
    │
    ▼
┌─────────────────────────────┐
│ Question Generator (LLM)    │ → 10 research questions
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Search Agent (crawl4ai)     │ → 30 URLs (3 per question)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Content Extractor (crawl4ai)│ → Extracted text, code, headings
└─────────────────────────────┘
    │
    ▼ [SAVE: research/]
┌─────────────────────────────┐
│ Outline Generator (LLM)     │ → YAML with Code/Mermaid markers
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Outline Reviewer (LLM)      │ → Score ≥ 8? 
└─────────────────────────────┘
    │ No: iterate (max 3)
    │ Yes: ▼
┌─────────────────────────────┐
│ Human Review                │ → User approves/edits YAML
└─────────────────────────────┘
    │
    ▼ [SAVE: inputs/{topic}.yaml]
┌─────────────────────────────┐
│ Content Generator (LLM)     │ → Blog sections with citations
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Code/Diagram Generator      │ → Code snippets, Mermaid diagrams
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Blog Combiner               │ → Full markdown blog post
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ SEO Optimizer               │ → Keyword analysis, readability
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Quality Checker (2 LLMs)    │ → Score ≥ 8?
└─────────────────────────────┘
    │ No: refine (max 3)
    │ Yes: ▼
┌─────────────────────────────┐
│ Image Generator (Banana)    │ → Header image
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ HTML Converter              │ → Styled HTML with Mermaid.js
└─────────────────────────────┘
    │
    ▼ [SAVE: outputs/<blog>/]
    
Final Output:
├── research/          # All research data
├── drafts/            # Markdown versions
├── final/             # HTML output
├── images/            # Generated images
└── metadata.json      # Scores, tags, etc.
```

---

## **Next Steps** ▶️

Phase 1 is complete. Ready to implement **Phase 2: Research Question Generation**:

1. Create `src/planning/question_generator.py`
2. Implement `src/research/search_agent.py` with crawl4ai (Google Search)
3. Implement `src/research/content_extractor.py` with crawl4ai (content extraction)
4. Add research data storage to file handler

Let me know when you're ready to proceed! 🚀