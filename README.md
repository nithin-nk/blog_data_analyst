# Blog Data Analyst

AI-powered blog content generation and analysis agent that researches topics, generates high-quality content, and optimizes for SEO.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- API Keys:
  - Google Gemini API key
  - OpenAI API key (for dual LLM quality checking)
  - Banana API key (for image generation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nithin-nk/blog_data_analyst.git
   cd blog_data_analyst
   ```

2. **Create virtual environment**
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Usage

1. **Create a YAML input file** in the `inputs/` directory (see `inputs/example_input.yaml`)

2. **Run the blog generator**
   ```bash
   python -m src.main --input inputs/your_input.yaml
   ```

3. **Find your generated blog** in `outputs/<blog-name>/final/`

### Command Line Options

```bash
python -m src.main --help

Options:
  -i, --input PATH          Path to YAML input file [required]
  -o, --output-dir PATH     Output directory for generated content
  --max-iterations INTEGER  Maximum refinement iterations (default: 3)
  --skip-image              Skip image generation
  --dry-run                 Test mode without API calls
  -v, --verbose             Enable verbose logging
  --help                    Show this message and exit
```

## 📝 YAML Input Format

Create a YAML file with your blog topic and outline:

## 📝 YAML Input Format

Create a YAML file with your blog topic and outline:

```yaml
topic: "Your Blog Topic Here"

outline:
  - "Introduction question or section"
  - "Key concept to explain"
  - "Code: Example with code snippet"
  - "Mermaid: Visual diagram section"
  - "Another important section"
  - "Conclusion"

metadata:
  target_audience: "Your target audience"
  difficulty: "Beginner/Intermediate/Advanced"
```

**Special Markers:**
- `Code:` - Generates a code snippet for this section
- `Mermaid:` - Generates a Mermaid diagram for visualization

See `inputs/example_input.yaml` for a complete example.

## 🏗️ Project Structure

```
blog_data_analyst/
├── src/
│   ├── config/          # Environment configuration
│   ├── parsers/         # YAML input parsing
│   ├── research/        # Web search & content extraction
│   ├── generation/      # LLM content generation
│   ├── optimization/    # SEO & quality checking
│   ├── refinement/      # Iterative improvement
│   ├── media/           # Image generation
│   ├── converters/      # Markdown to HTML
│   └── utils/           # Utilities (logging, file handling)
├── tests/               # Test suite
├── inputs/              # YAML input files
├── outputs/             # Generated blog posts
│   ├── drafts/
│   ├── final/
│   └── images/
└── pyproject.toml       # Project metadata & dependencies
```

## 🔄 Blog Generation Pipeline

1. **User Input**: The user provides the blog idea and outline in YAML format with questions.
   
   **Example:**
   ```yaml
   Topic: How to build memory for AI agents using the open source framework mem0?
   
   Outline:
     1. What is memory for AI agents?
     2. Why AI agents need memory with an example?
     3. Why we cannot use traditional databases for memory for AI agents?
     4. What is the architecture for mem0 memory?
        - Mermaid: Architectural diagram 
     5. Why we need vector database for storing memory for mem0?
     6. Why we need graph database for mem0?
     7. How to extract custom memories using mem0?
        - Code: custom prompt
     8. Create different memory layers for different sub-agents in mem0?
     9. How to incorporate extracted memory in an AI agent prompt with an example?
        - Code: Prompt for a sample AI agent
     10. How will the memory layer improve the AI agents?
     11. Conclusion
   ```

2. **Web Research**: Get web URLs by performing a Google search for different topics in the outline
   - Tool: [browser-use](https://github.com/browser-use/browser-use)

3. **Content Extraction**: Extract the contents from the web pages
   - Tool: [crawl4ai](https://github.com/unclecode/crawl4ai)

4. **Content Generation**: AI Agent generates content for all subtopics with citations

5. **Blog Compilation**: AI Agent generates the combined blog post for all topics

6. **Metadata Generation**: AI generates catchy title, tags, and search description

7. **Assets Creation**: AI agent generates code snippets and mermaid diagrams

8. **Content Refinement**: AI agent refines the blog post to make it a cohesive unit

9. **SEO Optimization**:
   - Keyword density optimization
   - Internal/external linking suggestions
   - Meta description optimization
   - Readability scoring (Flesch-Kincaid, etc.)
   - Header hierarchy validation (H1→H2→H3)

10. **Quality Check**: Check the quality of the blog post (score out of 10) with reasons for improvement using two LLMs

11. **Iteration**: If score is less than 7, go back to step 8 to implement feedback (with iteration limit)

12. **Image Generation**: If score is above 8, generate an image for the blog post using Banana

13. **Format Conversion**: Convert the blog post from Markdown to HTML

14. **Save Output**: Save the final output as an HTML file

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_parsers.py -v
```

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## 📊 Features

✅ Project structure initialized  
✅ All modules scaffolded with docstrings  
✅ Configuration system with environment variables  
✅ Virtual environment with dependencies  
✅ Test framework setup  
✅ Example YAML inputs  

**Status**: Phase 0 Complete - Ready for Phase 1 implementation. See `plan.md` for next steps.
