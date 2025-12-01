# Blog Reviewer Agent

The Blog Reviewer Agent is a multi-model review system that evaluates blog content quality using three different LLM models simultaneously: Gemini 2.5 Pro, Gemini 2.5 Flash, and GPT-5-chat (Azure OpenAI).

## Overview

The reviewer implements an iterative improvement loop that:
1. Reviews content with 3 models in parallel
2. Calculates the average score across all models
3. If score > 9, saves the final content to `final/` directory as HTML
4. If score ≤ 9 and feedback can be applied, regenerates content with combined feedback
5. Continues until score > 9 or max iterations (5) reached
6. If threshold never met, selects the version with the highest average score

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Blog Reviewer Agent                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│   │ Gemini 2.5  │  │ Gemini 2.5  │  │ GPT-5-chat          │     │
│   │ Pro         │  │ Flash       │  │ (Azure OpenAI)      │     │
│   └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘     │
│          │                │                     │                │
│          └────────────────┴─────────────────────┘                │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │  Aggregate  │                              │
│                    │  Results    │                              │
│                    └──────┬──────┘                              │
│                           │                                      │
│              ┌────────────▼────────────┐                        │
│              │  Average Score > 9?     │                        │
│              └────────────┬────────────┘                        │
│                    Yes    │    No                                │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                         │
│    ┌─────────────────┐      ┌──────────────────┐               │
│    │ Save to final/  │      │ Regenerate with  │               │
│    │ as HTML         │      │ feedback         │               │
│    └─────────────────┘      └──────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

Add the following to your `.env` file:

```env
# Azure OpenAI Configuration (for GPT-5-chat)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_API_VERSION=2025-01-01-preview
AZURE_DEPLOYMENT_NAME=gpt-5-chat

# Google API Keys (for Gemini models)
GOOGLE_API_KEY=your_google_key
GOOGLE_API_KEY_1=your_backup_key_1
GOOGLE_API_KEY_2=your_backup_key_2
```

### Settings in `settings.py`

```python
# Blog Reviewer Settings
blog_reviewer_max_iterations: int = 5
blog_reviewer_score_threshold: float = 9.0
```

## Usage

### Via CLI

```bash
# Generate blog with automatic review
python -m src.main generate --input inputs/my_blog.yaml

# Skip review (dry run)
python -m src.main generate --input inputs/my_blog.yaml --dry-run
```

### Programmatic Usage

```python
import asyncio
from src.refinement.blog_reviewer import BlogReviewer

reviewer = BlogReviewer()

# Review content with all models
result = asyncio.run(reviewer.review_with_all_models(
    title="My Blog Title",
    content="# My Blog Content..."
))

print(f"Average Score: {result.average_score}")
print(f"Passes Threshold: {result.passes_threshold}")
print(f"Feedback: {result.combined_feedback}")

# Full review and improve loop
final_content, final_review, history = asyncio.run(reviewer.review_and_improve(
    title="My Blog Title",
    content="# My Blog Content...",
    max_iterations=5,
    score_threshold=9.0,
))
```

## Review Criteria

Each model evaluates the blog based on:

| Criteria | Weight | Description |
|----------|--------|-------------|
| Content Quality | 30% | Comprehensive, valuable, well-organized |
| E-E-A-T | 25% | Experience, Expertise, Authoritativeness, Trustworthiness |
| Style & Readability | 20% | Concise, clear, well-formatted |
| Technical Accuracy | 15% | Correct code examples, accurate concepts |
| SEO & Discoverability | 10% | Clear title, natural keywords |

### Scoring Guidelines

- **9-10**: Publication-ready, exceptional quality
- **7-8**: Good quality, minor improvements needed
- **5-6**: Acceptable, needs significant work
- **3-4**: Poor quality, major rework required
- **1-2**: Fundamentally flawed

## Output Structure

### Review History

Saved to `outputs/<topic>/review/review_history.yaml`:

```yaml
total_iterations: 3
iterations:
  - iteration: 1
    average_score: 7.5
    passed_threshold: false
    models:
      - name: gemini-2.5-pro
        score: 8.0
        feedback_count: 3
        can_apply_feedback: true
        error: null
      - name: gemini-2.5-flash
        score: 7.5
        feedback_count: 4
        can_apply_feedback: true
        error: null
      - name: gpt-5-chat
        score: 7.0
        feedback_count: 5
        can_apply_feedback: true
        error: null
    combined_feedback:
      - "Add more code examples"
      - "Improve introduction clarity"
      - "Include real-world use cases"
    can_apply_feedback: true
```

### Final Output

When score > 9:
- `outputs/<topic>/final/<topic>.html` - Final HTML blog post
- `outputs/<topic>/final/<topic>.md` - Final markdown version

When score ≤ 9:
- Content remains in `outputs/<topic>/drafts/<topic>.md`
- Review history shows improvement attempts

## Console Output

The reviewer provides detailed console output showing:

```
======================================================================
📝 BLOG REVIEW & REFINEMENT PHASE
======================================================================
Models: Gemini 2.5 Pro, Gemini 2.5 Flash, GPT-5-chat
Score Threshold: > 9.0
Max Iterations: 5
======================================================================

============================================================
📝 REVIEW ITERATION 1/5
============================================================
🔍 Starting multi-model review...
   ├─ Gemini 2.5 Pro: reviewing...
   ├─ Gemini 2.5 Flash: reviewing...
   └─ GPT-5-chat: reviewing...

📊 Review Results:
   ├─ gemini-2.5-pro: ⚠️ Score 7.5/10
   ├─ gemini-2.5-flash: ⚠️ Score 7.0/10
   ├─ gpt-5-chat: ⚠️ Score 8.0/10
   └─ Average Score: ⚠️ 7.50/10

📋 Applying 8 feedback items...
🔄 Regenerating content with feedback...
✅ Content regenerated successfully

============================================================
📝 REVIEW ITERATION 2/5
============================================================
...

🎉 SUCCESS! Score 9.23 > 9.0
   Blog approved after 2 iteration(s)
```

## Error Handling

- If one model fails, the average is calculated from successful models only
- If all models fail, the original content is preserved
- Rate limit errors trigger automatic retry with different API keys
- Regeneration failures preserve the original content

## Testing

Run the unit tests:

```bash
pytest tests/test_blog_reviewer.py -v
```

Test coverage includes:
- Individual model review functionality
- Aggregated results calculation
- Iteration loop logic
- Best version selection
- Review history saving
- Content regeneration with feedback
