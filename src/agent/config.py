"""Configuration constants for the blog agent."""

# Research settings
MAX_URLS_PER_QUERY = 3
MAX_SOURCES_PER_SECTION = 10

# Validation settings
MIN_SOURCES_PER_SECTION = 4
MAX_VALIDATION_RETRIES = 3

# Retry limits
MAX_SECTION_RETRIES = 2  # Section refinement retries

# Dynamic research for refine loop
MAX_RESEARCH_QUERIES_PER_ISSUE = 2  # Queries per completeness issue
MAX_RESEARCH_URLS_PER_QUERY = 3  # URLs per query (6 total sources)

# Query diversification modifiers for retries
# Each retry uses a different focus to find varied sources
QUERY_DIVERSIFICATION_MODIFIERS = [
    ["tutorial", "guide", "how to"],
    ["documentation", "official docs", "API reference"],
    ["benchmark", "comparison", "performance"],
    ["case study", "real-world", "production"],
    ["example", "sample code", "implementation"],
]

# LLM settings
LLM_MODEL_LITE = "gemini-2.5-flash-lite"
LLM_MODEL_FULL = "gemini-2.5-flash"
LLM_TEMPERATURE_LOW = 0.3
LLM_TEMPERATURE_MEDIUM = 0.7

# Gemini API Pricing (per 1M tokens, USD)
GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.02, "output": 0.08},
}

# Target word counts (adjusted for punchy style - shorter than formal writing)
TARGET_WORDS_MAP = {
    "short": 500,
    "medium": 1000,
    "long": 1500,
}

# Originality check threshold for short sentences (prevent false positives)
ORIGINALITY_THRESHOLD_SHORT_SENTENCES = 0.85  # vs 0.7 for longer sentences

# Writing style guide
STYLE_GUIDE = """
Write short, punchy sentences for experienced engineers. Talk like you're explaining to a colleague over coffee.

SENTENCE STRUCTURE (HARD REQUIREMENTS):
- 5-15 words per sentence (STRICT constraint)
- Average 10-14 words across each section
- One idea per sentence
- No semicolons (they create compound thoughts)
- No nested clauses
- Active voice whenever possible
- Use fragments when natural ("Still, caching is important.")

LANGUAGE:
- Simple words. Be direct.
- "This is easy" NOT "This presents a straightforward approach"
- "Fails" NOT "proves ineffective due to"
- "Use" NOT "leverage" or "utilize"
- No fluff: "In today's world", "It's worth noting that", "It's important to"
- No preachy phrases: "you must", "non-negotiable", "critical"

CODE:
- Code is 50-70% of implementation sections
- PRIMARY content, not afterthoughts
- Minimal comments - let code speak
- Self-explanatory variable names
- Include imports, make it copy-pasteable
- Python preferred, YAML for configs

FORMATTING:
- Short paragraphs (2-4 sentences max)
- Bullet points for lists
- One blank line between ideas
- Open with clear problem, no warm-up

BAD → GOOD EXAMPLES:

❌ BAD (Formal, 28 words):
"Caching presents a crucial optimization opportunity because LLM agents typically require significant execution time and inference operations incur substantial costs."

✅ GOOD (Punchy, 3 sentences, 15 total words):
"Caching is important. LLM agents take time to run. Inference is expensive."

❌ BAD (Formal, 24 words):
"Traditional caching mechanisms prove ineffective due to their reliance on exact string matching, which is fundamentally ill-suited to natural language inputs."

✅ GOOD (Punchy, 3 sentences, 15 total words):
"Traditional caching fails. It works on exact matching. Natural language rarely matches exactly."

❌ BAD (Academic, 19 words):
"The implementation leverages Redis with RediSearch to efficiently store and retrieve semantically similar query embeddings."

✅ GOOD (Punchy, 2 sentences, 11 total words):
"Use Redis with RediSearch. It stores and finds similar embeddings fast."

When discussing tradeoffs, be honest about drawbacks. Keep it punchy.
"""
