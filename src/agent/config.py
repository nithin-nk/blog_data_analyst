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
LLM_MODEL_FULL = "gemini-3-flash-preview"
LLM_TEMPERATURE_LOW = 0.3
LLM_TEMPERATURE_MEDIUM = 0.7


def get_llm_config(
    model_env_var: str,
    temp_env_var: str,
    default_model: str,
    default_temp: float,
) -> tuple[str, float]:
    """
    Get LLM model and temperature from environment variables or defaults.

    Args:
        model_env_var: Environment variable name for model (e.g., "TOPIC_DISCOVERY_MODEL")
        temp_env_var: Environment variable name for temperature (e.g., "TOPIC_DISCOVERY_TEMPERATURE")
        default_model: Default model if env var not set
        default_temp: Default temperature if env var not set

    Returns:
        Tuple of (model_name, temperature)

    Examples:
        >>> get_llm_config("TOPIC_DISCOVERY_MODEL", "TOPIC_DISCOVERY_TEMPERATURE",
        ...                LLM_MODEL_LITE, LLM_TEMPERATURE_MEDIUM)
        ('gemini-2.5-flash-lite', 0.7)  # Uses defaults if .env missing

        >>> # With .env: TOPIC_DISCOVERY_MODEL=gemini-2.0-flash-exp
        >>> get_llm_config("TOPIC_DISCOVERY_MODEL", "TOPIC_DISCOVERY_TEMPERATURE",
        ...                LLM_MODEL_LITE, LLM_TEMPERATURE_MEDIUM)
        ('gemini-2.0-flash-exp', 0.7)  # Uses custom model, default temp
    """
    import os

    model = os.getenv(model_env_var, default_model)
    temp_str = os.getenv(temp_env_var)
    temperature = float(temp_str) if temp_str else default_temp

    return model, temperature


# Gemini API Pricing (per 1M tokens, USD)
GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.02, "output": 0.08},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
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
- 10-18 words per sentence (target range)
- Average 12-16 words across each section
- One complete thought per sentence
- No semicolons (use natural connectors like 'because', 'when', 'which' instead)
- Avoid deeply nested clauses
- Active voice whenever possible
- Use fragments when natural ("Still, caching is important.")
- Combine related ideas with conjunctions rather than breaking into choppy fragments

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

✅ GOOD (Punchy, complete thought, 15 words):
"Caching is important because LLM agents take time to run and inference is expensive."

❌ BAD (Formal, 24 words):
"Traditional caching mechanisms prove ineffective due to their reliance on exact string matching, which is fundamentally ill-suited to natural language inputs."

✅ GOOD (Punchy, complete thought, 16 words):
"Traditional caching fails because it relies on exact matching, which rarely works for natural language."

❌ BAD (Academic, 19 words):
"The implementation leverages Redis with RediSearch to efficiently store and retrieve semantically similar query embeddings."

✅ GOOD (Punchy, complete thought, 12 words):
"Use Redis with RediSearch to store and find similar embeddings fast."

When discussing tradeoffs, be honest about drawbacks. Keep it punchy.
"""
