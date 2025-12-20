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

# Target word counts
TARGET_WORDS_MAP = {
    "short": 800,
    "medium": 1500,
    "long": 2500,
}

# Writing style guide
STYLE_GUIDE = """
Write in a direct, technical style for experienced engineers.
- Open with a clear problem statement, no warm-up.
- Be opinionated. Say "you need X" not "you might consider".
- Use short sentences. Keep paragraphs short (2-4 sentences).
- Use bullet points liberally to improve readability.
- Include specific tool names and real config examples.
- No fluff: "In today's world", "It's worth noting that".
- Address the reader as "you".
- Code: Python preferred, include imports, be runnable.
- YAML for configuration examples.
"""
