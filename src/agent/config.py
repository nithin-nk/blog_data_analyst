"""Configuration constants for the blog agent."""

# Research settings
MAX_URLS_PER_QUERY = 3
MAX_SOURCES_PER_SECTION = 10

# Validation settings
MIN_SOURCES_PER_SECTION = 4
MAX_VALIDATION_RETRIES = 2

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
