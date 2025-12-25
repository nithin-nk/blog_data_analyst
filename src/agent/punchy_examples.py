"""
Punchy writing examples for each section role.
Used by writer and critic prompts to demonstrate the target style.
"""

PUNCHY_EXAMPLES_BY_ROLE = {
    "hook": """
❌ BAD (Formal, wordy):
"In today's rapidly evolving landscape of artificial intelligence applications, the challenge of maintaining optimal response times while simultaneously managing escalating computational costs has emerged as a critical concern for organizations deploying large language model systems at scale."

✅ GOOD (Punchy, direct):
"A 3-second response time in an AI application isn't a minor UX inconvenience. It's an abandonment trigger. For LLM-powered systems, balancing model accuracy with sub-second latency presents a significant challenge."
""",

    "problem": """
❌ BAD (Academic, passive voice):
"Traditional caching mechanisms prove ineffective due to their reliance on exact string matching, which is fundamentally ill-suited to the inherent variability present in natural language inputs. This results in suboptimal cache hit rates and leads to redundant computational overhead."

✅ GOOD (Punchy, active):
"Traditional caching fails. It works on exact matching. Natural language rarely matches exactly. Result: cache misses everywhere. Redundant computation. Wasted money."
""",

    "why": """
❌ BAD (Vague benefits):
"Implementing semantic caching provides numerous advantages including improved performance characteristics, enhanced cost efficiency, and better resource utilization across distributed systems."

✅ GOOD (Specific, punchy):
"Semantic caching cuts API calls by 40-80%. Response times drop from 3 seconds to 300ms. Infrastructure costs decrease 30-70%. Simple. Effective."
""",

    "implementation": """
❌ BAD (Long narrative before code):
"In order to implement semantic caching effectively, we must first establish a connection to our Redis instance and configure the appropriate data structures. The following implementation demonstrates how to leverage Redis's vector search capabilities in conjunction with OpenAI's embedding API to create a robust caching layer."

✅ GOOD (Minimal intro, let code speak):
"Here's the Redis setup. Read the code. Copy it. It works.

```python
import redis
import numpy as np

r = redis.Redis(host="localhost", port=6379)
# ... rest of code
```

Store embeddings as bytes. TTL controls freshness."
""",

    "deep_dive": """
❌ BAD (Theoretical, complex):
"The theoretical underpinnings of semantic similarity measurement involve vector space representations wherein embeddings are positioned such that semantically related concepts exhibit reduced cosine distances, thereby enabling efficient retrieval through approximate nearest neighbor search algorithms."

✅ GOOD (Practical, concrete):
"Semantic search works like this. Convert query to embedding. Find similar embeddings in Redis. Return cached result if distance < threshold. Skip if distance too high. Simple vector math."
""",

    "tradeoffs": """
❌ BAD (Sugar-coating):
"While semantic caching offers significant benefits, practitioners should be mindful of certain considerations that may arise during implementation."

✅ GOOD (Honest, direct):
"Semantic caching has real drawbacks. Every query adds 100-400ms for embedding generation. Cache misses hurt more than no cache. Similarity threshold is hard to tune. Get it wrong and you serve bad answers."
""",

    "conclusion": """
❌ BAD (Repetitive summary):
"In conclusion, semantic caching represents a powerful optimization technique that leverages vector similarity to reduce redundant LLM invocations, thereby improving response times and reducing costs, as we have explored throughout this article."

✅ GOOD (Action-oriented, brief):
"Start simple. Add Redis with RediSearch. Generate embeddings. Store results. Tune your threshold. Monitor hit rates. You'll cut costs and speed up responses. If you need help, comment below."
"""
}


def get_example_for_role(role: str) -> str:
    """
    Get the punchy writing example for a given section role.

    Args:
        role: Section role (hook, problem, why, implementation, deep_dive, tradeoffs, conclusion)

    Returns:
        String containing BAD → GOOD examples, or empty string if role not found
    """
    return PUNCHY_EXAMPLES_BY_ROLE.get(role, "")
