**FINAL BLOG AGENT ARCHITECTURE - COMPLETE SPECIFICATION**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         BLOG AGENT v1.0 - FINAL                             │
│                                                                              │
│          "An AI agent that writes technical blogs like a human"             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT                                                                       │
│  ├── Title (required)                                                       │
│  ├── Context (required) - from Twitter, other blogs, notes                  │
│  ├── Target length: short (~800) | medium (~1500) | long (~2500 words)      │
│  └── Flags:                                                                 │
│      ├── --review-sections    Pause after each section for human review     │
│      ├── --review-final       Pause only before final output                │
│      ├── --review-all         Pause at every decision point                 │
│      ├── --no-citations       Disable citations (default: ON)               │
│      └── --no-hook            Skip hook generation                          │
│                                                                              │
│  OUTPUT                                                                      │
│  ├── final.md                 Publication-ready markdown                    │
│  ├── images/*.png             Rendered mermaid diagrams                     │
│  ├── metadata.json            Token usage, sources, job stats               │
│  └── fact_check.md            Claims flagged for human verification         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## SYSTEM COMPONENTS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFRASTRUCTURE LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         QUOTA MANAGER                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  Projects: 4 Gemini API keys (separate GCP projects)                │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │Project 1 │ │Project 2 │ │Project 3 │ │Project 4 │               │   │
│  │  │RPD: 45/250│ │RPD: 12/250│ │RPD: 0/250│ │RPD: 89/250│              │   │
│  │  │TPM: ok    │ │TPM: ok    │ │TPM: ok   │ │TPM: ok    │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  │                                                                      │   │
│  │  Selection Strategy:                                                 │   │
│  │  1. Pick project with most remaining RPD                            │   │
│  │  2. On 429 error → switch to next project                           │   │
│  │  3. All exhausted → pause job, notify human                         │   │
│  │                                                                      │   │
│  │  Tracking:                                                           │   │
│  │  • Log every request: timestamp, tokens_in, tokens_out, model       │   │
│  │  • Parse usageMetadata from each response                           │   │
│  │  • Reset counts at midnight Pacific Time                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       CHECKPOINT MANAGER                             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ~/.blog_agent/                                                     │   │
│  │  ├── config.yaml              # API keys, default settings          │   │
│  │  ├── usage/                                                         │   │
│  │  │   └── {date}.json          # Daily token usage per project       │   │
│  │  │                                                                   │   │
│  │  └── jobs/                                                          │   │
│  │      └── {job_id}/            # One folder per blog job             │   │
│  │          ├── state.json       # Current phase, progress, can_resume │   │
│  │          ├── input.json       # Original title + context            │   │
│  │          ├── topic_context.json # Discovery search results          │   │
│  │          ├── plan.json        # Outline, search queries, metadata   │   │
│  │          ├── research/                                              │   │
│  │          │   ├── cache/       # Raw fetched articles (for resume)   │   │
│  │          │   ├── validated/   # Post-quality-filter sources         │   │
│  │          │   └── sources.json # URL → section mapping for citations │   │
│  │          ├── drafts/                                                │   │
│  │          │   ├── sections/    # Individual section drafts           │   │
│  │          │   ├── v1.md        # Combined draft v1                   │   │
│  │          │   └── v2.md        # After refinement                    │   │
│  │          ├── feedback/                                              │   │
│  │          │   ├── section_{n}_critic.json                            │   │
│  │          │   └── final_critic.json                                  │   │
│  │          ├── human_inputs/    # Your guidance/feedback saved        │   │
│  │          ├── fact_check.md    # Claims to verify                    │   │
│  │          ├── images/          # Rendered mermaid PNGs               │   │
│  │          └── final.md         # Approved output                     │   │
│  │                                                                      │   │
│  │  State Machine:                                                      │   │
│  │  topic_discovery → planning → researching → validating_sources →    │   │
│  │  writing → reviewing → assembling → final_review → done | failed    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        RETRY MANAGER                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  gemini_api:                                                        │   │
│  │    max_retries: 3                                                   │   │
│  │    backoff: [1s, 2s, 4s]     # Exponential                         │   │
│  │    on_429: switch_project                                           │   │
│  │    on_all_exhausted: pause_notify_human                             │   │
│  │    on_500: retry_same_project                                       │   │
│  │                                                                      │   │
│  │  web_search:                                                        │   │
│  │    max_retries: 2                                                   │   │
│  │    backoff: [2s, 5s]                                                │   │
│  │    on_rate_limit: wait_60s_retry                                    │   │
│  │    on_fail: try_alternate_query                                     │   │
│  │                                                                      │   │
│  │  web_fetch:                                                         │   │
│  │    max_retries: 2                                                   │   │
│  │    timeout: 30s                                                     │   │
│  │    on_fail: skip_source      # Don't fail whole job                │   │
│  │    on_timeout: skip_source                                          │   │
│  │                                                                      │   │
│  │  mermaid_render:                                                    │   │
│  │    max_retries: 2                                                   │   │
│  │    fallback: save_raw_mermaid  # Human can render manually         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## AGENT PIPELINE - DETAILED FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                      PHASE 0.5: TOPIC DISCOVERY                              │
│                      Model: Flash-Lite                                       │
│                      LLM Calls: 1                                            │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PURPOSE: Gather current web context about cutting-edge topics before       │
│           planning subtopics, since LLM training data may be outdated.      │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 0.5.1: GENERATE DISCOVERY QUERIES                            │  │
│    │ Model: Flash-Lite                                                 │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │ QUERY GENERATOR PROMPT                                             │  │
│    │ ┌─────────────────────────────────────────────────────────────┐   │  │
│    │ │ Generate 3-5 search queries to learn about this topic:      │   │  │
│    │ │                                                              │   │  │
│    │ │ Title: "{title}"                                            │   │  │
│    │ │ Context: "{context}"                                        │   │  │
│    │ │                                                              │   │  │
│    │ │ Goals:                                                       │   │  │
│    │ │ - Understand what this topic is about                       │   │  │
│    │ │ - Find key subtopics and concepts                           │   │  │
│    │ │ - Discover recent developments (2024-2025)                  │   │  │
│    │ │ - Identify practical use cases                              │   │  │
│    │ │                                                              │   │  │
│    │ │ Output JSON: { "queries": ["...", "...", ...] }             │   │  │
│    │ └─────────────────────────────────────────────────────────────┘   │  │
│    │                                                                     │  │
│    │ OUTPUT:                                                            │  │
│    │ {                                                                  │  │
│    │   "queries": [                                                     │  │
│    │     "semantic caching LLM 2024",                                  │  │
│    │     "GPTCache how it works",                                      │  │
│    │     "vector similarity caching AI applications",                  │  │
│    │     "semantic cache vs traditional cache LLM"                     │  │
│    │   ]                                                               │  │
│    │ }                                                                  │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 0.5.2: SEARCH                                                │  │
│    │ (No LLM - DuckDuckGo API)                                         │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  from duckduckgo_search import DDGS                                │  │
│    │                                                                     │  │
│    │  For each query in discovery_queries:                              │  │
│    │    results = DDGS().text(query, max_results=5)                    │  │
│    │    collect: title, url, description (snippet)                     │  │
│    │                                                                     │  │
│    │  Deduplicate by URL                                                │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 0.5.3: COMPILE SNIPPETS                                      │  │
│    │ (No LLM - programmatic)                                           │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  Format for planner:                                               │  │
│    │  topic_context = [                                                 │  │
│    │    {                                                               │  │
│    │      "title": "GPTCache: A Library for Creating Semantic Cache",  │  │
│    │      "url": "https://github.com/...",                             │  │
│    │      "snippet": "GPTCache is a library for creating semantic..."  │  │
│    │    },                                                              │  │
│    │    ...                                                             │  │
│    │  ]                                                                 │  │
│    │                                                                     │  │
│    │  Limit to top 15-20 unique results                                │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  OUTPUT (topic_context.json)                                                │
│  {                                                                          │
│    "queries_used": ["...", "..."],                                         │
│    "results": [                                                            │
│      { "title": "...", "url": "...", "snippet": "..." },                   │
│      ...                                                                    │
│    ],                                                                       │
│    "result_count": 18                                                      │
│  }                                                                          │
│                                                                              │
│  CHECKPOINT: Save topic_context.json                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                         PHASE 1: PLANNING                                    │
│                         Model: Flash-Lite                                    │
│                         LLM Calls: 1                                         │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  INPUT                                                                       │
│  ├── title: "Semantic Caching for LLM Applications"                         │
│  ├── context: "Saw GPTCache on Twitter. Redis vector search..."            │
│  ├── target_length: "medium" (1500 words)                                   │
│  ├── style_guide: [embedded from your blog analysis]                        │
│  └── topic_context: [search snippets from Phase 0.5]                        │
│                                                                              │
│  PLANNER PROMPT                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ You are planning a technical blog post.                             │   │
│  │                                                                      │   │
│  │ Blog structure must follow:                                         │   │
│  │ 1. Hook (optional) - story, surprising stat, or provocative question│   │
│  │ 2. Problem - what's broken with current approach                    │   │
│  │ 3. Why - why new approach matters                                   │   │
│  │ 4. Subtopics - 2-4 sections diving deep                            │   │
│  │ 5. Conclusion - practical takeaways                                 │   │
│  │                                                                      │   │
│  │ For each section, provide:                                          │   │
│  │ - title                                                             │   │
│  │ - role (hook/problem/why/implementation/conclusion)                 │   │
│  │ - search_queries (2-3 specific queries for research)                │   │
│  │ - needs_code (true/false)                                           │   │
│  │ - needs_diagram (true if architecture/flow explanation)             │   │
│  │ - target_words (distribute {total_words} across sections)           │   │
│  │                                                                      │   │
│  │ ## Topic Research (from web search)                                 │   │
│  │ The following snippets provide current context about this topic:    │   │
│  │ {topic_context_snippets}                                            │   │
│  │                                                                      │   │
│  │ Use this research to inform your subtopic selection.                │   │
│  │ Focus on aspects that appear important based on this context.       │   │
│  │                                                                      │   │
│  │ Output JSON.                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT (plan.json)                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                    │   │
│  │   "blog_title": "Semantic Caching for LLM Applications",            │   │
│  │   "target_words": 1500,                                             │   │
│  │   "sections": [                                                     │   │
│  │     {                                                               │   │
│  │       "id": "hook",                                                 │   │
│  │       "title": null,                                                │   │
│  │       "role": "hook",                                               │   │
│  │       "hook_type": "statistic",                                     │   │
│  │       "hook_idea": "Cost of redundant LLM calls globally",          │   │
│  │       "search_queries": ["LLM API cost statistics 2024"],           │   │
│  │       "needs_code": false,                                          │   │
│  │       "needs_diagram": false,                                       │   │
│  │       "target_words": 100                                           │   │
│  │     },                                                              │   │
│  │     {                                                               │   │
│  │       "id": "problem",                                              │   │
│  │       "title": "Why Traditional Caching Fails for LLMs",            │   │
│  │       "role": "problem",                                            │   │
│  │       "search_queries": [                                           │   │
│  │         "LLM caching challenges",                                   │   │
│  │         "exact match cache miss NLP"                                │   │
│  │       ],                                                            │   │
│  │       "needs_code": false,                                          │   │
│  │       "needs_diagram": false,                                       │   │
│  │       "target_words": 200                                           │   │
│  │     },                                                              │   │
│  │     {                                                               │   │
│  │       "id": "why",                                                  │   │
│  │       "title": "What Semantic Caching Does Differently",            │   │
│  │       "role": "why",                                                │   │
│  │       "search_queries": [                                           │   │
│  │         "semantic caching embeddings",                              │   │
│  │         "vector similarity cache LLM"                               │   │
│  │       ],                                                            │   │
│  │       "needs_code": false,                                          │   │
│  │       "needs_diagram": true,                                        │   │
│  │       "target_words": 300                                           │   │
│  │     },                                                              │   │
│  │     {                                                               │   │
│  │       "id": "implementation",                                       │   │
│  │       "title": "Building Semantic Cache with Redis",                │   │
│  │       "role": "implementation",                                     │   │
│  │       "search_queries": [                                           │   │
│  │         "Redis vector search tutorial",                             │   │
│  │         "GPTCache implementation Python"                            │   │
│  │       ],                                                            │   │
│  │       "needs_code": true,                                           │   │
│  │       "needs_diagram": false,                                       │   │
│  │       "target_words": 500                                           │   │
│  │     },                                                              │   │
│  │     {                                                               │   │
│  │       "id": "production",                                           │   │
│  │       "title": "Production Considerations",                         │   │
│  │       "role": "deep_dive",                                          │   │
│  │       "search_queries": [                                           │   │
│  │         "semantic cache TTL strategy",                              │   │
│  │         "embedding cache invalidation"                              │   │
│  │       ],                                                            │   │
│  │       "needs_code": true,                                           │   │
│  │       "needs_diagram": false,                                       │   │
│  │       "target_words": 300                                           │   │
│  │     },                                                              │   │
│  │     {                                                               │   │
│  │       "id": "conclusion",                                           │   │
│  │       "title": "Conclusion",                                        │   │
│  │       "role": "conclusion",                                         │   │
│  │       "search_queries": [],                                         │   │
│  │       "needs_code": false,                                          │   │
│  │       "needs_diagram": false,                                       │   │
│  │       "target_words": 100                                           │   │
│  │     }                                                               │   │
│  │   ]                                                                 │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  CHECKPOINT: Save plan.json                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                         PHASE 2: RESEARCH                                    │
│                         Model: None (no LLM calls)                           │
│                         LLM Calls: 0                                         │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  FOR EACH section with search_queries:                                      │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 2.1: SEARCH                                                   │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  from duckduckgo_search import DDGS                                │  │
│    │                                                                     │  │
│    │  For each query in section.search_queries:                         │  │
│    │    results = DDGS().text(query, max_results=5)                     │  │
│    │    collect URLs                                                    │  │
│    │                                                                     │  │
│    │  Deduplicate URLs across queries                                   │  │
│    │  Prioritize:                                                       │  │
│    │    • Recent (2024-2025)                                            │  │
│    │    • Official docs, GitHub, reputable tech blogs                   │  │
│    │    • Avoid: forums, Q&A sites (unless highly relevant)             │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 2.2: FETCH & EXTRACT                                          │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  import trafilatura                                                 │  │
│    │                                                                     │  │
│    │  For each URL:                                                     │  │
│    │    # Check cache first (for resume)                                │  │
│    │    if url in research/cache/:                                      │  │
│    │      load from cache                                               │  │
│    │    else:                                                           │  │
│    │      html = fetch(url, timeout=30s)                                │  │
│    │      content = trafilatura.extract(html)                           │  │
│    │      save to research/cache/{url_hash}.json                        │  │
│    │                                                                     │  │
│    │  Store:                                                            │  │
│    │    • url, title, content, fetch_timestamp                         │  │
│    │    • estimated_tokens (len(content) / 4)                          │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 2.3: CHUNK LONG ARTICLES                                      │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  If article > 4000 tokens:                                         │  │
│    │    Split by headings/paragraphs                                    │  │
│    │    Keep chunks that mention key terms from search query            │  │
│    │    Discard clearly irrelevant sections                             │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  CHECKPOINT: Save research/cache/                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                    PHASE 2.5: RESEARCH VALIDATION                            │
│                    Model: Flash-Lite                                         │
│                    LLM Calls: 1 (batched)                                    │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PURPOSE: Filter out low-quality, outdated, or irrelevant sources           │
│                                                                              │
│  VALIDATOR PROMPT                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ You are evaluating research sources for a technical blog about:     │   │
│  │ "{blog_title}"                                                      │   │
│  │                                                                      │   │
│  │ For each source, evaluate:                                          │   │
│  │                                                                      │   │
│  │ Sources:                                                            │   │
│  │ [                                                                   │   │
│  │   {                                                                 │   │
│  │     "id": 1,                                                        │   │
│  │     "url": "...",                                                   │   │
│  │     "title": "...",                                                 │   │
│  │     "snippet": "[first 500 chars]",                                 │   │
│  │     "target_section": "implementation"                              │   │
│  │   },                                                                │   │
│  │   ...                                                               │   │
│  │ ]                                                                   │   │
│  │                                                                      │   │
│  │ For each source, respond:                                           │   │
│  │ {                                                                   │   │
│  │   "id": 1,                                                          │   │
│  │   "relevant": true/false,                                           │   │
│  │   "quality": "high" | "medium" | "low",                             │   │
│  │   "freshness": "current" | "dated" | "outdated",                    │   │
│  │   "use": true/false,                                                │   │
│  │   "reason": "Official Redis docs, highly relevant"                  │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  POST-PROCESSING:                                                           │
│  • Keep sources where use=true                                              │
│  • Ensure at least 2 sources per section (re-search if needed)             │
│  • Save to research/validated/                                              │
│  • Build sources.json mapping: source_url → section_id                      │
│                                                                              │
│  CHECKPOINT: Save research/validated/, research/sources.json                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                    PHASE 3: WRITE + FEEDBACK LOOP                            │
│                    Model: Flash (writing), Flash (critic)                    │
│                    LLM Calls: ~3-4 per section                               │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  FOR EACH section in plan.sections:                                         │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 3.1: WRITE SECTION                                            │  │
│    │ Model: Flash                                                       │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │ WRITER PROMPT                                                      │  │
│    │ ┌─────────────────────────────────────────────────────────────┐   │  │
│    │ │ ## Style Guide                                               │   │  │
│    │ │ Write in a direct, technical style for experienced engineers.│   │  │
│    │ │ - Open with a clear problem statement, no warm-up.           │   │  │
│    │ │ - Be opinionated. Say "you need X" not "you might consider". │   │  │
│    │ │ - Keep paragraphs short (2-4 sentences).                     │   │  │
│    │ │ - Use bullet points only for listing items.                  │   │  │
│    │ │ - Include specific tool names and real config examples.      │   │  │
│    │ │ - No fluff: "In today's world", "It's worth noting that".    │   │  │
│    │ │ - Address the reader as "you".                               │   │  │
│    │ │ - Code: Python preferred, include imports, be runnable.      │   │  │
│    │ │ - YAML for configuration examples.                           │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Previous Sections (for voice consistency)                 │   │  │
│    │ │ {previous_sections_text}                                     │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Research Sources                                          │   │  │
│    │ │ {validated_sources_for_this_section}                         │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Your Task                                                 │   │  │
│    │ │ Write the "{section_title}" section.                         │   │  │
│    │ │ Role: {section_role}                                         │   │  │
│    │ │ Target: ~{target_words} words                                │   │  │
│    │ │ Include code: {needs_code}                                   │   │  │
│    │ │ Include diagram: {needs_diagram}                             │   │  │
│    │ │                                                              │   │  │
│    │ │ If including mermaid diagram, use this format:               │   │  │
│    │ │ ```mermaid                                                   │   │  │
│    │ │ graph TD                                                     │   │  │
│    │ │   A[Query] --> B[Embedding]                                  │   │  │
│    │ │ ```                                                          │   │  │
│    │ │                                                              │   │  │
│    │ │ IMPORTANT: Do not copy sentences from research sources.      │   │  │
│    │ │ Synthesize and write in your own words.                      │   │  │
│    │ │ Note which sources informed your writing for citations.      │   │  │
│    │ └─────────────────────────────────────────────────────────────┘   │  │
│    │                                                                     │  │
│    │ OUTPUT:                                                            │  │
│    │ {                                                                  │  │
│    │   "content": "...(markdown text)...",                             │  │
│    │   "sources_used": ["url1", "url2"],                               │  │
│    │   "claims_to_verify": [                                           │  │
│    │     "Redis HNSW index has O(log n) search complexity"             │  │
│    │   ]                                                               │  │
│    │ }                                                                  │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 3.2: ORIGINALITY CHECK                                        │  │
│    │ (No LLM - programmatic comparison)                                 │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  from difflib import SequenceMatcher                               │  │
│    │                                                                     │  │
│    │  For each sentence in section_content:                             │  │
│    │    For each source in validated_sources:                           │  │
│    │      similarity = SequenceMatcher(sentence, source_chunk).ratio()  │  │
│    │      if similarity > 0.7:                                          │  │
│    │        flag as potential plagiarism                                │  │
│    │                                                                     │  │
│    │  Also check:                                                       │  │
│    │    • N-gram overlap (3-gram, 4-gram)                              │  │
│    │    • Exact phrase matches > 8 words                               │  │
│    │                                                                     │  │
│    │  OUTPUT:                                                           │  │
│    │  {                                                                 │  │
│    │    "flagged_sentences": [                                          │  │
│    │      {                                                             │  │
│    │        "sentence": "...",                                          │  │
│    │        "similar_to": "source_url",                                 │  │
│    │        "similarity": 0.78                                          │  │
│    │      }                                                             │  │
│    │    ]                                                               │  │
│    │  }                                                                 │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 3.3: SECTION CRITIC                                           │  │
│    │ Model: Flash                                                       │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │ CRITIC PROMPT                                                      │  │
│    │ ┌─────────────────────────────────────────────────────────────┐   │  │
│    │ │ You are a senior technical blog editor.                      │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Style Guide                                               │   │  │
│    │ │ {same_style_guide_as_writer}                                 │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Section Being Reviewed                                    │   │  │
│    │ │ Title: "{section_title}"                                     │   │  │
│    │ │ Role: {section_role}                                         │   │  │
│    │ │ Target words: {target_words}                                 │   │  │
│    │ │ Actual words: {actual_words}                                 │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Content                                                   │   │  │
│    │ │ ---                                                          │   │  │
│    │ │ {section_content}                                            │   │  │
│    │ │ ---                                                          │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Originality Flags (from automated check)                  │   │  │
│    │ │ {flagged_sentences}                                          │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Evaluation (score 1-10, 8+ is passing)                    │   │  │
│    │ │                                                              │   │  │
│    │ │ 1. Technical Accuracy: Claims correct? Misleading statements?│   │  │
│    │ │ 2. Completeness: Covers what title promises?                 │   │  │
│    │ │ 3. Code Quality: Imports? Runnable? Well-explained?          │   │  │
│    │ │ 4. Clarity: Easy to follow? Terms explained?                 │   │  │
│    │ │ 5. Voice: Matches style guide? No fluff? Opinionated?        │   │  │
│    │ │ 6. Originality: Flagged sentences need rewriting?            │   │  │
│    │ │ 7. Length: Within 20% of target?                             │   │  │
│    │ │ 8. Diagram Quality: (if mermaid present) Correct? Clear?     │   │  │
│    │ │                                                              │   │  │
│    │ │ Also identify:                                               │   │  │
│    │ │ - failure_type: null | "writing" | "research_gap" | "human"  │   │  │
│    │ │ - specific issues with line numbers                          │   │  │
│    │ │ - claims that should be fact-checked by human                │   │  │
│    │ └─────────────────────────────────────────────────────────────┘   │  │
│    │                                                                     │  │
│    │ OUTPUT:                                                            │  │
│    │ {                                                                  │  │
│    │   "scores": {                                                      │  │
│    │     "technical_accuracy": 9,                                       │  │
│    │     "completeness": 7,                                             │  │
│    │     "code_quality": 8,                                             │  │
│    │     "clarity": 8,                                                  │  │
│    │     "voice": 6,                                                    │  │
│    │     "originality": 5,                                              │  │
│    │     "length": 9,                                                   │  │
│    │     "diagram_quality": null                                        │  │
│    │   },                                                               │  │
│    │   "overall_pass": false,                                           │  │
│    │   "failure_type": "writing",                                       │  │
│    │   "issues": [                                                      │  │
│    │     {                                                              │  │
│    │       "dimension": "voice",                                        │  │
│    │       "location": "paragraph 2",                                   │  │
│    │       "problem": "Starts with 'It is important to note...'",       │  │
│    │       "suggestion": "Remove filler, lead with the actual point"    │  │
│    │     },                                                             │  │
│    │     {                                                              │  │
│    │       "dimension": "originality",                                  │  │
│    │       "location": "paragraph 4, sentence 2",                       │  │
│    │       "problem": "Too similar to source redis.io/docs",            │  │
│    │       "suggestion": "Rephrase in your own words"                   │  │
│    │     }                                                              │  │
│    │   ],                                                               │  │
│    │   "fact_check_needed": [                                           │  │
│    │     "HNSW provides O(log n) search - verify this claim"            │  │
│    │   ],                                                               │  │
│    │   "missing_research": null,                                        │  │
│    │   "praise": "Strong opening, good practical example"               │  │
│    │ }                                                                  │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 3.4: DECISION GATE                                            │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │                    ALL scores ≥ 8?                                  │  │
│    │                          │                                          │  │
│    │            ┌─────────────┴─────────────┐                           │  │
│    │            │                           │                            │  │
│    │           YES                          NO                           │  │
│    │            │                           │                            │  │
│    │            ▼                           ▼                            │  │
│    │     ┌───────────┐          ┌─────────────────────┐                 │  │
│    │     │  SECTION  │          │  Check failure_type │                 │  │
│    │     │  PASSED   │          └──────────┬──────────┘                 │  │
│    │     └───────────┘                     │                            │  │
│    │            │               ┌──────────┼──────────┐                 │  │
│    │            │               │          │          │                 │  │
│    │            │               ▼          ▼          ▼                 │  │
│    │            │           "writing"  "research"  "human"              │  │
│    │            │               │        "_gap"       │                 │  │
│    │            │               │          │          │                 │  │
│    │            │               ▼          ▼          ▼                 │  │
│    │            │          ┌────────┐ ┌────────┐ ┌────────┐            │  │
│    │            │          │ REFINE │ │RE-RSCH │ │ HUMAN  │            │  │
│    │            │          │(3.5)   │ │+ WRITE │ │ INPUT  │            │  │
│    │            │          └───┬────┘ └───┬────┘ └───┬────┘            │  │
│    │            │              │          │          │                  │  │
│    │            │              └────┬─────┴──────────┘                  │  │
│    │            │                   │                                   │  │
│    │            │                   ▼                                   │  │
│    │            │           retry_count < 2?                            │  │
│    │            │                   │                                   │  │
│    │            │         ┌────────┴────────┐                          │  │
│    │            │         │                 │                           │  │
│    │            │        YES                NO                          │  │
│    │            │         │                 │                           │  │
│    │            │         │                 ▼                           │  │
│    │            │         │         ┌─────────────┐                    │  │
│    │            │         │         │ HUMAN HELP  │                    │  │
│    │            │         │         │ "I'm stuck" │                    │  │
│    │            │         │         └──────┬──────┘                    │  │
│    │            │         │                │                           │  │
│    │            │         └──────┬─────────┘                           │  │
│    │            │                │                                      │  │
│    │            │                ▼                                      │  │
│    │            │       Loop back to STEP 3.1                          │  │
│    │            │       (with issues as guidance)                       │  │
│    │            │                                                       │  │
│    │            └────────────────┬──────────────────────────────────   │  │
│    │                             │                                      │  │
│    │                             ▼                                      │  │
│    │                  --review-sections flag?                           │  │
│    │                             │                                      │  │
│    │                   ┌────────┴────────┐                             │  │
│    │                   │                 │                              │  │
│    │                  YES                NO                             │  │
│    │                   │                 │                              │  │
│    │                   ▼                 │                              │  │
│    │            ┌─────────────┐          │                              │  │
│    │            │HUMAN REVIEW │          │                              │  │
│    │            │   PAUSE     │          │                              │  │
│    │            └──────┬──────┘          │                              │  │
│    │                   │                 │                              │  │
│    │                   ▼                 │                              │  │
│    │            [approve/edit/           │                              │  │
│    │             rewrite/skip]           │                              │  │
│    │                   │                 │                              │  │
│    │                   └────────┬────────┘                              │  │
│    │                            │                                       │  │
│    │                            ▼                                       │  │
│    │                    NEXT SECTION                                    │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 3.5: REFINE SECTION                                           │  │
│    │ Model: Flash                                                       │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │ REFINER PROMPT                                                     │  │
│    │ ┌─────────────────────────────────────────────────────────────┐   │  │
│    │ │ You are refining a blog section based on editor feedback.    │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Original Section                                          │   │  │
│    │ │ {section_content}                                            │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Issues to Fix                                             │   │  │
│    │ │ {issues_from_critic}                                         │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Human Guidance (if any)                                   │   │  │
│    │ │ {human_feedback}                                             │   │  │
│    │ │                                                              │   │  │
│    │ │ Rewrite the section addressing ALL issues.                   │   │  │
│    │ │ Maintain the same structure unless issues require changes.   │   │  │
│    │ │ Keep what's working (noted in praise).                       │   │  │
│    │ └─────────────────────────────────────────────────────────────┘   │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  CHECKPOINT: Save drafts/sections/{section_id}.md after each section       │
│  CHECKPOINT: Save feedback/section_{id}_critic.json                        │
│  CHECKPOINT: Accumulate fact_check_needed into fact_check.md               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                    PHASE 4: FINAL ASSEMBLY                                   │
│                    Model: Flash                                              │
│                    LLM Calls: 3-4                                            │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.1: COMBINE SECTIONS                                         │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  • Concatenate all approved sections in order                      │  │
│    │  • Add main title as H1                                            │  │
│    │  • Ensure consistent heading hierarchy (H2 for sections)           │  │
│    │  • Save as drafts/v1.md                                            │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.2: FINAL CRITIC                                             │  │
│    │ Model: Flash                                                       │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │ FINAL CRITIC PROMPT                                                │  │
│    │ ┌─────────────────────────────────────────────────────────────┐   │  │
│    │ │ You are reviewing a complete blog post draft.                │   │  │
│    │ │                                                              │   │  │
│    │ │ ## Full Draft                                                │   │  │
│    │ │ {complete_draft}                                             │   │  │
│    │ │                                                              │   │  │
│    │ │ Individual sections already passed quality checks.           │   │  │
│    │ │ Now evaluate the WHOLE:                                      │   │  │
│    │ │                                                              │   │  │
│    │ │ 1. Coherence (1-10): Do sections flow into each other?       │   │  │
│    │ │ 2. Voice Consistency (1-10): Same tone throughout?           │   │  │
│    │ │ 3. No Redundancy (1-10): Any repeated points across sections?│   │  │
│    │ │ 4. Narrative Arc (1-10): Problem→Solution journey clear?     │   │  │
│    │ │ 5. Hook Effectiveness (1-10): Does opening grab attention?   │   │  │
│    │ │ 6. Conclusion Strength (1-10): Actionable takeaways?         │   │  │
│    │ │ 7. Overall Polish (1-10): Ready to publish?                  │   │  │
│    │ │                                                              │   │  │
│    │ │ Also provide:                                                │   │  │
│    │ │ - Transition fixes needed between specific sections          │   │  │
│    │ │ - Any remaining fact-check items                             │   │  │
│    │ │ - Suggested meta description (for SEO)                       │   │  │
│    │ │ - Estimated reading time                                     │   │  │
│    │ └─────────────────────────────────────────────────────────────┘   │  │
│    │                                                                     │  │
│    │ OUTPUT:                                                            │  │
│    │ {                                                                  │  │
│    │   "scores": { ... },                                               │  │
│    │   "overall_pass": true/false,                                      │  │
│    │   "transition_fixes": [                                            │  │
│    │     {                                                              │  │
│    │       "between": ["problem", "why"],                               │  │
│    │       "issue": "Abrupt jump",                                      │  │
│    │       "suggestion": "Add bridging sentence about solutions"        │  │
│    │     }                                                              │  │
│    │   ],                                                               │  │
│    │   "fact_check_final": [...],                                       │  │
│    │   "meta_description": "Learn how semantic caching reduces...",     │  │
│    │   "reading_time_minutes": 6,                                       │  │
│    │   "word_count": 1487                                               │  │
│    │ }                                                                  │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.3: FINAL REFINE (if needed)                                 │  │
│    │ Model: Flash                                                       │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  If overall_pass = false:                                          │  │
│    │    Apply transition_fixes                                          │  │
│    │    Re-run final critic                                             │  │
│    │    Max 2 iterations                                                │  │
│    │                                                                     │  │
│    │  Save as drafts/v2.md (or v3.md)                                   │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.4: RENDER MERMAID DIAGRAMS                                  │  │
│    │ API: kroki.io                                                      │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  import requests, base64, re                                       │  │
│    │                                                                     │  │
│    │  # Find all mermaid blocks in draft                                │  │
│    │  mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', draft)     │  │
│    │                                                                     │  │
│    │  for i, diagram in enumerate(mermaid_blocks):                      │  │
│    │      encoded = base64.urlsafe_b64encode(diagram.encode()).decode() │  │
│    │      url = f"https://kroki.io/mermaid/png/{encoded}"               │  │
│    │      response = requests.get(url)                                  │  │
│    │                                                                     │  │
│    │      if response.ok:                                               │  │
│    │          save to images/diagram_{i}.png                            │  │
│    │          replace mermaid block with ![](images/diagram_{i}.png)    │  │
│    │      else:                                                         │  │
│    │          keep raw mermaid (fallback for human)                     │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.5: ADD CITATIONS                                            │  │
│    │ (Default ON, skip if --no-citations)                               │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  Load sources.json (source_url → section mapping)                  │  │
│    │                                                                     │  │
│    │  Option A: References section at end                               │  │
│    │  ---                                                               │  │
│    │  ## References                                                     │  │
│    │                                                                     │  │
│    │  1. [Redis Vector Search Documentation](https://redis.io/...)      │  │
│    │  2. [GPTCache: Semantic Caching for LLMs](https://github.com/...)  │  │
│    │  ---                                                               │  │
│    │                                                                     │  │
│    │  Option B: Inline citations (for key claims)                       │  │
│    │  "Redis uses HNSW algorithm for vector indexing                    │  │
│    │   ([source](https://redis.io/docs/...))."                          │  │
│    │                                                                     │  │
│    │  → Use Option A (cleaner, less intrusive)                          │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ STEP 4.6: GENERATE METADATA                                        │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  metadata.json:                                                    │  │
│    │  {                                                                 │  │
│    │    "job_id": "2024-12-19_semantic-caching",                        │  │
│    │    "title": "Semantic Caching for LLM Applications",               │  │
│    │    "meta_description": "Learn how semantic caching...",            │  │
│    │    "word_count": 1487,                                             │  │
│    │    "reading_time_minutes": 6,                                      │  │
│    │    "created_at": "2024-12-19T10:30:00Z",                           │  │
│    │    "completed_at": "2024-12-19T10:52:00Z",                         │  │
│    │    "total_duration_minutes": 22,                                   │  │
│    │    "token_usage": {                                                │  │
│    │      "total_in": 89000,                                            │  │
│    │      "total_out": 12000,                                           │  │
│    │      "by_phase": { ... }                                           │  │
│    │    },                                                              │  │
│    │    "llm_calls": 24,                                                │  │
│    │    "sources_used": 12,                                             │  │
│    │    "sections": 6,                                                  │  │
│    │    "diagrams_generated": 1,                                        │  │
│    │    "human_interventions": 0                                        │  │
│    │  }                                                                 │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  CHECKPOINT: Save drafts/v{n}.md, images/, metadata.json                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════════ │
│                    PHASE 5: HUMAN FINAL REVIEW                               │
│  ══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────────────┐  │
│    │ PRESENT TO HUMAN                                                   │  │
│    ├────────────────────────────────────────────────────────────────────┤  │
│    │                                                                     │  │
│    │  Terminal UI shows:                                                │  │
│    │                                                                     │  │
│    │  ┌─────────────────────────────────────────────────────────────┐  │  │
│    │  │  ✓ BLOG COMPLETE                                            │  │  │
│    │  │                                                              │  │  │
│    │  │  Title: Semantic Caching for LLM Applications               │  │  │
│    │  │  Words: 1,487 | Reading time: 6 min | Sections: 6           │  │  │
│    │  │  LLM calls: 24 | Tokens: 101k | Duration: 22 min            │  │  │
│    │  │                                                              │  │  │
│    │  │  ┌─ Quality Scores ─────────────────────────────────────┐   │  │  │
│    │  │  │ Coherence: 9 | Voice: 9 | Polish: 9 | Overall: 9.0   │   │  │  │
│    │  │  └──────────────────────────────────────────────────────┘   │  │  │
│    │  │                                                              │  │  │
│    │  │  ⚠ FACT CHECK REQUIRED (see fact_check.md):                 │  │  │
│    │  │    • "HNSW provides O(log n) search complexity"             │  │  │
│    │  │    • "GPTCache reduces latency by 80%"                      │  │  │
│    │  │                                                              │  │  │
│    │  │  Files:                                                      │  │  │
│    │  │    → final.md (ready to publish)                            │  │  │
│    │  │    → images/diagram_0.png                                   │  │  │
│    │  │    → fact_check.md                                          │  │  │
│    │  │    → metadata.json                                          │  │  │
│    │  │                                                              │  │  │
│    │  └─────────────────────────────────────────────────────────────┘  │  │
│    │                                                                     │  │
│    │  [v] view final.md   [f] fact_check.md   [e] edit   [a] approve    │  │
│    │                                                                     │  │
│    └────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│                                                                              │
│    Human actions:                                                           │
│    • [a] approve → Copy to final.md, mark job complete                     │
│    • [e] edit → Open in $EDITOR, save changes                              │
│    • [r] request changes → Enter feedback, re-run Phase 4                  │
│    • [q] quit → Save state, can resume later                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## TERMINAL UI DESIGN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BLOG AGENT v1.0                                    Tokens: 45.2k | 24 calls│
│  Job: semantic-caching              Project: 2/4 (178/250 RPD remaining)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ● Phase: WRITING                          [--review-sections ON]           │
│                                                                              │
│  Sections:                                                                  │
│  ├── ✓ Hook                          [9.2] 98 words                        │
│  ├── ✓ Problem Statement             [8.8] 215 words                       │
│  ├── ✓ Why Semantic Caching          [9.0] 312 words  [has diagram]        │
│  ├── ◉ Implementation Deep-dive      [WRITING...]                          │
│  ├── ○ Production Considerations                                            │
│  └── ○ Conclusion                                                           │
│                                                                              │
│  ┌─ Current Action ─────────────────────────────────────────────────────┐  │
│  │ Writing section "Implementation Deep-dive"                            │  │
│  │ Using 4 validated sources | Target: 500 words | Code: yes            │  │
│  │                                                                       │  │
│  │ ████████████░░░░░░░░ 60% generating...                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Recent Activity ────────────────────────────────────────────────────┐  │
│  │ 10:42:15  Section "Why Semantic Caching" passed critic (score: 9.0)  │  │
│  │ 10:41:03  Generated mermaid diagram for architecture                 │  │
│  │ 10:40:22  Section "Why Semantic Caching" written (312 words)         │  │
│  │ 10:39:45  Fetched 4 sources for "Why Semantic Caching"               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [p]ause  [f]eedback  [v]iew draft  [s]kip section  [q]uit & save          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**After section completes (with --review-sections):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BLOG AGENT v1.0                                    Tokens: 52.1k | 28 calls│
│  Job: semantic-caching              Project: 2/4 (174/250 RPD remaining)   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ◉ SECTION REVIEW: "Implementation Deep-dive"                               │
│                                                                              │
│  ┌─ Scores ─────────────────────────────────────────────────────────────┐  │
│  │ Technical: 9 │ Complete: 8 │ Code: 9 │ Clarity: 9 │ Voice: 8 │ Orig: 9│  │
│  │                                                                       │  │
│  │ Overall: 8.7 ✓ PASSED                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Preview ────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ## Building Semantic Cache with Redis                               │  │
│  │                                                                       │  │
│  │  Semantic caching stores query embeddings alongside cached responses.│  │
│  │  When a new query arrives, you compute its embedding and search for  │  │
│  │  similar vectors. If similarity exceeds your threshold, return the   │  │
│  │  cached response without hitting the LLM.                            │  │
│  │                                                                       │  │
│  │  Here's a minimal implementation using Redis Stack:                  │  │
│  │                                                                       │  │
│  │  ```python                                                           │  │
│  │  import redis                                                        │  │
│  │  import numpy as np                                                  │  │
│  │  from redis.commands.search.query import Query                       │  │
│  │  from openai import OpenAI                                           │  │
│  │  ...                                                                 │  │
│  │  ```                                                                 │  │
│  │                                                                       │  │
│  │                                          [showing 15/52 lines]       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Critic Notes ───────────────────────────────────────────────────────┐  │
│  │ ✓ Strong: Clear explanation, runnable code example                   │  │
│  │ → Minor: Consider adding error handling to the code snippet          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Enter] approve & continue   [v]iew full   [e]dit   [f]eedback & rewrite  │
│  [s]kip section               [q]uit & save                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## TOKEN BUDGET - FINAL ESTIMATE

| Phase | LLM Calls | Model | Tokens (in) | Tokens (out) |
|-------|-----------|-------|-------------|--------------|
| **Topic Discovery** | 1 | Flash-Lite | 500 | 200 |
| **Planning** | 1 | Flash-Lite | 2,500 | 1,000 |
| **Research Validation** | 1 | Flash-Lite | 4,000 | 500 |
| **Per Section (×6):** | | | | |
| → Write | 1 | Flash | 6,000 | 1,500 |
| → Originality Check | 0 | (programmatic) | 0 | 0 |
| → Critic | 1 | Flash | 2,500 | 500 |
| → Refine (50% need) | 0.5 | Flash | 3,000 | 1,000 |
| → Re-critic | 0.5 | Flash | 2,000 | 400 |
| **Section subtotal (×6)** | ~18 | | ~81,000 | ~20,400 |
| **Final Assembly:** | | | | |
| → Combine | 0 | (programmatic) | 0 | 0 |
| → Final Critic | 1 | Flash | 8,000 | 1,000 |
| → Final Refine | 1 | Flash | 6,000 | 2,000 |
| → Mermaid render | 0 | (kroki API) | 0 | 0 |
| → Citations | 0 | (programmatic) | 0 | 0 |
| **TOTAL** | **~25** | | **~102,000** | **~25,100** |

**Budget check:**
- 4 projects × 250 RPD = **1,000 requests/day** → Using ~25 ✓
- 4 projects × 250k TPM = **1M tokens/min** → Using ~127k total ✓
- **Plenty of headroom for retries or longer blogs**

---

## CLI COMMANDS - FINAL

```bash
# Start new blog
blog-agent start \
  --title "Semantic Caching for LLM Applications" \
  --context "Saw GPTCache on Twitter. Redis vector search. Reduce latency and cost." \
  --length medium \
  --review-sections

# Start with all defaults (no review pauses)
blog-agent start --title "..." --context "..."

# Flags
--review-sections     # Pause after each section
--review-final        # Pause only before final output  
--review-all          # Pause at every decision point
--no-citations        # Skip references section
--no-hook             # Skip hook generation
--length short|medium|long  # Target word count

# Resume interrupted job
blog-agent resume <job_id>
blog-agent resume <job_id> --review-sections  # Change review mode

# List jobs
blog-agent jobs                    # All jobs
blog-agent jobs --status incomplete
blog-agent jobs --status complete

# View quota across all projects
blog-agent quota

# View specific job details
blog-agent show <job_id>

# Export final blog
blog-agent export <job_id> --format md|html
```

---
