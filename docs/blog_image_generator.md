# Blog Image Generator

The Blog Image Generator is an AI-powered agent that automatically creates cover/social sharing images for blog posts using Google's Gemini 2.5 Flash model with image generation capabilities.

## Overview

The generator follows a two-step process:
1. **Description Generation**: Analyzes the blog title and content to generate a creative visual description
2. **Image Generation**: Uses Gemini's image generation model to create a 1200x630 social sharing image

## Features

- **AI-Powered Description Generation**: Creates vivid, contextually relevant image descriptions based on blog content
- **Automatic Alt Text**: Generates accessible alt text for SEO and accessibility
- **Style Detection**: Determines appropriate image style (illustration, abstract, photorealistic, etc.)
- **Base64 Output**: Stores image as base64 in YAML alongside mermaid diagrams
- **Fallback Handling**: Graceful degradation if image generation fails

## Usage

### As Part of the Pipeline

The blog image generator is automatically invoked after diagram generation when running the `generate` command:

```bash
python -m src.main generate --input inputs/your_blog.yaml
```

To skip image generation:

```bash
python -m src.main generate --input inputs/your_blog.yaml --skip-image
```

### Programmatic Usage

```python
from src.media.blog_image_generator import BlogImageGenerator

# Initialize generator
generator = BlogImageGenerator()

# Generate image for a blog post
blog_image = generator.generate_blog_image(
    title="Memory for AI Agents Using Mem0",
    content="Your blog markdown content here...",
    progress_callback=lambda msg: print(msg)  # Optional
)

# Access results
print(f"Description: {blog_image.description}")
print(f"Alt text: {blog_image.alt_text}")
print(f"Style: {blog_image.style}")
print(f"Format: {blog_image.format}")
# blog_image.image_base64 contains the base64 encoded image
```

## Output Format

The generated image is stored in `diagrams.yaml` alongside mermaid diagrams:

```yaml
diagrams:
  - heading: "System Architecture"
    diagram_type: flowchart
    mermaid_code: |
      flowchart TD
        A --> B
    image_base64: "base64_encoded_mermaid_image..."
    score: 9.5

blog_image:
  title: "Memory for AI Agents Using Mem0"
  description: "A futuristic visualization of interconnected AI neural networks with glowing memory nodes..."
  alt_text: "AI agent memory system with neural connections"
  style: "illustration"
  image_base64: "base64_encoded_cover_image..."
  format: "png"
```

## Configuration

The following settings can be configured in `.env` or environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Google API key for Gemini access |
| `BLOG_IMAGE_DESCRIPTION_MODEL` | `gemini-2.5-flash` | Model for generating descriptions |

## Architecture

```
BlogImageGenerator
├── generate_image_description()  # LLM call to create visual description
├── generate_image()              # Gemini image generation API call
├── generate_blog_image()         # Main workflow orchestrator
└── save_to_diagrams_yaml()       # Persist to YAML file
```

## Error Handling

- **Missing API Key**: Raises `ValueError` during initialization
- **Description Generation Failure**: Returns fallback generic description
- **Image Generation Failure**: Logs warning and propagates exception (caught in main.py)

## Dependencies

- `google-genai`: Google's GenAI Python SDK for image generation
- `pydantic`: Data validation and settings management
- `langchain-core`: For message formatting
- `PyYAML`: YAML file handling

## Testing

Run tests with:

```bash
python -m pytest tests/test_blog_image_generator.py -v
```

## Notes

- Images are generated at 1200x630 aspect ratio (ideal for social sharing)
- The generator avoids text in images as AI-generated text renders poorly
- Content is truncated to 3000 characters to stay within token limits
- The image is stored as base64 to keep all assets in a single YAML file
