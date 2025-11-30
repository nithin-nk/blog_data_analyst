"""
Integration test for ContentGenerator with updated prompt.
Tests if the LLM generates concise, professional blog content.
"""
import yaml
from pathlib import Path
from src.generation.content_generator import ContentGenerator

def create_test_outline():
    """Create a minimal test outline."""
    return {
        'topic': 'Building Production-Ready REST APIs with FastAPI',
        'sections': [
            {
                'heading': 'Why FastAPI for Production APIs',
                'summary': 'Explains the key benefits of using FastAPI for production REST APIs including performance, type safety, and automatic documentation.',
                'references': [
                    'https://fastapi.tiangolo.com/',
                    'https://www.python.org/dev/peps/pep-0484/'
                ]
            }
        ]
    }

def create_test_research():
    """Create minimal test research data."""
    return {
        'contents': [
            {
                'url': 'https://fastapi.tiangolo.com/',
                'markdown': '''
# FastAPI

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)
- **Fast to code**: Increase the speed to develop features by about 200% to 300%
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging
- **Easy**: Designed to be easy to use and learn. Less time reading docs
- **Short**: Minimize code duplication. Multiple features from each parameter declaration
- **Robust**: Get production-ready code. With automatic interactive documentation
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema

## Performance

Independent TechEmpower benchmarks show FastAPI applications running under Uvicorn as one of the fastest Python frameworks available, only below Starlette and Uvicorn themselves (used internally by FastAPI).

## Type Safety

FastAPI uses Python type hints for request and response validation. This provides:
- Automatic data validation
- Automatic documentation
- Editor support with autocomplete
- Fewer runtime errors

## Automatic Documentation

FastAPI automatically generates interactive API documentation using:
- Swagger UI (at /docs)
- ReDoc (at /redoc)

This documentation is generated from your code and type hints, ensuring it's always up-to-date.
'''
            },
            {
                'url': 'https://www.python.org/dev/peps/pep-0484/',
                'markdown': '''
# PEP 484 - Type Hints

This PEP introduces a provisional module to provide standard definitions and tools for type hints.

## Rationale

Python will remain a dynamically typed language, and the authors have no desire to ever make type hints mandatory. However, type hints provide:

- Better IDE support
- Static type checkers can find bugs
- Documentation for function signatures
- Runtime type checking (optional)

## Syntax

Function annotations:
```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```

Variable annotations:
```python
from typing import List, Dict

names: List[str] = ['Alice', 'Bob']
config: Dict[str, int] = {'timeout': 30}
```
'''
            }
        ]
    }

def main():
    print("=" * 80)
    print("INTEGRATION TEST: Content Generator with Updated Prompt")
    print("=" * 80)
    print()
    
    # Create test data
    print("📝 Creating test outline and research data...")
    outline_data = create_test_outline()
    research_data = create_test_research()
    
    # Save to temporary files
    test_dir = Path('/tmp/content_generator_test')
    test_dir.mkdir(exist_ok=True)
    
    outline_path = test_dir / 'test_outline.yaml'
    research_path = test_dir / 'test_research.yaml'
    
    with open(outline_path, 'w') as f:
        yaml.dump(outline_data, f)
    
    with open(research_path, 'w') as f:
        yaml.dump(research_data, f)
    
    print(f"✓ Test files created at {test_dir}")
    print()
    
    # Initialize ContentGenerator
    print("🤖 Initializing ContentGenerator...")
    generator = ContentGenerator()
    print()
    
    # Generate content
    print("🚀 Generating blog content...")
    print("=" * 80)
    print()
    
    def progress_callback(message):
        """Print progress messages to console."""
        print(message)
    
    content = generator.generate_blog_post(
        outline_path,
        research_path,
        progress_callback=progress_callback
    )
    
    print()
    print("=" * 80)
    print("📄 GENERATED CONTENT")
    print("=" * 80)
    print()
    print(content)
    print()
    print("=" * 80)
    print("✅ Integration test complete!")
    print("=" * 80)
    
    # Save output
    output_path = test_dir / 'generated_content.md'
    generator.save_content(content, output_path)
    print(f"\n💾 Content saved to: {output_path}")

if __name__ == '__main__':
    main()
