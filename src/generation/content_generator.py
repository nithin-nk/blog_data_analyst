import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.llm_helpers import gemini_llm_call

logger = get_logger(__name__)

def gemini_llm_call(messages, model_name: str = "gemini-2.5-flash", settings=None, structured_output=None, max_retries=None, retry_delay=None):
    """
    Helper to call Gemini LLM with cycling through GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2.
    Tries both main and preview models for each key. Logs warnings on rate limit errors.
    """
    if settings is None:
        settings = get_settings()
    keys = [
        getattr(settings, "google_api_key", None),
        getattr(settings, "google_api_key_1", None),
        getattr(settings, "google_api_key_2", None)
    ]
    keys = [k for k in keys if k]
    models = [model_name, model_name+"-preview"]
    max_retries = max_retries or getattr(settings, "max_retries", 3)
    retry_delay = retry_delay or getattr(settings, "retry_delay", 2)
    last_error = None
    for attempt in range(max_retries):
        for key in keys:
            for m in models:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=m,
                        google_api_key=key,
                        temperature=0.7,
                        convert_system_message_to_human=True
                    )
                    if structured_output:
                        llm = llm.with_structured_output(structured_output)
                    response = llm.invoke(messages)
                    return response.content if hasattr(response, "content") else response
                except Exception as e:
                    logger.warning(f"Gemini LLM rate/error for key {key[:6]}..., model {m}: {e}")
                    last_error = e
                    time.sleep(retry_delay)
        logger.info(f"Retrying Gemini LLM call, attempt {attempt+1}/{max_retries}")
    logger.error(f"All Gemini LLM keys exhausted. Last error: {last_error}")
    raise last_error

class ContentGenerator:
    def __init__(self):
        self.settings = get_settings()
        # LLMs now handled by helper

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_research_content(self, references: List[str], research_data: Dict[str, Any]) -> str:
        """
        Retrieves content for the top 3 references from the research data.
        """
        content_parts = []
        found_count = 0
        
        # Create a lookup map for faster access
        url_to_content = {item['url']: item for item in research_data.get('contents', [])}
        
        for url in references:
            if found_count >= 3:
                break
            
            if url in url_to_content:
                item = url_to_content[url]
                # Use markdown content if available, otherwise snippet
                text = item.get('markdown', item.get('snippet', ''))
                # Truncate if too long to avoid context window issues (though Gemini has large context)
                # Let's limit to ~10k chars per source to be safe and efficient
                if len(text) > 10000:
                    text = text[:10000] + "...(truncated)"
                
                content_parts.append(f"Source: {url}\nContent:\n{text}\n")
                found_count += 1
                
        return "\n---\n".join(content_parts)

    def _generate_section_content(self, 
                                section: Dict[str, Any], 
                                research_content: str, 
                                previous_context: str,
                                topic: str) -> str:
        heading = section.get('heading', '')
        summary = section.get('summary', '')
        prompt = f"""
        You are an expert technical blog writer. You are writing a section for a blog post titled "{topic}".
        
        Current Subtopic: {heading}
        Subtopic Summary: {summary}
        
        Research Content (from top references):
        {research_content}
        
        Context from previous sections:
        {previous_context if previous_context else "This is the first section."}
        
        Instructions:
        1. Write concise and to-the-point content for this subtopic.
        2. Use bullet points where appropriate to improve readability. Avoid long, dense paragraphs.
        3. Include relevant links to the provided references (max 2 links). Format: [Link Text](URL).
        4. Ensure a smooth flow from the previous context.
        5. Optimize for SEO (use relevant keywords naturally).
        6. If relevant, generate code snippets surrounded by triple backticks (e.g., ```python ... ```).
        7. If relevant, generate Mermaid flow diagrams surrounded by triple backticks (e.g., ```mermaid ... ```).
        8. The content should be simple and easy to understand.
        9. Base the content strictly on the provided research context.
        10. Do not write a conclusion for the whole blog, just this section.
        
        Generate the content for this section now.
        """
        messages = [HumanMessage(content=prompt)]
        return gemini_llm_call(messages, model_name="gemini-2.5-flash", settings=self.settings)

    def generate_blog_post(self, outline_path: Path, research_path: Path, progress_callback: Optional[Any] = None) -> str:
        """
        Generates the full blog post by iterating through the outline.
        """
        outline_data = self._load_yaml(outline_path)
        research_data = self._load_yaml(research_path)
        
        topic = outline_data.get('topic', 'Untitled Blog')
        sections = outline_data.get('sections', [])
        
        full_content = f"# {topic}\n\n"
        previous_context = ""
        
        total_sections = len(sections)
        logger.info(f"Starting content generation for topic: {topic} ({total_sections} sections)")
        
        for i, section in enumerate(sections):
            heading = section.get('heading', '')
            msg = f"Generating section {i+1}/{total_sections}: {heading}"
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)
            
            references = section.get('references', [])
            research_content = self._get_research_content(references, research_data)
            
            section_content = self._generate_section_content(
                section, 
                research_content, 
                previous_context, 
                topic
            )
            
            # Append to full content
            full_content += f"## {heading}\n\n{section_content}\n\n"
            
            # Update context
            previous_context += f"\nSummary of {heading}:\n{section.get('summary', '')}\nGenerated Content:\n{section_content}\n"
            
            # Rate limit wait (except for the last one)
            if i < total_sections - 1:
                msg = "Waiting 20 seconds to respect rate limits..."
                logger.info(msg)
                if progress_callback:
                    progress_callback(msg)
                time.sleep(20)
                
        return full_content

    def save_content(self, content: str, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved generated content to {output_path}")
