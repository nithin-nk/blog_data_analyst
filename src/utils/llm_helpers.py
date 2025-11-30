import time
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings
from src.utils.logger import get_logger

def gemini_llm_call(messages, model_name: str = "gemini-2.5-flash", settings=None, structured_output=None, max_retries=None, retry_delay=None):
    """
    Helper to call Gemini LLM with cycling through GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2.
    Tries both main and preview models for each key. Logs warnings on rate limit errors.
    """
    logger = get_logger(__name__)
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
