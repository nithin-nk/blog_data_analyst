import time
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings
from src.utils.logger import get_logger

def gemini_llm_call(messages, model_name: str = "gemini-2.5-flash", settings=None, structured_output=None, max_retries=None, retry_delay=None):
    """
    Helper to call Gemini LLM with cycling through GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2.
    Tries both main and preview models for each key. Logs warnings on rate limit errors.
    
    Key rotation happens immediately on 429/rate limit errors instead of waiting.
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
                        convert_system_message_to_human=True,
                        max_retries=0,  # Disable LangChain's internal retries so we can rotate keys
                    )
                    if structured_output:
                        llm = llm.with_structured_output(structured_output)
                    response = llm.invoke(messages)
                    return response.content if hasattr(response, "content") else response
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str
                    if is_rate_limit:
                        logger.warning(f"Rate limit hit for key {key[:6]}..., model {m} - rotating to next key")
                    else:
                        logger.warning(f"Gemini LLM error for key {key[:6]}..., model {m}: {e}")
                    last_error = e
                    # Only sleep on non-rate-limit errors; for rate limits, try next key immediately
                    if not is_rate_limit:
                        time.sleep(retry_delay)
        # Sleep before retrying all keys again
        logger.info(f"All keys exhausted, waiting {retry_delay}s before retry {attempt+1}/{max_retries}")
        time.sleep(retry_delay)
    
    logger.error(f"All Gemini LLM keys exhausted after {max_retries} attempts. Last error: {last_error}")
    raise last_error
