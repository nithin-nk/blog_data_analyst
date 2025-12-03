import time
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.key_manager import get_key_manager


def gemini_llm_call(
    messages,
    model_name: str = "gemini-2.5-flash",
    settings=None,
    structured_output=None,
    max_retries=None,
    retry_delay=None,
):
    """
    Helper to call Gemini LLM with intelligent key rotation.
    
    Uses round-robin distribution across all configured API keys with automatic
    cooldown tracking when rate limits are hit. Keys are rotated immediately
    on 429 errors and placed in 60-second cooldown.
    
    When all keys are in cooldown, the function waits indefinitely until a key
    becomes available (will not fail due to rate limits alone).
    
    Args:
        messages: The messages to send to the LLM
        model_name: The Gemini model to use (default: gemini-2.5-flash)
        settings: Optional settings object
        structured_output: Optional Pydantic model for structured output
        max_retries: Maximum retry attempts for non-rate-limit errors (default from settings)
        retry_delay: Delay between retries in seconds (default from settings)
        
    Returns:
        The LLM response content
        
    Raises:
        Exception: If all retries exhausted due to non-rate-limit errors
    """
    logger = get_logger(__name__)
    
    if settings is None:
        settings = get_settings()
    
    # Initialize key manager with settings
    key_manager = get_key_manager(settings)
    
    models = [model_name, model_name + "-preview"]
    max_retries = max_retries or getattr(settings, "max_retries", 3)
    retry_delay = retry_delay or getattr(settings, "retry_delay", 2)
    last_error = None
    non_rate_limit_failures = 0
    
    while True:
        # Get the next available key from round-robin rotation
        key = key_manager.get_next_key()
        
        if key is None:
            # All keys in cooldown - wait for the shortest cooldown to expire
            wait_time = key_manager.get_wait_time_for_next_key()
            if wait_time > 0:
                logger.info(
                    f"All keys in cooldown, waiting {wait_time:.1f}s for next available key..."
                )
                time.sleep(wait_time)
                # Try again after waiting
                continue
            else:
                # No wait time but no key available - shouldn't happen, but handle it
                logger.warning("No keys available and no cooldown wait time - retrying in 5s")
                time.sleep(5)
                continue
        
        # Try both main and preview models
        for m in models:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=m,
                    google_api_key=key,
                    temperature=0.7,
                    convert_system_message_to_human=True,
                    max_retries=0,  # Disable LangChain's internal retries
                )
                
                if structured_output:
                    llm = llm.with_structured_output(structured_output)
                
                response = llm.invoke(messages)
                return response.content if hasattr(response, "content") else response
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str
                    or "quota" in error_str
                    or "rate" in error_str
                    or "resource_exhausted" in error_str
                )
                
                if is_rate_limit:
                    logger.warning(
                        f"Rate limit hit for key {key[:8]}..., model {m} - "
                        f"marking key in cooldown (60s)"
                    )
                    key_manager.mark_key_rate_limited(key)
                    # Break out of model loop to get next key (or wait if all in cooldown)
                    break
                else:
                    logger.warning(
                        f"Gemini LLM error for key {key[:8]}..., model {m}: {e}"
                    )
                    last_error = e
                    # For non-rate-limit errors, try next model
                    continue
        else:
            # All models failed for this key with non-rate-limit errors
            non_rate_limit_failures += 1
            last_error = last_error or Exception("Unknown error")
            
            if non_rate_limit_failures >= max_retries:
                # Only give up on non-rate-limit errors
                status = key_manager.get_cooldown_status()
                logger.error(
                    f"Gemini LLM failed after {max_retries} non-rate-limit errors. "
                    f"Key status: {status}. Last error: {last_error}"
                )
                raise last_error
            
            logger.info(
                f"Non-rate-limit error, retrying in {retry_delay}s "
                f"({non_rate_limit_failures}/{max_retries})"
            )
            time.sleep(retry_delay)
