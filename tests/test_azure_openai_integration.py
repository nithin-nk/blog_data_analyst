"""
Integration tests for Azure OpenAI LLM connectivity and rate limit diagnosis.

These tests help diagnose Azure OpenAI connectivity issues and rate limits.
Run with: pytest tests/test_azure_openai_integration.py -v -s

Expected environment variables (or in .env):
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_API_VERSION (optional, defaults to 2025-01-01-preview)
- AZURE_DEPLOYMENT_NAME (optional, defaults to gpt-5-chat)
"""

import os
import time
import pytest
from typing import Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI, RateLimitError, APIError, AuthenticationError


def get_azure_config() -> Dict[str, str]:
    """Get Azure OpenAI configuration from environment."""
    return {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "api_version": os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        "deployment_name": os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-5-chat"),
    }


def has_azure_credentials() -> bool:
    """Check if Azure OpenAI credentials are available."""
    config = get_azure_config()
    return bool(config["api_key"] and config["endpoint"])


skip_no_azure = pytest.mark.skipif(
    not has_azure_credentials(),
    reason="Azure OpenAI credentials not set (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT required)",
)


class TestAzureOpenAIConnectivity:
    """Tests to verify Azure OpenAI is properly configured and reachable."""

    def test_azure_credentials_present(self):
        """Test that Azure OpenAI credentials are configured."""
        config = get_azure_config()
        
        print("\n" + "=" * 60)
        print("AZURE OPENAI CONFIGURATION CHECK")
        print("=" * 60)
        
        # Check API Key
        if config["api_key"]:
            masked_key = config["api_key"][:8] + "..." + config["api_key"][-4:]
            print(f"✓ API Key: {masked_key}")
        else:
            print("✗ API Key: NOT SET")
        
        # Check Endpoint
        if config["endpoint"]:
            print(f"✓ Endpoint: {config['endpoint']}")
        else:
            print("✗ Endpoint: NOT SET")
        
        # Check API Version
        print(f"✓ API Version: {config['api_version']}")
        
        # Check Deployment Name
        print(f"✓ Deployment Name: {config['deployment_name']}")
        
        print("=" * 60)
        
        if not config["api_key"]:
            pytest.fail("AZURE_OPENAI_API_KEY is not set")
        if not config["endpoint"]:
            pytest.fail("AZURE_OPENAI_ENDPOINT is not set")

    @skip_no_azure
    def test_azure_client_initialization(self):
        """Test that Azure OpenAI client can be initialized."""
        config = get_azure_config()
        
        try:
            client = AzureOpenAI(
                api_key=config["api_key"],
                api_version=config["api_version"],
                azure_endpoint=config["endpoint"],
            )
            print("\n✓ Azure OpenAI client initialized successfully")
            assert client is not None
        except Exception as e:
            print(f"\n✗ Failed to initialize Azure OpenAI client: {e}")
            pytest.fail(f"Client initialization failed: {e}")

    @skip_no_azure
    def test_azure_simple_completion(self):
        """Test a simple completion request to verify connectivity."""
        config = get_azure_config()
        
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )
        
        print("\n" + "=" * 60)
        print("SIMPLE COMPLETION TEST")
        print("=" * 60)
        print(f"Deployment: {config['deployment_name']}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=config["deployment_name"],
                messages=[
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                max_tokens=10,
                temperature=0,
            )
            
            elapsed = time.time() - start_time
            
            content = response.choices[0].message.content
            usage = response.usage
            
            print(f"✓ Response: {content}")
            print(f"✓ Latency: {elapsed:.2f}s")
            print(f"✓ Tokens used: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")
            print("=" * 60)
            
            assert content is not None
            assert len(content) > 0
            
        except RateLimitError as e:
            elapsed = time.time() - start_time
            print(f"✗ RATE LIMIT ERROR after {elapsed:.2f}s")
            print(f"   Error: {e}")
            self._diagnose_rate_limit(e)
            pytest.fail(f"Rate limit hit: {e}")
            
        except AuthenticationError as e:
            print(f"✗ AUTHENTICATION ERROR")
            print(f"   Error: {e}")
            print("\n   Possible causes:")
            print("   - Invalid API key")
            print("   - API key has been revoked")
            print("   - Endpoint URL is incorrect")
            pytest.fail(f"Authentication failed: {e}")
            
        except APIError as e:
            elapsed = time.time() - start_time
            print(f"✗ API ERROR after {elapsed:.2f}s")
            print(f"   Error: {e}")
            print(f"   Status Code: {getattr(e, 'status_code', 'N/A')}")
            pytest.fail(f"API error: {e}")

    def _diagnose_rate_limit(self, error: RateLimitError):
        """Diagnose and print helpful info about rate limit error."""
        print("\n" + "-" * 60)
        print("RATE LIMIT DIAGNOSIS")
        print("-" * 60)
        
        error_body = getattr(error, 'body', {}) or {}
        error_message = str(error)
        
        # Extract retry-after if available
        if hasattr(error, 'response') and error.response:
            headers = error.response.headers
            retry_after = headers.get('retry-after', 'N/A')
            x_ratelimit_remaining = headers.get('x-ratelimit-remaining-requests', 'N/A')
            x_ratelimit_reset = headers.get('x-ratelimit-reset-requests', 'N/A')
            
            print(f"   Retry-After: {retry_after}s")
            print(f"   Remaining Requests: {x_ratelimit_remaining}")
            print(f"   Reset Time: {x_ratelimit_reset}")
        
        print(f"   Error Message: {error_message[:200]}")
        
        print("\n   Possible causes:")
        print("   - Too many requests per minute (TPM limit)")
        print("   - Too many tokens per minute")
        print("   - Concurrent request limit exceeded")
        print("   - Quota exhausted for the billing period")
        
        print("\n   Recommendations:")
        print("   1. Check Azure portal for your deployment's TPM limit")
        print("   2. Increase the deployment's quota in Azure")
        print("   3. Add retry logic with exponential backoff")
        print("   4. Reduce request frequency")
        print("   5. Consider using a different deployment or region")
        print("-" * 60)


class TestAzureOpenAIRateLimits:
    """Tests specifically for diagnosing rate limit issues."""

    @skip_no_azure
    def test_rate_limit_headers(self):
        """Test to examine rate limit headers from Azure OpenAI."""
        config = get_azure_config()
        
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )
        
        print("\n" + "=" * 60)
        print("RATE LIMIT HEADERS INSPECTION")
        print("=" * 60)
        
        try:
            # Make a request and capture the response
            response = client.chat.completions.with_raw_response.create(
                model=config["deployment_name"],
                messages=[
                    {"role": "user", "content": "Hi"}
                ],
                max_tokens=5,
            )
            
            # Parse the actual response
            completion = response.parse()
            headers = response.headers
            
            print("Response Headers Related to Rate Limits:")
            rate_limit_headers = [
                'x-ratelimit-limit-requests',
                'x-ratelimit-limit-tokens',
                'x-ratelimit-remaining-requests',
                'x-ratelimit-remaining-tokens',
                'x-ratelimit-reset-requests',
                'x-ratelimit-reset-tokens',
                'retry-after',
                'retry-after-ms',
            ]
            
            for header in rate_limit_headers:
                value = headers.get(header, 'N/A')
                print(f"   {header}: {value}")
            
            print(f"\nResponse Content: {completion.choices[0].message.content}")
            print("=" * 60)
            
        except RateLimitError as e:
            print(f"✗ Rate limit hit during test")
            print(f"   Error: {e}")
            if hasattr(e, 'response') and e.response:
                print("\nHeaders from error response:")
                for header in e.response.headers:
                    if 'rate' in header.lower() or 'retry' in header.lower():
                        print(f"   {header}: {e.response.headers[header]}")
            pytest.skip("Rate limit encountered - see output for details")
        except Exception as e:
            print(f"✗ Error: {e}")
            pytest.fail(str(e))

    @skip_no_azure
    def test_sequential_requests_timing(self):
        """Test sequential requests to measure rate limiting behavior."""
        config = get_azure_config()
        
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )
        
        print("\n" + "=" * 60)
        print("SEQUENTIAL REQUESTS TEST")
        print("=" * 60)
        print(f"Testing 3 sequential requests...")
        print("-" * 60)
        
        results = []
        
        for i in range(3):
            start_time = time.time()
            status = "success"
            error_msg = None
            tokens_used = 0
            
            try:
                response = client.chat.completions.create(
                    model=config["deployment_name"],
                    messages=[
                        {"role": "user", "content": f"Count: {i + 1}"}
                    ],
                    max_tokens=10,
                )
                tokens_used = response.usage.total_tokens
            except RateLimitError as e:
                status = "rate_limited"
                error_msg = str(e)[:100]
            except Exception as e:
                status = "error"
                error_msg = str(e)[:100]
            
            elapsed = time.time() - start_time
            results.append({
                "request": i + 1,
                "status": status,
                "elapsed": elapsed,
                "tokens": tokens_used,
                "error": error_msg,
            })
            
            emoji = "✓" if status == "success" else "✗"
            print(f"   {emoji} Request {i + 1}: {status} ({elapsed:.2f}s, {tokens_used} tokens)")
            if error_msg:
                print(f"      Error: {error_msg}")
            
            # Small delay between requests
            if i < 2:
                time.sleep(1)
        
        print("-" * 60)
        
        # Summary
        successes = sum(1 for r in results if r["status"] == "success")
        rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
        errors = sum(1 for r in results if r["status"] == "error")
        
        print(f"Summary: {successes} success, {rate_limited} rate limited, {errors} errors")
        
        if rate_limited > 0:
            print("\n⚠️  Rate limiting detected!")
            print("   Your deployment may have low TPM/RPM limits.")
            print("   Consider increasing quota in Azure portal.")
        
        print("=" * 60)
        
        # Don't fail if rate limited, just report
        if successes == 0:
            pytest.skip("All requests were rate limited or failed")

    @skip_no_azure
    def test_retry_with_backoff(self):
        """Test retry mechanism with exponential backoff."""
        config = get_azure_config()
        
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )
        
        print("\n" + "=" * 60)
        print("RETRY WITH BACKOFF TEST")
        print("=" * 60)
        
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(1, max_retries + 1):
            print(f"\nAttempt {attempt}/{max_retries}")
            start_time = time.time()
            
            try:
                response = client.chat.completions.create(
                    model=config["deployment_name"],
                    messages=[
                        {"role": "user", "content": "Test retry mechanism"}
                    ],
                    max_tokens=20,
                )
                
                elapsed = time.time() - start_time
                print(f"✓ Success on attempt {attempt} ({elapsed:.2f}s)")
                print(f"   Response: {response.choices[0].message.content[:50]}...")
                print("=" * 60)
                return  # Success, exit test
                
            except RateLimitError as e:
                elapsed = time.time() - start_time
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                
                print(f"✗ Rate limited on attempt {attempt} ({elapsed:.2f}s)")
                
                # Try to get retry-after from headers
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        try:
                            delay = max(int(retry_after), delay)
                            print(f"   Server suggested retry-after: {retry_after}s")
                        except ValueError:
                            pass
                
                if attempt < max_retries:
                    print(f"   Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    print(f"   Max retries ({max_retries}) exceeded")
                    print("\n" + "=" * 60)
                    pytest.skip("Rate limit persists after all retries")
                    
            except Exception as e:
                print(f"✗ Error on attempt {attempt}: {e}")
                pytest.fail(str(e))
        
        print("=" * 60)


class TestAzureOpenAIWithBlogReviewer:
    """Test Azure OpenAI integration through the BlogReviewer."""

    @skip_no_azure
    def test_blog_reviewer_azure_connection(self):
        """Test that BlogReviewer can connect to Azure OpenAI."""
        from src.refinement.blog_reviewer import BlogReviewer, ModelReviewResult
        
        reviewer = BlogReviewer()
        
        print("\n" + "=" * 60)
        print("BLOG REVIEWER AZURE OPENAI TEST")
        print("=" * 60)
        print(f"Deployment: {reviewer.settings.azure_deployment_name}")
        print(f"Endpoint: {reviewer.settings.azure_openai_endpoint}")
        print("-" * 60)
        
        # Simple test content
        title = "Test Blog Post"
        content = """# Test Blog Post

This is a simple test blog post to verify Azure OpenAI connectivity.

## Section 1

Some content here.

## Conclusion

This is the end.
"""
        
        start_time = time.time()
        
        result = reviewer._review_with_azure_openai(
            title=title,
            content=content,
        )
        
        elapsed = time.time() - start_time
        
        print(f"Latency: {elapsed:.2f}s")
        
        if result.error:
            print(f"✗ Error: {result.error}")
            
            # Diagnose the error
            error_lower = result.error.lower()
            if "429" in result.error or "rate" in error_lower:
                print("\n   DIAGNOSIS: Rate limit error")
                print("   The Azure OpenAI deployment has exceeded its quota.")
                print("   Solutions:")
                print("   1. Wait and retry later")
                print("   2. Increase TPM quota in Azure portal")
                print("   3. Use a different deployment")
            elif "401" in result.error or "auth" in error_lower:
                print("\n   DIAGNOSIS: Authentication error")
                print("   Check your AZURE_OPENAI_API_KEY")
            elif "404" in result.error:
                print("\n   DIAGNOSIS: Deployment not found")
                print(f"   Check if deployment '{reviewer.settings.azure_deployment_name}' exists")
            
            print("=" * 60)
            pytest.fail(f"Azure OpenAI review failed: {result.error}")
        else:
            print(f"✓ Success!")
            print(f"   Score: {result.score}/10")
            print(f"   Feedback items: {len(result.feedback)}")
            print(f"   Can apply feedback: {result.can_apply_feedback}")
            if result.feedback:
                print(f"   Sample feedback: {result.feedback[0][:80]}...")
            print("=" * 60)
            
            assert isinstance(result, ModelReviewResult)
            assert 1.0 <= result.score <= 10.0


class TestAzureQuotaDiagnostics:
    """Diagnostic tests to understand Azure OpenAI quota and limits."""

    @skip_no_azure
    def test_deployment_info(self):
        """Display deployment configuration information."""
        config = get_azure_config()
        
        print("\n" + "=" * 60)
        print("AZURE OPENAI DEPLOYMENT INFO")
        print("=" * 60)
        print(f"Endpoint: {config['endpoint']}")
        print(f"Deployment: {config['deployment_name']}")
        print(f"API Version: {config['api_version']}")
        print("-" * 60)
        print("\nTo check your quota limits:")
        print("1. Go to Azure Portal")
        print("2. Navigate to your Azure OpenAI resource")
        print("3. Go to 'Model deployments' -> 'Manage Deployments'")
        print("4. Click on your deployment to see TPM/RPM limits")
        print("-" * 60)
        print("\nCommon quota limits for Azure OpenAI:")
        print("- TPM (Tokens Per Minute): 10K - 120K+ depending on tier")
        print("- RPM (Requests Per Minute): 6 - 3500+ depending on tier")
        print("=" * 60)

    @skip_no_azure 
    def test_measure_single_request_tokens(self):
        """Measure tokens used in a typical blog review request."""
        from src.refinement.blog_reviewer import BlogReviewer
        
        reviewer = BlogReviewer()
        config = get_azure_config()
        
        # Sample blog content (typical size)
        sample_content = """# Memory for AI Agents

AI agents need memory to be effective. Here's how to implement it.

## Understanding Memory

Memory allows agents to remember past interactions and learn over time.

## Implementation

```python
from mem0 import Memory
m = Memory()
m.add("User prefers Python", user_id="alice")
```

## Conclusion

Memory is essential for intelligent agents.
"""
        
        print("\n" + "=" * 60)
        print("TOKEN USAGE MEASUREMENT")
        print("=" * 60)
        
        prompt = reviewer._get_review_prompt("Test Blog", sample_content)
        
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )
        
        try:
            response = client.chat.completions.create(
                model=config["deployment_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            usage = response.usage
            print(f"Prompt tokens: {usage.prompt_tokens}")
            print(f"Completion tokens: {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
            print("-" * 60)
            print("\nIf your TPM limit is 10,000:")
            requests_per_minute = 10000 // usage.total_tokens
            print(f"  Max requests/minute: ~{requests_per_minute}")
            print(f"  Safe rate: 1 request every {60 // max(requests_per_minute, 1)}s")
            print("=" * 60)
            
        except RateLimitError as e:
            print(f"✗ Rate limited - cannot measure tokens")
            print(f"   Error: {e}")
            pytest.skip("Rate limited")
        except Exception as e:
            pytest.fail(str(e))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
