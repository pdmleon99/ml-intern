from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

SUPPORTED_MODELS = {
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
}


def get_llm(llm_config: dict):
    """Build LLM client from user-provided config.
    The api_key comes from the request header, never from env vars.
    """
    provider = llm_config["provider"]
    api_key = llm_config["api_key"]
    model = llm_config["model"]

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "groq":
        return ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=2048,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def validate_llm_config(llm_config: dict) -> bool:
    """Make a minimal test call to verify the key works.
    Returns True if valid, raises ValueError if not.
    """
    try:
        llm = get_llm(llm_config)
        llm.invoke("Reply with just the word OK.")
        return True
    except Exception as e:
        error_msg = str(e).replace(llm_config.get("api_key", ""), "[REDACTED]")
        raise ValueError(f"Key validation failed: {error_msg}")
