from fastapi import Request
from fastapi.responses import JSONResponse

SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


async def extract_llm_config(request: Request, call_next):
    # Let CORS preflight and static paths pass through untouched
    if request.method == "OPTIONS" or request.url.path in SKIP_PATHS:
        return await call_next(request)

    # The API key always travels in headers, including for the SSE stream endpoint —
    # never in the URL/query string, which would leak into server logs and browser history.
    api_key = request.headers.get("X-LLM-Api-Key")
    provider = request.headers.get("X-LLM-Provider", "anthropic")
    model = request.headers.get("X-LLM-Model", "claude-haiku-4-5")

    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"detail": "API key required. Configure your key in Settings."},
        )

    valid_providers = ["anthropic", "openai", "groq"]
    if provider not in valid_providers:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported provider: {provider}"},
        )

    request.state.llm_config = {
        "api_key": api_key,
        "provider": provider,
        "model": model,
    }

    response = await call_next(request)
    return response
