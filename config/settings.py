# config/settings.py
# Resolves the LLM configuration at runtime. Auto-detects the provider,
# model, and API endpoint from the key prefix. Falls back to a local
# Transformer model if no valid key is found.

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Agent constraints
MAX_AGENT_STEPS = 10   # Hard cap on ReAct loop iterations to prevent runaway execution
API_TIMEOUT     = 30.0 # Seconds to wait for an LLM API response before failing over


def get_llm_config(cli_api_key: str = None) -> dict:
    """
    Returns a configuration dict for the LLM to use this run.

    Priority order for the API key:
      1. cli_api_key argument (passed from --api-key CLI flag or web UI)
      2. LLM_API_KEY env var
      3. OPENAI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY env vars
      4. No key found  local transformer mode

    Returns a dict with keys:
      api_key, base_url, model_name, provider, is_live
    """
    # Allow explicitly forcing local mode via env var
    provider_override = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider_override in {"local", "local_transformers", "transformers"}:
        return _local_transformer_config()

    # Resolve the best available API key
    api_key = (
        cli_api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        return _local_transformer_config()

    api_key = api_key.strip()

    # Treat placeholder / demo keys as "no key" so we don't try live calls
    if api_key.startswith("your_") or api_key.startswith("sk-proj-***") or len(api_key) < 10:
        return _local_transformer_config()

    # Auto-detect provider from key prefix
    env_base_url  = os.getenv("LLM_BASE_URL")
    env_model     = os.getenv("LLM_MODEL_NAME")

    if api_key.startswith("sk-ant-"):
        provider     = "anthropic"
        default_url  = "https://api.anthropic.com/v1"
        default_model = "claude-3-5-sonnet-20240620"

    elif api_key.startswith("sk-or-"):
        provider      = "openrouter"
        default_url   = "https://openrouter.ai/api/v1"
        default_model = "meta-llama/llama-3.1-8b-instruct:free"

    elif api_key.startswith("sk-"):
        provider      = "openai"
        default_url   = "https://api.openai.com/v1"
        default_model = "gpt-4o"

    elif api_key.startswith("gsk_"):
        provider      = "groq"
        default_url   = "https://api.groq.com/openai/v1"
        default_model = "llama3-8b-8192"

    elif api_key.startswith("AIzaSy") or api_key.startswith("AQ"):
        provider      = "gemini"
        default_url   = "https://generativelanguage.googleapis.com/v1beta/openai"
        default_model = "gemini-2.5-flash"

    else:
        # Unknown key format  treat as generic OpenAI-compatible endpoint
        provider      = "generic"
        default_url   = env_base_url or "https://api.openai.com/v1"
        default_model = env_model    or "gpt-4o"

    # Resolve final base URL — avoid cross-provider URL mismatches
    resolved_url = env_base_url or default_url
    if provider == "gemini" and "api.openai.com" in resolved_url:
        resolved_url = default_url
    elif provider == "openai" and "googleapis.com" in resolved_url:
        resolved_url = default_url

    # Resolve final model name
    resolved_model = env_model or default_model
    if provider == "gemini" and "gpt-" in resolved_model.lower():
        resolved_model = default_model
    elif provider == "openai" and "gemini" in resolved_model.lower():
        resolved_model = default_model

    return {
        "api_key":    api_key,
        "base_url":   resolved_url.rstrip("/"),
        "model_name": resolved_model,
        "provider":   provider,
        "is_live":    True,
    }


def _local_transformer_config() -> dict:
    """Returns the configuration for running fully offline with a local model."""
    return {
        "api_key":    None,
        "base_url":   None,
        "model_name": os.getenv("LOCAL_TRANSFORMER_MODEL", "google/flan-t5-small"),
        "provider":   "local_transformers",
        "is_live":    True,
    }