# config/settings.py
import os
from dotenv import load_dotenv

# Load workspace environment variables with override enabled
load_dotenv(override=True)

# Rigid Agent Constraints
MAX_AGENT_STEPS = 10  # Strict iteration cap to prevent infinite runtime loops
API_TIMEOUT = 30.0    # Operational threshold for robust failure handling

def get_llm_config(cli_api_key=None) -> dict:
    """
    Resolves the LLM configuration dynamically.
    Auto-detects provider and endpoints from the resolved API key prefix.
    """
    # 1. Resolve API Key (CLI argument overrides env vars)
    api_key = cli_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return {
            "api_key": None,
            "base_url": None,
            "model_name": None,
            "is_live": False
        }
    
    api_key = api_key.strip()
    
    # Check if it's a dummy or mock key from the test environment to run cleanly in simulation mode
    is_dummy = (
        api_key.startswith("your_")
        or api_key.startswith("sk-proj-***")
        or len(api_key) < 10
    )
    if is_dummy:
        return {
            "api_key": None,
            "base_url": None,
            "model_name": None,
            "is_live": False
        }

    # 2. Auto-detect provider and assign default configurations
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME")
    
    if api_key.startswith("sk-ant-"):
        # Anthropic Key
        default_base_url = "https://api.anthropic.com/v1"
        default_model = "claude-3-5-sonnet-20240620"
        provider = "anthropic"
    elif api_key.startswith("sk-or-"):
        # OpenRouter Key
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "meta-llama/llama-3.1-8b-instruct:free"
        provider = "openrouter"
    elif api_key.startswith("sk-"):
        # OpenAI Key
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o"
        provider = "openai"
    elif api_key.startswith("gsk_"):
        # Groq Key
        default_base_url = "https://api.groq.com/openai/v1"
        default_model = "llama3-8b-8192"
        provider = "groq"
    elif api_key.startswith("AIzaSy") or api_key.startswith("AQ"):
        # Google Gemini Key
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        default_model = "gemini-2.5-flash"
        provider = "gemini"
    else:
        # Generic/Unknown key type: check if custom env configuration exists
        default_base_url = base_url or "https://api.openai.com/v1"
        default_model = model_name or "gpt-4o"
        provider = "generic"

    # Use environment values if they were explicitly set, otherwise use defaults.
    # Overrides conflicting standard base URLs when the key type differs.
    resolved_base_url = base_url
    if not resolved_base_url:
        resolved_base_url = default_base_url
    else:
        if provider == "gemini" and "api.openai.com" in resolved_base_url:
            resolved_base_url = default_base_url
        elif provider == "openai" and "googleapis.com" in resolved_base_url:
            resolved_base_url = default_base_url

    resolved_model = model_name
    if not resolved_model:
        resolved_model = default_model
    else:
        if provider == "gemini" and "gpt-" in resolved_model.lower():
            resolved_model = default_model
        elif provider == "openai" and "gemini" in resolved_model.lower():
            resolved_model = default_model

    return {
        "api_key": api_key,
        "base_url": resolved_base_url.rstrip("/"),
        "model_name": resolved_model,
        "provider": provider,
        "is_live": True
    }
