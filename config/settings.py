# config/settings.py
import os
from dotenv import load_dotenv

# Load workspace environment variables
load_dotenv()

# General LLM Configuration (supports any OpenAI-compatible or Google Gemini endpoints)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# Rigid Agent Constraints
MAX_AGENT_STEPS = 10  # Strict iteration cap to prevent infinite runtime loops
API_TIMEOUT = 30.0    # Operational threshold for robust failure handling