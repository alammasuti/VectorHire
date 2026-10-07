import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root (two levels up from this file)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Database
DB_SERVER   = os.environ.get("DB_SERVER", "")
DB_NAME     = os.environ.get("DB_NAME", "")
DB_USER     = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_TABLE    = os.environ.get("DB_TABLE", "")

# AI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL      = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# All keys that must be present for the app to work
REQUIRED_KEYS = ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_TABLE"]

def validate():
    """Raise an error if any required setting is missing."""
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
