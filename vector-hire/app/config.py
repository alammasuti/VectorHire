import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

DB_SERVER = os.environ.get("DB_SERVER", "").strip()
DB_NAME = os.environ.get("DB_NAME", "").strip()
DB_USER = os.environ.get("DB_USER", "").strip()
DB_PASSWORD = os.environ.get("DB_PASSWORD", "").strip()
DB_TABLE = os.environ.get("DB_TABLE", "").strip()
DB_USE_WINDOWS_AUTH = os.environ.get("DB_USE_WINDOWS_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()

# Required for all connections.
REQUIRED_KEYS = ["OPENAI_API_KEY", "DB_SERVER", "DB_NAME", "DB_TABLE"]
AUTH_KEYS = ["DB_USER", "DB_PASSWORD"]


def validate():
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k, "").strip()]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    if DB_USE_WINDOWS_AUTH or (not DB_USER and not DB_PASSWORD):
        return

    missing_auth = [k for k in AUTH_KEYS if not os.environ.get(k, "").strip()]
    if missing_auth:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_auth)}")
