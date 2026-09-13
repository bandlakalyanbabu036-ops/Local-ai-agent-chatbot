import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
CONVERSATION_DATABASE_PATH = Path(
	os.getenv("CONVERSATION_DATABASE_PATH", PROJECT_ROOT / "data" / "conversations.db")
)
CONVERSATION_ID = "default"
