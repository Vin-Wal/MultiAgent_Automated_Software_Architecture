import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_API_KEY: str = os.environ["LLM_API_KEY"]
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    TOP_K: int = int(os.getenv("TOP_K", "3"))

    DOCS_ROOT: Path = Path(os.getenv(
        "DOCS_ROOT",
        str(Path(__file__).parent.parent / "genai_final_project" / "agent_docs"),
    ))
    CHROMA_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


cfg = Config()
