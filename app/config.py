"""
Application configuration.

All environment-dependent values live here, in one place, as a typed
Pydantic model. Nothing else in the codebase should call os.getenv()
directly — everything imports `settings` from this module instead.

Why this matters for a "production-oriented" app:
- Typos in env var names fail fast at startup (validation error),
  instead of silently returning None deep inside some service.
- Types are enforced (e.g. CHUNK_SIZE really is an int, not a string).
- One object to mock in tests instead of monkeypatching env vars everywhere.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic-settings reads matching keys from the .env file automatically.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (Groq) ---
    groq_api_key: str = Field(..., description="API key for Groq-hosted LLM")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Embeddings ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    huggingface_hub_token: str | None = Field(default=None)
    hf_home: str = Field(default="./.hf_cache")

    # --- Vector store ---
    chroma_persist_dir: str = Field(default="./data/chroma")

    # --- File storage ---
    upload_dir: str = Field(default="./data/uploads")
    max_upload_size_mb: int = Field(default=20, gt=0)
    allowed_file_extensions: str = Field(default=".pdf,.docx,.txt")

    # --- Database ---
    database_url: str = Field(default="sqlite:///./data/app.db")

    # --- Chunking ---
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Retrieval pipeline ---
    # Candidates pulled from EACH retriever (vector, BM25) before merging.
    # Deliberately generous — the reranker below is what narrows this down
    # precisely; retrieval's job here is just "don't miss anything relevant."
    retrieval_candidate_k: int = Field(default=12, gt=0)

    # Final number of chunks handed to the LLM after reranking.
    rerank_top_n: int = Field(default=4, gt=0)

    # Chunks scoring below this are dropped entirely, even if it means
    # fewer than rerank_top_n survive (possibly zero, which correctly
    # triggers the "couldn't find anything relevant" fallback instead of
    # padding the context with weakly-related chunks from unrelated
    # documents). cross-encoder/ms-marco-MiniLM-L-6-v2 outputs unbounded
    # relevance logits, not 0-1 probabilities — scores around/above 0
    # typically indicate genuine relevance, negative scores indicate the
    # pair is likely unrelated. None disables the filter (old behavior).
    rerank_min_score: float | None = Field(default=0.0)

    # Cross-encoder model for reranking. Runs locally via sentence-transformers
    # (already a dependency for embeddings) — no extra API/key needed.
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Small/fast Groq model for query rewriting — deliberately different from
    # groq_model (used for actual answer generation) since rewriting a
    # question doesn't need a 70B model's reasoning power.
    query_rewrite_model: str = Field(default="llama-3.1-8b-instant")

    # --- CORS ---
    allowed_origins: str = Field(default="http://localhost:3000")

    # --- App ---
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    default_user_id: str = Field(default="default_user")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        """CORS middleware wants a list, not a raw comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def allowed_file_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_file_extensions.split(",") if ext.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories_exist(self) -> None:
        """Create local storage directories if they don't exist yet.
        Called once at app startup — keeps first-run setup zero-config."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance (singleton). lru_cache ensures the .env file
    is only read/parsed once per process, and every module gets the same
    object rather than re-parsing on every import.
    """
    return Settings()


settings = get_settings()

# Must happen here, at import time, before any module imports
# sentence-transformers/huggingface_hub — those libraries read HF_HOME
# from the environment at their own import time, not on every call.
# Isolates this project's model cache from other projects on shared
# machines (this is what fixed the corrupted shared-cache issue).
os.environ["HF_HOME"] = settings.hf_home
if settings.huggingface_hub_token:
    os.environ["HF_TOKEN"] = settings.huggingface_hub_token