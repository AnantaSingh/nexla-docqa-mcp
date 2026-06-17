"""Centralized, env-driven configuration.

All knobs live here so the rest of the codebase reads settings from one place.
Values come from environment variables / a local `.env` (see `.env.example`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/docqa/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- API keys ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # --- Models ---
    answer_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-haiku-4-5"
    embed_model: str = "text-embedding-3-small"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Embedding provider: "openai" (default) | "fastembed" (local) ---
    embed_provider: str = "openai"
    local_embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieval knobs ---
    vector_top_k: int = 30
    bm25_top_k: int = 30
    rerank_top_n: int = 8
    rerank_enabled: bool = True
    rrf_k: int = 60
    max_context_tokens: int = 6000

    # --- Chunking knobs ---
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 100

    # --- Paths (relative to repo root unless absolute) ---
    data_dir: str = "data"
    chroma_dir: str = ".chroma"
    fastembed_cache: str = ".fastembed_cache"  # persists the local reranker/embedder models

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def fastembed_cache_path(self) -> Path:
        p = Path(self.fastembed_cache)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def chunks_path(self) -> Path:
        """All chunk records (text + metadata) persisted for BM25 rebuild + lookup."""
        return self.chroma_path / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.chroma_path / "manifest.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
