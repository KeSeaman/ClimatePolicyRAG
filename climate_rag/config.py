from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via CLIMATE_RAG_* env vars or a .env file."""

    model_config = SettingsConfigDict(env_prefix="CLIMATE_RAG_", env_file=".env", extra="ignore")

    # "fastembed" runs a local ONNX model; "openai" calls any OpenAI-compatible
    # /embeddings endpoint (OpenAI, Ollama, vLLM, ...) at llm_base_url.
    embedding_backend: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # fastembed defaults to a temp dir, which re-downloads the model after every reboot.
    embedding_cache_dir: Path = Path.home() / ".cache" / "fastembed"

    # Defaults target a local Ollama server; point these at a hosted provider to switch.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma3:4b"
    llm_api_key: str = "ollama"  # Ollama ignores the key, but the client requires one
    llm_temperature: float = 0.2
    # Hard cap on answer length; without it a small local model can generate indefinitely.
    llm_max_tokens: int = 512
    llm_timeout: float = 300.0

    # Faithfulness judge for `climate-rag eval --judge`; unset values fall back to llm_*.
    # Use a stronger model than the generator: a model grading itself is lenient.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_max_tokens: int = 1500

    # "vector": embedding similarity only. "hybrid": embedding + BM25 keyword ranking,
    # merged with reciprocal rank fusion, which also catches exact acronyms and names.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    # Ask the LLM for alternative phrasings of each question and search with all of them.
    query_expansion: bool = False
    # Rank fusion (hybrid mode or expansion): candidates per ranking (unset: max(4k, 20)),
    # the RRF constant, and the weight of the original question vs each rephrasing.
    fusion_pool: int | None = None
    rrf_k: int = 60
    original_query_weight: float = 1.0

    top_k: int = 5
    chunk_chars: int = 2000
    http_timeout: float = 30.0
