"""
config.py
=========
Centralna konfiguracja modułu RAG dla ActumAI Core.

Wszystkie parametry są nadpisywalne przez zmienne środowiskowe (.env),
co pozwala na bezpieczne różnicowanie ustawień między dev / staging / prod
bez zmiany kodu.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"  # np. BAAI/bge-m3


class RerankProvider(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    COHERE = "cohere"


class Settings(BaseSettings):
    """
    Ustawienia aplikacji. Wartości domyślne są bezpieczne dla środowiska
    lokalnego/deweloperskiego. W produkcji należy jawnie ustawić
    zmienne środowiskowe (szczególnie klucze API).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Ogólne
    # ------------------------------------------------------------------ #
    APP_NAME: str = "ActumAI Core - Legal RAG Engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # ------------------------------------------------------------------ #
    # Dane źródłowe
    # ------------------------------------------------------------------ #
    DATA_DIR: Path = Field(
        default=Path("./data/Docs_done_rag"),
        description="Katalog root z dokumentami .md / .mdux (akty prawne).",
    )
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".md", ".mdux")

    # ------------------------------------------------------------------ #
    # Embeddingi
    # ------------------------------------------------------------------ #
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OPENAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    HF_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1536  # dopasuj do modelu (bge-m3 = 1024)

    # ------------------------------------------------------------------ #
    # LLM (generacja odpowiedzi)
    # ------------------------------------------------------------------ #
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0  # zero -> deterministyczne, mniej "kreatywne" odpowiedzi

    # ------------------------------------------------------------------ #
    # Baza wektorowa
    # ------------------------------------------------------------------ #
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.CHROMA
    COLLECTION_NAME: str = "actumai_legal_kb"

    # Chroma (lokalnie, persystentnie na dysku)
    CHROMA_PERSIST_DIR: Path = Path("./storage/chroma_db")

    # Qdrant (lokalnie lub cloud)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128

    # ------------------------------------------------------------------ #
    # Retrieval / Anti-Hallucination Pipeline
    # ------------------------------------------------------------------ #
    SIMILARITY_TOP_K: int = 10          # ile fragmentów pobrać z bazy wektorowej
    RERANK_TOP_N: int = 3               # ile fragmentów zostawić po re-rankingu
    RERANK_PROVIDER: RerankProvider = RerankProvider.SENTENCE_TRANSFORMER
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    COHERE_API_KEY: Optional[str] = None
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"

    # Próg ufności re-rankera. Jeśli najlepszy fragment po re-rankingu ma
    # score poniżej tego progu, system NIE wywołuje LLM i od razu zwraca
    # formułkę o braku informacji - to jest twardy "circuit breaker"
    # przeciw halucynacjom, niezależny od tego, czy LLM posłucha promptu.
    MIN_RERANK_SCORE: float = 0.15

    NO_CONTEXT_RESPONSE: str = (
        "Brak wystarczających informacji w bazie wiedzy do udzielenia "
        "precyzyjnej odpowiedzi prawnej."
    )

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    CORS_ALLOW_ORIGINS: list[str] = ["*"]

    # ------------------------------------------------------------------ #
    # Walidatory
    # ------------------------------------------------------------------ #
    @field_validator("DATA_DIR", "CHROMA_PERSIST_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)


settings = Settings()
