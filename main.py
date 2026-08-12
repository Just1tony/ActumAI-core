"""
main.py
=======
FastAPI dla silnika ActumAI Core - API zapytań do bazy wiedzy prawnej (RAG).

Uruchomienie (dev):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI dostępny pod: /docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from ingest import build_index
from retriever import LegalRAGPipeline, get_pipeline

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("actumai.api")

_ingest_in_progress = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ładuje pipeline RAG (embeddingi, LLM, index) raz przy starcie aplikacji."""
    logger.info("Uruchamianie %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    try:
        get_pipeline()
    except Exception as exc:  # noqa: BLE001
        # Nie wywalamy procesu - endpoint /query zwróci 503, dopóki index
        # nie zostanie zbudowany (np. świeże środowisko przed pierwszym ingestem).
        logger.error("Nie udało się zainicjalizować pipeline'u RAG: %s", exc)
    yield
    logger.info("Zamykanie aplikacji.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Silnik RAG dla polskiej bazy wiedzy prawnej (Kodeks Cywilny, KPC, "
        "RODO, Prawo Pracy i inne akty). Odpowiedzi generowane WYŁĄCZNIE "
        "na podstawie kontekstu z bazy wektorowej, z re-rankingiem i "
        "podaniem dokładnej podstawy prawnej."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------ #
# Modele Pydantic (request / response)
# ------------------------------------------------------------------------ #
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pytanie prawne w języku polskim.",
        examples=["Jaki jest okres wypowiedzenia umowy o pracę na czas nieokreślony?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Opcjonalne nadpisanie liczby fragmentów pobieranych z bazy wektorowej "
        "przed re-rankingiem (domyślnie z konfiguracji serwera).",
    )


class SourceDocumentResponse(BaseModel):
    act: str = Field(..., description="Nazwa aktu prawnego, np. 'Kodeks pracy'.")
    article: str = Field(default="", description="Numer artykułu, np. '52'.")
    anchor: str = Field(default="", description="Identyfikator kotwicy/sekcji źródłowej.")
    section_context: str = Field(default="", description="Dział/Rozdział, jeśli dotyczy.")
    snippet: str = Field(..., description="Fragment tekstu użyty jako kontekst.")
    score: float = Field(..., description="Score re-rankera (im wyższy, tym trafniejszy).")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Odpowiedź modelu z podstawą prawną.")
    has_sufficient_context: bool = Field(
        ..., description="Czy w bazie wiedzy znaleziono wystarczający kontekst."
    )
    sources: list[SourceDocumentResponse] = Field(
        default_factory=list, description="Fragmenty źródłowe użyte do odpowiedzi."
    )
    latency_ms: int = Field(..., description="Czas przetwarzania zapytania w milisekundach.")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    vector_store: str
    collection: str
    pipeline_ready: bool


class IngestResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    detail: str


# ------------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------------ #
@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"{settings.APP_NAME} - zobacz /docs dla dokumentacji Swagger."}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Sprawdza stan aplikacji i gotowość pipeline'u RAG.",
)
async def health() -> HealthResponse:
    pipeline_ready = True
    try:
        get_pipeline()
    except Exception:  # noqa: BLE001
        pipeline_ready = False

    return HealthResponse(
        status="ok" if pipeline_ready else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        vector_store=settings.VECTOR_STORE_PROVIDER.value,
        collection=settings.COLLECTION_NAME,
        pipeline_ready=pipeline_ready,
    )


@app.post(
    f"{settings.API_PREFIX}/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["RAG"],
    summary="Zadaje pytanie prawne do bazy wiedzy (semantic search + rerank + LLM).",
)
async def query(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    try:
        pipeline: LegalRAGPipeline = get_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline RAG niedostępny.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Baza wiedzy nie jest jeszcze gotowa. Uruchom /api/v1/ingest "
                "lub skrypt ingest.py, a następnie spróbuj ponownie."
            ),
        ) from exc

    try:
        result = pipeline.query(payload.question, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Błąd podczas przetwarzania zapytania RAG.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Wystąpił nieoczekiwany błąd podczas generowania odpowiedzi.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return QueryResponse(
        answer=result.answer,
        has_sufficient_context=result.has_sufficient_context,
        sources=[
            SourceDocumentResponse(
                act=s.act,
                article=s.article,
                anchor=s.anchor,
                section_context=s.section_context,
                snippet=s.snippet,
                score=s.score,
            )
            for s in result.sources
        ],
        latency_ms=elapsed_ms,
    )


def _run_ingest_job() -> None:
    global _ingest_in_progress
    global _pipeline_instance_note
    try:
        logger.info("Start reindeksacji bazy wiedzy (background task)...")
        build_index()
        # wymuś ponowne załadowanie pipeline'u przy kolejnym zapytaniu
        import retriever as retriever_module

        retriever_module._pipeline_instance = None
        logger.info("Reindeksacja zakończona pomyślnie.")
    except Exception:  # noqa: BLE001
        logger.exception("Reindeksacja nie powiodła się.")
    finally:
        _ingest_in_progress = False


@app.post(
    f"{settings.API_PREFIX}/ingest",
    response_model=IngestResponse,
    tags=["RAG"],
    summary="Uruchamia (asynchronicznie) ponowne zaindeksowanie bazy wiedzy.",
)
async def trigger_ingest(background_tasks: BackgroundTasks) -> IngestResponse:
    global _ingest_in_progress
    if _ingest_in_progress:
        return IngestResponse(
            status="already_running",
            message="Reindeksacja jest już w toku - poczekaj na zakończenie.",
        )
    _ingest_in_progress = True
    background_tasks.add_task(_run_ingest_job)
    return IngestResponse(
        status="started",
        message="Reindeksacja uruchomiona w tle. Sprawdź logi lub /health po zakończeniu.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
