"""
retriever.py
============
Rdzeń pipeline'u RAG dla ActumAI Core (Anti-Hallucination Pipeline).

Przepływ zapytania:
  1. Semantic Search: pobranie `SIMILARITY_TOP_K` najbliższych fragmentów
     z bazy wektorowej.
  2. Re-ranking: przecięcie do `RERANK_TOP_N` najlepszych fragmentów
     (SentenceTransformerRerank lub Cohere Rerank).
  3. Circuit breaker: jeżeli najlepszy fragment po re-rankingu ma score
     poniżej `MIN_RERANK_SCORE`, zwracamy formułkę o braku informacji
     BEZ wywoływania LLM - to gwarantuje "zero halucynacji" niezależnie
     od tego, czy model zastosuje się do system promptu.
  4. Generacja: restrykcyjny system prompt wymuszający odpowiedź
     wyłącznie na podstawie dostarczonego kontekstu, z podaniem
     dokładnej podstawy prawnej (akt + artykuł).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings as LlamaSettings

from config import EmbeddingProvider, RerankProvider, settings
from ingest import get_embedding_model, get_vector_store

logger = logging.getLogger("actumai.retriever")

# ---------------------------------------------------------------------- #
# Restrykcyjny system prompt / QA template
# ---------------------------------------------------------------------- #
LEGAL_QA_TEMPLATE = PromptTemplate(
    "Jesteś asystentem prawnym systemu ActumAI Core. Odpowiadasz WYŁĄCZNIE "
    "na podstawie fragmentów aktów prawnych dostarczonych poniżej w sekcji "
    "KONTEKST. Nigdy nie korzystasz z wiedzy spoza kontekstu, nawet jeśli "
    "znasz odpowiedź skądinąd.\n\n"
    "ZASADY BEZWZGLĘDNE:\n"
    "1. Jeżeli w KONTEKŚCIE nie ma informacji wystarczającej do udzielenia "
    "precyzyjnej odpowiedzi, odpowiedz DOKŁADNIE tym zdaniem i niczym więcej: "
    f"\"{settings.NO_CONTEXT_RESPONSE}\"\n"
    "2. Każde twierdzenie prawne MUSI być poparte dokładną podstawą prawną "
    "w formacie: [Nazwa aktu, art. X] - dane te znajdziesz w metadanych "
    "fragmentów (act, article).\n"
    "3. Nie zgaduj, nie interpoluj, nie uogólniaj przepisów, których nie ma "
    "w kontekście. Nie doradzaj poza literą przytoczonych przepisów.\n"
    "4. Jeśli kontekst zawiera fragmenty częściowo powiązane, ale niewystarczające "
    "do precyzyjnej odpowiedzi, zastosuj zasadę nr 1.\n"
    "5. Odpowiadaj w języku polskim, zwięźle i precyzyjnie, w stylu prawniczym.\n\n"
    "---------------------\n"
    "KONTEKST:\n"
    "{context_str}\n"
    "---------------------\n\n"
    "PYTANIE: {query_str}\n\n"
    "ODPOWIEDŹ (z podstawą prawną):"
)


@dataclass
class SourceDocument:
    act: str
    article: str
    anchor: str
    snippet: str
    score: float
    section_context: str = ""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceDocument] = field(default_factory=list)
    has_sufficient_context: bool = True


def get_reranker():
    """Zwraca skonfigurowany node postprocessor do re-rankingu."""
    if settings.RERANK_PROVIDER == RerankProvider.SENTENCE_TRANSFORMER:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=settings.RERANK_MODEL,
            top_n=settings.RERANK_TOP_N,
        )
    elif settings.RERANK_PROVIDER == RerankProvider.COHERE:
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        return CohereRerank(
            api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_RERANK_MODEL,
            top_n=settings.RERANK_TOP_N,
        )
    raise ValueError(f"Nieobsługiwany RERANK_PROVIDER: {settings.RERANK_PROVIDER}")


def get_llm():
    """Zwraca skonfigurowany model LLM do generacji odpowiedzi."""
    if settings.LLM_PROVIDER == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
    raise ValueError(f"Nieobsługiwany LLM_PROVIDER: {settings.LLM_PROVIDER}")


class LegalRAGPipeline:
    """
    Pipeline RAG dla bazy wiedzy prawnej. Ładuje istniejący (wcześniej
    zbudowany przez ingest.py) index z bazy wektorowej i wystawia
    metodę `.query()` z pełnym anti-hallucination flow.
    """

    def __init__(self) -> None:
        logger.info("Inicjalizacja LegalRAGPipeline...")
        self.embed_model = get_embedding_model()
        self.llm = get_llm()
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = self.llm

        vector_store = get_vector_store()
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model
        )

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.SIMILARITY_TOP_K,
        )
        self.reranker = get_reranker()
        # Dodatkowy, "twardy" filtr - odrzuca węzły poniżej progu podobieństwa
        # jeszcze przed re-rankingiem (tania optymalizacja + drugi bezpiecznik).
        self.similarity_cutoff = SimilarityPostprocessor(similarity_cutoff=0.05)

        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=LEGAL_QA_TEMPLATE,
            response_mode="compact",
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[self.similarity_cutoff, self.reranker],
        )
        logger.info(
            "Pipeline gotowy. top_k=%d, rerank_top_n=%d, min_rerank_score=%.2f",
            settings.SIMILARITY_TOP_K,
            settings.RERANK_TOP_N,
            settings.MIN_RERANK_SCORE,
        )

    def _nodes_to_sources(self, nodes: list[NodeWithScore]) -> list[SourceDocument]:
        sources = []
        for n in nodes:
            meta = n.node.metadata or {}
            sources.append(
                SourceDocument(
                    act=str(meta.get("act", "nieznany akt")),
                    article=str(meta.get("article", "")),
                    anchor=str(meta.get("anchor", "")),
                    snippet=n.node.get_content()[:500],
                    score=float(n.score) if n.score is not None else 0.0,
                    section_context=str(meta.get("section_context", "")),
                )
            )
        return sources

    def query(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        """
        Wykonuje pełny cykl RAG dla pytania użytkownika.

        1. Retrieval (opcjonalnie z nadpisanym top_k).
        2. Circuit breaker na podstawie score po re-rankingu.
        3. Generacja odpowiedzi z restrykcyjnym promptem (jeśli kontekst
           jest wystarczający).
        """
        if not question or not question.strip():
            raise ValueError("Pytanie nie może być puste.")

        retriever = self.retriever
        if top_k and top_k != settings.SIMILARITY_TOP_K:
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)

        retrieved_nodes = retriever.retrieve(question)
        retrieved_nodes = self.similarity_cutoff.postprocess_nodes(retrieved_nodes)
        reranked_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes, query_str=question
        )

        if not reranked_nodes:
            logger.info("Brak trafień w bazie wiedzy dla pytania: %r", question)
            return RAGAnswer(
                answer=settings.NO_CONTEXT_RESPONSE,
                sources=[],
                has_sufficient_context=False,
            )

        top_score = reranked_nodes[0].score or 0.0
        if top_score < settings.MIN_RERANK_SCORE:
            logger.info(
                "Circuit breaker: top_score=%.4f < MIN_RERANK_SCORE=%.4f - "
                "zwracam formułkę bez wywołania LLM.",
                top_score,
                settings.MIN_RERANK_SCORE,
            )
            return RAGAnswer(
                answer=settings.NO_CONTEXT_RESPONSE,
                sources=self._nodes_to_sources(reranked_nodes),
                has_sufficient_context=False,
            )

        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=LEGAL_QA_TEMPLATE,
            response_mode="compact",
        )
        response = response_synthesizer.synthesize(query=question, nodes=reranked_nodes)

        answer_text = str(response).strip()
        has_context = settings.NO_CONTEXT_RESPONSE not in answer_text

        return RAGAnswer(
            answer=answer_text,
            sources=self._nodes_to_sources(reranked_nodes),
            has_sufficient_context=has_context,
        )


_pipeline_instance: Optional[LegalRAGPipeline] = None


def get_pipeline() -> LegalRAGPipeline:
    """Singleton - pipeline ładuje modele/index tylko raz per proces."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = LegalRAGPipeline()
    return _pipeline_instance
