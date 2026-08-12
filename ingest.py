"""
ingest.py
=========
Moduł ładowania i indeksowania bazy wiedzy prawnej ActumAI Core.

Odpowiedzialność:
  1. Rekurencyjne wyszukanie plików .md / .mdux w katalogu z danymi.
  2. Parsowanie YAML front-matter (metadane aktu prawnego).
  3. Chunkowanie z zachowaniem struktury artykułów prawnych:
     - jeżeli dokument zawiera nagłówki `## Art. N. {#anchor}` -> chunk = 1 artykuł
       (obowiązki, definicje itp. nie są sztucznie przecinane w połowie artykułu),
     - jeżeli dokument NIE ma takiej struktury (np. opisy proceduralne, "Flows")
       -> fallback: Hierarchical/Sentence Splitter po nagłówkach Markdown.
  4. Zbudowanie embeddingów i zapis do wybranej bazy wektorowej (Chroma/Qdrant).

Uruchomienie:
    python ingest.py --data-dir ./data/Docs_done_rag --collection actumai_legal_kb
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.settings import Settings as LlamaSettings

from config import EmbeddingProvider, VectorStoreProvider, settings

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("actumai.ingest")

# Rozpoznaje nagłówki artykułów typu:
#   ## Art. 6 {#rodo-art-6}
#   ## Art. 123a. {#kp-pierwszy-art-123a}
ARTICLE_HEADER_RE = re.compile(
    r"^#{1,3}\s*Art\.\s*(?P<num>[\w\-]+)\.?\s*(\{#(?P<anchor>[\w\-]+)\})?\s*$",
    re.MULTILINE,
)

# Nagłówki wyższego rzędu (Dział / Rozdział / Oddział), utrzymywane jako
# kontekst hierarchiczny dopisywany do metadanych każdego artykułu poniżej.
SECTION_HEADER_RE = re.compile(
    r"^#{1,3}\s*(?P<label>Dział|Rozdział|Oddział)\s+(?P<value>.+?)\s*$",
    re.MULTILINE,
)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class ParsedDoc:
    frontmatter: dict = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(raw_text: str) -> ParsedDoc:
    """Wydziela blok YAML front-matter (pierwszy) od reszty treści Markdown."""
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        return ParsedDoc(frontmatter={}, body=raw_text)
    try:
        fm = yaml.safe_load(match.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError as exc:
        logger.warning("Nie udało się sparsować front-matter: %s", exc)
        fm = {}
    body = raw_text[match.end():]
    return ParsedDoc(frontmatter=fm, body=body)


def _act_name_from_path(file_path: Path, frontmatter: dict) -> str:
    """Best-effort ustalenie nazwy aktu prawnego (do cytowania w odpowiedziach)."""
    for key in ("title", "law"):
        if frontmatter.get(key):
            return str(frontmatter[key])
    # fallback: nazwa katalogu nadrzędnego (np. "RODO", "KPC", "KP")
    return file_path.parent.name if file_path.parent.name != "Docs_done_rag" else file_path.stem


def split_into_article_documents(
    file_path: Path, parsed: ParsedDoc
) -> list[Document]:
    """
    Dzieli treść dokumentu na chunk = 1 artykuł, zachowując kontekst
    Działu/Rozdziału jako metadane. Zwraca listę llama_index Document.
    Jeśli w dokumencie nie ma żadnego nagłówka artykułu, zwraca pustą listę
    (sygnał dla wywołującego, by użyć fallbacku).
    """
    body = parsed.body
    matches = list(ARTICLE_HEADER_RE.finditer(body))
    if not matches:
        return []

    act_name = _act_name_from_path(file_path, parsed.frontmatter)
    status = parsed.frontmatter.get("status", "nieznany")
    last_consolidated = parsed.frontmatter.get("last_consolidated", "")
    canonical_citation = parsed.frontmatter.get("canonical_citation", "")

    # Zbieramy pozycje nagłówków sekcji (Dział/Rozdział), żeby dla każdego
    # artykułu dociągnąć ostatnią sekcję poprzedzającą go w tekście.
    section_matches = list(SECTION_HEADER_RE.finditer(body))

    def _section_context_before(pos: int) -> str:
        active = [m for m in section_matches if m.start() < pos]
        if not active:
            return ""
        # bierzemy do 2 ostatnich (np. Dział + Rozdział)
        parts = [f"{m.group('label')} {m.group('value')}" for m in active[-2:]]
        return " / ".join(parts)

    documents: list[Document] = []
    for idx, m in enumerate(matches):
        article_num = m.group("num")
        anchor = m.group("anchor") or f"art-{article_num}"
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        article_text = body[start:end].strip().strip("*").strip()

        if not article_text or len(article_text) < 5:
            continue

        section_context = _section_context_before(start)
        metadata = {
            "source_file": str(file_path.name),
            "source_path": str(file_path),
            "act": act_name,
            "article": article_num,
            "anchor": anchor,
            "section_context": section_context,
            "status": status,
            "last_consolidated": str(last_consolidated),
            "canonical_citation": canonical_citation,
            "legal_basis": f"{act_name}, art. {article_num}",
        }
        documents.append(
            Document(
                text=article_text,
                metadata=metadata,
                excluded_llm_metadata_keys=["source_path", "source_file"],
                excluded_embed_metadata_keys=["source_path", "source_file", "anchor"],
            )
        )
    return documents


def build_fallback_documents(file_path: Path, parsed: ParsedDoc) -> list[Document]:
    """
    Fallback dla dokumentów bez struktury artykułowej (np. opisy procedur
    w katalogu Flows/). Traktujemy cały dokument jako jeden Document -
    dalszy podział na chunki wykona SentenceSplitter/MarkdownNodeParser
    na etapie budowy indeksu.
    """
    act_name = _act_name_from_path(file_path, parsed.frontmatter)
    title = parsed.frontmatter.get("title", act_name)
    metadata = {
        "source_file": str(file_path.name),
        "source_path": str(file_path),
        "act": act_name,
        "article": "",
        "anchor": "",
        "section_context": str(title),
        "status": parsed.frontmatter.get("status", "nieznany"),
        "last_consolidated": str(parsed.frontmatter.get("last_verified", "")),
        "canonical_citation": "",
        "legal_basis": act_name,
    }
    body = parsed.body.strip()
    if not body:
        return []
    return [Document(text=body, metadata=metadata)]


def load_documents(data_dir: Path) -> list[Document]:
    """Rekurencyjnie ładuje wszystkie wspierane pliki i zwraca listę Document."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Katalog z danymi nie istnieje: {data_dir}")

    files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in settings.SUPPORTED_EXTENSIONS
    ]
    logger.info("Znaleziono %d plików źródłowych w %s", len(files), data_dir)

    all_documents: list[Document] = []
    article_chunked, fallback_chunked, skipped = 0, 0, 0

    for file_path in files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Pomijam plik (błąd kodowania): %s", file_path)
            skipped += 1
            continue

        parsed = parse_frontmatter(raw_text)
        docs = split_into_article_documents(file_path, parsed)

        if docs:
            article_chunked += 1
        else:
            docs = build_fallback_documents(file_path, parsed)
            fallback_chunked += 1

        if not docs:
            skipped += 1
            continue

        all_documents.extend(docs)

    logger.info(
        "Ładowanie zakończone. Pliki dzielone per-artykuł: %d | fallback: %d | "
        "pominięte: %d | łącznie chunków-dokumentów: %d",
        article_chunked,
        fallback_chunked,
        skipped,
        len(all_documents),
    )
    return all_documents


def get_embedding_model():
    """Zwraca skonfigurowany model embeddingów wg ustawień."""
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI:
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    elif settings.EMBEDDING_PROVIDER == EmbeddingProvider.HUGGINGFACE:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=settings.HF_EMBEDDING_MODEL)
    raise ValueError(f"Nieobsługiwany EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def get_vector_store():
    """Zwraca skonfigurowany vector store (Chroma lokalnie lub Qdrant)."""
    if settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.CHROMA:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        collection = client.get_or_create_collection(settings.COLLECTION_NAME)
        return ChromaVectorStore(chroma_collection=collection)

    elif settings.VECTOR_STORE_PROVIDER == VectorStoreProvider.QDRANT:
        import qdrant_client
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        client = qdrant_client.QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )
        return QdrantVectorStore(
            client=client, collection_name=settings.COLLECTION_NAME
        )

    raise ValueError(f"Nieobsługiwany VECTOR_STORE_PROVIDER: {settings.VECTOR_STORE_PROVIDER}")


def build_index(data_dir: Optional[Path] = None) -> VectorStoreIndex:
    """
    Główna funkcja indeksująca: ładuje dokumenty, dzieli na węzły (nodes),
    liczy embeddingi i zapisuje do bazy wektorowej. Zwraca gotowy index.
    """
    data_dir = data_dir or settings.DATA_DIR

    embed_model = get_embedding_model()
    LlamaSettings.embed_model = embed_model
    # LLM nie jest potrzebny na etapie ingestu - wyłączamy, by uniknąć
    # przypadkowych wywołań/kosztów.
    LlamaSettings.llm = None

    documents = load_documents(data_dir)
    if not documents:
        raise RuntimeError("Brak dokumentów do zaindeksowania - sprawdź DATA_DIR.")

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Dokumenty per-artykuł mają już właściwy rozmiar chunku (1 artykuł),
    # więc nie chcemy ich dalej ciąć - MarkdownNodeParser zachowa je jako
    # pojedyncze węzły, o ile mieszczą się w rozsądnym rozmiarze; dłuższe
    # dokumenty fallbackowe zostaną doprecyzowane przez SentenceSplitter.
    node_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    nodes = []
    for doc in documents:
        if doc.metadata.get("article"):
            # chunk już ma granulację 1 artykuł -> parsujemy tylko strukturę markdown
            nodes.extend(node_parser.get_nodes_from_documents([doc]))
        else:
            nodes.extend(sentence_splitter.get_nodes_from_documents([doc]))

    logger.info("Zbudowano %d węzłów (nodes) do embeddingu.", len(nodes))

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info(
        "Indeksowanie zakończone. Kolekcja: '%s' | Vector store: %s",
        settings.COLLECTION_NAME,
        settings.VECTOR_STORE_PROVIDER.value,
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="ActumAI Core - ingest bazy wiedzy prawnej")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.DATA_DIR,
        help="Katalog root z plikami .md/.mdux",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.COLLECTION_NAME,
        help="Nazwa kolekcji w bazie wektorowej",
    )
    args = parser.parse_args()

    settings.COLLECTION_NAME = args.collection
    build_index(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
