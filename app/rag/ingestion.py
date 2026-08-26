# ingestion.py
from __future__ import annotations

from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.parser import DocumentParser
from rag.cleaner import TextCleaner
from rag.chunker import Chunker, Chunk
from rag.embedder import Embedder, EmbeddedChunk

from rag.loaders.base import DocumentLoadError
from rag.loaders.registry import LoaderRegistry
from rag.loaders.docx_loader import DocxLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.txt_loader import TxtLoader

from llm.embeddings import EmbeddingsClient

from database.vector_store import VectorStoreConfig, get_vector_store  # placeholder for future vector store implementation

@dataclass
class IngestionResult:
    document_id: str
    num_chunks: int
    status: str  # "success" | "failed"
    error: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        loader_registry: LoaderRegistry,
        parser: DocumentParser,
        cleaner: TextCleaner,
        chunker: Chunker,
        embedder: Embedder,
    ):
        self.loader_registry = loader_registry
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder


    def save_embedded_chunks(self, chunks: str, out_dir: str = "logs", prefix: str = "embedded_chunks") -> str:
        """
        Dump a list of EmbeddedChunk objects to a JSON file for inspection/debugging.
        Returns the path written to.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(out_dir) / f"{prefix}_{timestamp}.json"

        serializable = [
            {
                "text": ec.chunk.text,
                "metadata": ec.chunk.metadata,
                "vector_dim": len(ec.vector),
                "vector": ec.vector,
            }
            for ec in chunks
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        return str(out_path)

    def ingest(self, file_path: str | Path, document_id: str | None = None) -> IngestionResult:
        document_id = document_id or str(uuid.uuid4())

        try:
            print(f"Starting ingestion for document_id={document_id}, file_path={file_path}")
            loader = self.loader_registry.get_loader(file_path)
            raw_doc = loader.load(file_path)
            parsed_doc = self.parser.parse(raw_doc)
            cleaned_doc = self.cleaner.clean(parsed_doc)
            chunks = self.chunker.chunk(cleaned_doc)

            if not chunks:
                return IngestionResult(
                    document_id=document_id, num_chunks=0, status="failed",
                    error="No chunks produced (document may be empty after cleaning)",
                )

            embedded_chunks = self.embedder.embed_chunks(chunks)

            print(f"{len(chunks)} chunks from document_id={document_id}.")

            config = VectorStoreConfig(
                vector_size=384,
                backend="qdrant",
                collection_name="forge_documents",
                persist_directory="./qdrant_data",   # local folder where it'll persist to disk
            )

            self.vector_store = get_vector_store(config)

            self.vector_store.add(
                ids=self._generate_ids(document_id, chunks),
                vectors=[ec.vector for ec in embedded_chunks],
                documents=[ec.chunk.text for ec in embedded_chunks],
                metadatas=[ec.chunk.metadata for ec in embedded_chunks],
            )
            _ = embedded_chunks  # keep the reference alive/used until store is wired up
            # ------------------------------------------------------------------------

            self._persist_document_record(document_id, file_path, num_chunks=len(chunks))
            self.save_embedded_chunks(embedded_chunks)
            return IngestionResult(document_id=document_id, num_chunks=len(chunks), status="success")

        except DocumentLoadError as e:
            return IngestionResult(document_id=document_id, num_chunks=0, status="failed", error=str(e))
        except Exception as e:
            return IngestionResult(document_id=document_id, num_chunks=0, status="failed", error=str(e))

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_ids(document_id: str, chunks: list[Chunk]) -> list[str]:
        """id scheme: <document_id>::<chunk_index>, stable and collision-free
        within a document, easy to reconstruct for delete/update later."""
        return [f"{document_id}::{i}" for i in range(len(chunks))]

    @staticmethod
    def _persist_document_record(document_id: str, file_path: str | Path, num_chunks: int) -> None:
        # from app.database.models import persist_document_record
        # persist_document_record(document_id, str(file_path), num_chunks=num_chunks)
        pass

_pipeline_instance: IngestionPipeline | None = None

def get_ingestion_pipeline() -> IngestionPipeline:
    """Singleton-style accessor so FastAPI dependencies reuse one pipeline
    instead of rebuilding loaders/clients on every request."""
    global _pipeline_instance
    if _pipeline_instance is None:
        loader_registry = LoaderRegistry([PDFLoader(), DocxLoader(), TxtLoader()])  # register pdf/docx/txt loaders here
        parser = DocumentParser()
        cleaner = TextCleaner()
        chunker = Chunker(chunk_size=800, chunk_overlap=100)  # match your Chunker's real signature
        embedder = Embedder(EmbeddingsClient(model_name="all-MiniLM-L6-v2"))

        _pipeline_instance = IngestionPipeline(
            loader_registry=loader_registry,
            parser=parser,
            cleaner=cleaner,
            chunker=chunker,
            embedder=embedder,
        )
    return _pipeline_instance

def main():
    pipeline = get_ingestion_pipeline()
    print("Ingestion pipeline initialized. Ready to ingest documents.")
    result = pipeline.ingest(r"C:\Users\Admin\OneDrive\Desktop\Documents\Signed Offer Letter.pdf", document_id="test-doc-001")

    print(result)
    if result.status == "failed":
        raise SystemExit(f"Ingestion failed: {result.error}")
