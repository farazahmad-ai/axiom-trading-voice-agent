"""Document Ingestion Pipeline for Axiom Prop Trading Rules using Docling and Qdrant."""

import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

try:
    from prop_trading_rules_voice_agent.config import settings
except ImportError:
    from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Manages document parsing, hierarchical chunking, vector embedding, and Qdrant indexing."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.client = QdrantClient(path=settings.qdrant_path)

    def load_document(self, source: str) -> Any:
        """Parses a local PDF or document using Docling and returns its structural representation."""
        logger.info("Parsing document: %s", source)
        result = self.converter.convert(source)
        return result.document

    def convert_chunk(self, doc_chunk: Any) -> dict[str, Any]:
        """Converts a Docling chunk into a dictionary with prepended breadcrumb navigation."""
        headings = doc_chunk.meta.headings or []
        content = doc_chunk.text.strip()
        breadcrumb = " > ".join(headings)
        chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

        return {
            "headings": headings,
            "content": content,
            "chunk_text": chunk_text,
        }

    def process_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Parses a single PDF and chunks it into hierarchical segments."""
        doc = self.load_document(str(pdf_path))
        chunker = HierarchicalChunker()
        chunks = [self.convert_chunk(c) for c in chunker.chunk(doc)]
        logger.info("Generated %d chunks from %s", len(chunks), pdf_path.name)
        return chunks

    def run(self, docs_dir: str | Path | None = None) -> int:
        """Executes the full ingestion pipeline across all PDFs in the target directory."""
        directory = Path(docs_dir or settings.docs_dir)
        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in '%s'.", directory)
            return 0

        logger.info("Found %d PDF file(s) in '%s':", len(pdf_files), directory)
        for f in pdf_files:
            logger.info("  - %s", f.name)

        # Step 1: Parse and chunk all documents
        all_chunks: list[dict[str, Any]] = []
        for pdf_path in pdf_files:
            try:
                doc_chunks = self.process_pdf(pdf_path)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                logger.error("Failed to parse %s: %s", pdf_path.name, e)
                continue

        if not all_chunks:
            logger.warning("No chunks were produced from documents.")
            return 0

        logger.info("Total chunks collected: %d", len(all_chunks))

        # Step 2: Generate vector embeddings
        logger.info("Generating embeddings using %s...", settings.embedding_model)
        chunk_texts = [c["chunk_text"] for c in all_chunks]
        embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True)
        dim = self.embedder.get_embedding_dimension()

        # Step 3: Recreate Qdrant collection and upload vectors
        logger.info("Indexing vectors into Qdrant collection '%s'...", settings.collection_name)
        existing_collections = [col.name for col in self.client.get_collections().collections]
        if settings.collection_name in existing_collections:
            self.client.delete_collection(collection_name=settings.collection_name)

        self.client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={
                    "headings": chunk["headings"],
                    "content": chunk["content"],
                    "chunk_text": chunk["chunk_text"],
                },
            )
            for idx, chunk in enumerate(all_chunks)
        ]

        self.client.upsert(
            collection_name=settings.collection_name,
            points=points,
            wait=True,
        )

        info = self.client.get_collection(settings.collection_name)
        logger.info(
            "Successfully indexed %d points in '%s' (vector dimension: %d).",
            info.points_count,
            settings.collection_name,
            info.config.params.vectors.size,
        )

        self.client.close()
        return info.points_count


def main() -> None:
    logger.info("=== Axiom Prop Trading Rules — Document Ingestion Pipeline ===")
    logger.info("Docs folder:   %s", settings.docs_dir)
    logger.info("Embedding:     %s", settings.embedding_model)
    logger.info("Qdrant path:   %s", settings.qdrant_path)
    logger.info("Collection:    %s", settings.collection_name)

    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
