"""Retriever module for querying the Axiom Prop Trading Rules Qdrant vector store."""

import sys
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from prop_trading_rules_voice_agent.config import settings

# Global singletons for fast re-use across voice requests
_embedder: SentenceTransformer | None = None
_client: QdrantClient | None = None


def get_retriever_components() -> tuple[QdrantClient, SentenceTransformer]:
    """Returns initialized QdrantClient and SentenceTransformer singletons."""
    global _embedder, _client
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    if _client is None:
        _client = QdrantClient(path=settings.qdrant_path)
    return _client, _embedder


def retrieve(
    query: str,
    top_k: int = 4,
    score_threshold: float = 0.40,
    client: QdrantClient | None = None,
    embedder: SentenceTransformer | None = None,
) -> list[dict[str, Any]]:
    """
    Embeds the user query and returns the top-k most similar rule chunks from Qdrant.

    Args:
        query: The trader's question.
        top_k: Maximum number of chunks to return.
        score_threshold: Minimum cosine similarity score (0.0 to 1.0).
        client: Optional pre-configured Qdrant client.
        embedder: Optional pre-configured SentenceTransformer.

    Returns:
        List of chunk dictionaries with 'score', 'headings', 'content', and 'chunk_text'.
    """
    if client is None or embedder is None:
        default_client, default_embedder = get_retriever_components()
        client = client or default_client
        embedder = embedder or default_embedder

    query_vector = embedder.encode(query).tolist()

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    results: list[dict[str, Any]] = []
    for hit in hits.points:
        payload = hit.payload or {}
        results.append(
            {
                "score": round(hit.score, 4),
                "headings": payload.get("headings", []),
                "content": payload.get("content", ""),
                "chunk_text": payload.get("chunk_text", ""),
            }
        )

    return results


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clear, structured context string for the LLM."""
    if not retrieved_chunks:
        return ""

    context_blocks: list[str] = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        headings_str = " > ".join(chunk["headings"]) if chunk["headings"] else "General"
        block = f"[Source {i}: {headings_str}] (Score: {chunk['score']})\n{chunk['chunk_text']}"
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def main() -> None:
    """CLI tester for retriever verification."""
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the maximum daily drawdown limit?"

    print(f"\n--- Testing Retriever ---")
    print(f"Query: '{query}'\n")

    chunks = retrieve(query, top_k=3)
    if not chunks:
        print("No matching rules found above the confidence threshold.")
        return

    print(f"Found {len(chunks)} matching chunks:")
    context = build_context(chunks)
    print(context)


if __name__ == "__main__":
    main()

