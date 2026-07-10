import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

COLLECTION_NAME = "company_policies"
EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)


def embed_query(query):
    return embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()


def get_collection():
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Vector database not found at {CHROMA_DIR}. "
            "Run `python -m app.ingest` before querying."
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_collection(name=COLLECTION_NAME)


def retrieve_relevant_chunks(query, top_k=5):
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    collection = get_collection()
    query_embedding = embed_query(query.strip())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        chunks.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "chunk_number": metadata.get("chunk_number"),
                "distance": distance,
            }
        )

    return chunks


def format_chunk_for_display(chunk):
    source = chunk.get("source") or "unknown source"
    section = chunk.get("section") or "unknown section"
    chunk_number = chunk.get("chunk_number") or "?"
    distance = chunk.get("distance")

    distance_text = f"{distance:.4f}" if isinstance(distance, float) else "n/a"

    return (
        f"Source: {source}\n"
        f"Section: {section}\n"
        f"Chunk: {chunk_number}\n"
        f"Distance: {distance_text}\n\n"
        f"{chunk['text']}"
    )


if __name__ == "__main__":
    question = input("Question: ").strip()
    matches = retrieve_relevant_chunks(question)

    for result_number, chunk in enumerate(matches, start=1):
        print(f"\n--- Result {result_number} ---")
        print(format_chunk_for_display(chunk))
