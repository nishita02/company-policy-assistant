import os
import re
import uuid
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "data" / "docs"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

COLLECTION_NAME = "company_policies"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(encoding.encode(text))


def read_markdown_files():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def split_markdown_sections(text):
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())

        if heading_match:
            if current_lines:
                sections.append(
                    {
                        "heading": current_heading,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []

            current_heading = heading_match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def split_large_text(text, max_tokens=250, overlap_tokens=40):
    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = encoding.decode(tokens[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        start = max(0, end - overlap_tokens)

    return chunks


def chunk_section(section_text, max_tokens=250, overlap_tokens=40):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    chunks = []
    current_parts = []

    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []

            chunks.extend(split_large_text(paragraph, max_tokens, overlap_tokens))
            continue

        candidate_parts = current_parts + [paragraph]
        candidate_text = "\n\n".join(candidate_parts)

        if count_tokens(candidate_text) <= max_tokens:
            current_parts.append(paragraph)
        else:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def build_contextual_chunks(document, max_tokens=250, overlap_tokens=40):
    contextual_chunks = []
    sections = split_markdown_sections(document["text"])

    for section in sections:
        chunks = chunk_section(section["text"], max_tokens, overlap_tokens)

        for chunk in chunks:
            contextual_text = (
                f"Document: {document['source']}\n"
                f"Section: {section['heading']}\n\n"
                f"{chunk}"
            )
            contextual_chunks.append(
                {
                    "text": contextual_text,
                    "source": document["source"],
                    "section": section["heading"],
                }
            )

    return contextual_chunks


def embed_texts(texts):
    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def build_vector_database():
    documents = read_markdown_files()

    if not documents:
        raise RuntimeError(f"No markdown files found in {DOCS_DIR}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    ids = []
    texts = []
    metadatas = []

    for document in documents:
        chunks = build_contextual_chunks(document)

        for chunk_number, chunk in enumerate(chunks, start=1):
            source_stem = Path(chunk["source"]).stem
            chunk_id = f"{source_stem}_{chunk_number:03d}_{uuid.uuid4().hex[:8]}"

            ids.append(chunk_id)
            texts.append(chunk["text"])
            metadatas.append(
                {
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "chunk_number": chunk_number,
                    "token_count": count_tokens(chunk["text"]),
                }
            )

    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(texts)} chunks")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector database saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_database()
