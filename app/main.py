import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.retrieval import retrieve_relevant_chunks


load_dotenv()

LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


app = FastAPI(title="Company Policy Assistant")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    source: str | None
    section: str | None
    distance: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def build_context(chunks):
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or "unknown source"
        section = chunk.get("section") or "unknown section"

        context_parts.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def ask_ollama(question, context):
    system_prompt = (
        "You are a company policy assistant. Answer using only the provided "
        "policy context. If the context does not contain the answer, say that "
        "you could not find the answer in the available policy documents. "
        "Keep answers clear and practical. Cite policy names and section names, "
        "but do not mention internal context numbers, chunk numbers, retrieval "
        "distances, or implementation details."
    )
    user_prompt = f"Policy context:\n\n{context}\n\nQuestion: {question}"

    payload = {
        "model": LOCAL_CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running and `{LOCAL_CHAT_MODEL}` is pulled."
            ),
        ) from error

    return data.get("message", {}).get("content", "").strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chat_model": LOCAL_CHAT_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chunks = retrieve_relevant_chunks(request.question, top_k=request.top_k)

    if not chunks:
        return AskResponse(
            answer="I could not find relevant policy context for that question.",
            sources=[],
        )

    answer = ask_ollama(request.question, build_context(chunks))

    sources = [
        Source(
            source=chunk.get("source"),
            section=chunk.get("section"),
            distance=chunk.get("distance"),
        )
        for chunk in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
