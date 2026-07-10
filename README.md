# Company Policy Assistant

A local retrieval-augmented generation (RAG) assistant for answering questions from internal company policy documents.

The project currently runs fully on local/open-source components:

- FastAPI for the backend API
- Chroma for the local vector database
- Sentence Transformers with `BAAI/bge-small-en-v1.5` for embeddings
- Ollama with `llama3.2:3b` for local answer generation
- Markdown files as the policy knowledge base

The current version is designed as a practical prototype: it keeps data local, avoids paid API dependencies, and demonstrates the full RAG workflow end to end.

## Current Workflow

The assistant works in two phases: ingestion and question answering.

### 1. Policy Document Ingestion

Policy files are stored in:

```text
data/docs/
```

The ingestion script:

1. Reads each Markdown policy document.
2. Splits the documents into policy sections.
3. Breaks large sections into smaller chunks.
4. Adds document and section metadata to each chunk.
5. Creates embeddings for each chunk using `BAAI/bge-small-en-v1.5`.
6. Stores the chunks, metadata, and vectors in Chroma.

Run ingestion with:

```powershell
.\.venv\Scripts\python.exe -m app.ingest
```

This creates the local vector database at:

```text
data/chroma_db/
```

That folder is ignored by Git because it is generated data.

### 2. Question Answering

When a user asks a question through the API:

1. The question is embedded using the same embedding model.
2. Chroma searches for the most relevant policy chunks.
3. The retrieved chunks are passed to the local Ollama chat model.
4. The model answers using only the retrieved policy context.
5. The API returns the answer and the source policy sections.

The API endpoint is:

```text
POST /ask
```

Example request:

```json
{
  "question": "How long does temporary privileged access last?",
  "top_k": 3
}
```

## Why These Choices

### FastAPI

FastAPI is lightweight, easy to run locally, and gives automatic interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

It is a good fit for a small assistant backend because the request and response models can be clearly defined with Pydantic.

### Chroma

Chroma is simple to use for a local vector database. It works well for a prototype because it can persist vectors directly to disk without requiring a separate hosted database service.

For this stage, local persistence is enough. In a production system, this could later move to a managed vector database or a more scalable storage layer.

### `BAAI/bge-small-en-v1.5`

This project uses `BAAI/bge-small-en-v1.5` for embeddings because it is small, free to use, and strong enough for policy-document retrieval. It also runs locally through Sentence Transformers, so policy text does not need to be sent to an external embedding API.

The tradeoff is that local embedding generation may be slower than a hosted embedding API, especially on lower-powered machines.

### Ollama and `llama3.2:3b`

Ollama makes it straightforward to run a local chat model. `llama3.2:3b` is small enough for local development while still being capable of answering straightforward policy questions from retrieved context.

The main tradeoff is speed. Local LLM generation can be slower than hosted models, especially without a GPU.

### Markdown Policy Files

Markdown keeps the knowledge base easy to inspect, edit, and version-control. It also makes the ingestion logic simpler because the policy documents are plain text.

## Local Setup

Create and activate a Python virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Create a `.env` file from `.env.example`:

```text
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LOCAL_CHAT_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

Pull the local chat model:

```powershell
ollama pull llama3.2:3b
```

Build the vector database:

```powershell
.\.venv\Scripts\python.exe -m app.ingest
```

Start the API:

```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
app/
  __init__.py
  ingest.py       # Reads policy docs, chunks them, embeds them, and builds Chroma DB
  retrieval.py    # Embeds user questions and retrieves matching Chroma chunks
  main.py         # FastAPI app and /ask endpoint

data/
  docs/           # Source policy Markdown files
  chroma_db/      # Generated local vector database, ignored by Git

.env.example      # Example local configuration
requirements.txt  # Python dependencies
```

## Current Limitations

- The app is currently local-first.
- Ollama must be installed and running on the same machine.
- The Chroma database is stored locally on disk.
- Local LLM responses can be slow compared with hosted APIs.
- There is no frontend yet.
- There is no authentication or user management yet.

## Future Plans

### Frontend

A future version will add a simple chat interface so users can ask questions without using the FastAPI docs or `curl`.

The likely first version will be a local frontend that calls the FastAPI backend:

```text
Browser UI -> FastAPI -> Chroma retrieval -> Ollama answer
```

### AWS Deployment

The first AWS deployment plan is to keep the current architecture and host it on a single EC2 instance:

```text
User browser
  -> frontend
  -> FastAPI backend on EC2
  -> local Chroma database on EC2 disk
  -> Ollama running on EC2
```

This approach is suitable for a demo or early prototype because it closely matches the local development setup.

For a more production-ready deployment, the architecture could evolve into:

- S3 and CloudFront for the frontend
- EC2, ECS, or another container service for the FastAPI backend
- Persistent storage for the vector database
- A managed model provider for faster and more reliable inference
- Authentication and role-based access controls

### Optional OpenAI Integration

The current implementation intentionally uses free/local models. In the future, OpenAI models could be added as an optional provider for better speed, quality, and reliability.

The RAG workflow would remain the same:

```text
policy documents -> chunks -> embeddings -> vector search -> retrieved context -> answer
```

Only the model providers would change:

- Local embeddings could be replaced with OpenAI embeddings.
- Ollama chat generation could be replaced with an OpenAI chat model.

If OpenAI embeddings are used later, the Chroma database must be rebuilt because embeddings from different models are not compatible.

## Repository Notes

The following local files are intentionally ignored by Git:

```text
.env
.venv/
__pycache__/
*.pyc
data/chroma_db/
data/index.json
```

This keeps secrets, local environments, Python cache files, and generated vector databases out of the repository.
