# Agentic RAG Platform — Project Structure

```
agentic-rag-platform/
│
├── app/                        # Core application logic
│   ├── __init__.py
│   ├── agent.py                # LangGraph agent (retrieve → generate → grade loop)
│   ├── retriever.py            # Vector store interface (InMemoryVectorStore + Vertex AI embeddings)
│   ├── llm.py                  # Gemini 2.5 Flash (Google AI Studio) + Vertex AI embeddings
│   └── prompts.py              # LLM prompt templates (generate + answer grader)
│
├── ingestion/                  # Document ingestion pipeline
│   ├── __init__.py
│   └── ingest.py               # Load PDF/TXT → chunk → embed → store
│
├── api/                        # FastAPI server
│   ├── __init__.py
│   └── main.py                 # POST /query, POST /ingest, GET /health
│
├── tests/                      # Unit + integration tests (future)
│   └── __init__.py
│
├── .env.example                # Template for required environment variables
├── .gitignore                  # Excludes .env, venv/, __pycache__/
├── .gcloudignore               # Excludes venv/ from Cloud Build uploads
├── Dockerfile                  # Two-stage container build for Cloud Run
├── requirements.txt            # Python dependencies
├── project_structure.md        # This file
└── README.md                   # Project overview (coming soon)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — used by Cloud Run liveness probe |
| GET | `/docs` | Auto-generated Swagger UI (FastAPI) |
| POST | `/ingest` | Upload a PDF or TXT file → chunks → vector store |
| POST | `/query` | Ask a question → LangGraph agent → answer + sources |

## Agent Flow (LangGraph State Machine)

```
User Question
     ↓
[retrieve] — semantic search via Vertex AI embeddings
     ↓
[generate] — Gemini 2.5 Flash reads chunks, writes answer
     ↓
[check_answer] — Gemini grades its own answer
     ↓              ↓
    YES             NO (retry, max 2x)
     ↓
Final Answer + Sources
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Gemini 2.5 Flash (Google AI Studio) |
| Embeddings | Vertex AI text-embedding-004 |
| Vector Store | LangChain InMemoryVectorStore |
| API | FastAPI + Uvicorn |
| Containerisation | Docker (two-stage build) |
| Deployment | Google Cloud Run (serverless, scales to zero) |
| PDF Parsing | pypdf |
| Chunking | LangChain RecursiveCharacterTextSplitter |

## GCP Services Used

| Service | Role |
|---|---|
| Cloud Run | Hosts the FastAPI API (serverless, auto-scales to zero) |
| Vertex AI | text-embedding-004 for semantic embeddings |
| Cloud Build | Builds Docker image on deploy |
| Artifact Registry | Stores Docker images |

## Environment Variables Required

See `.env.example` for the full list. Key variables:

| Variable | Description |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | GCP region (us-central1 recommended) |
| `GEMINI_API_KEY` | Google AI Studio API key (free tier) |
