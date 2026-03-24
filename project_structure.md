# Agentic RAG Platform — Project Structure

```
agentic-rag-platform/
│
├── app/                        # Core application logic
│   ├── __init__.py
│   ├── agent.py                # LangGraph agent definition (nodes + edges)
│   ├── retriever.py            # Vertex AI Vector Search interface
│   ├── tools.py                # Agent tools (search, calculator, etc.)
│   └── prompts.py              # All LLM prompt templates
│
├── ingestion/                  # Data pipeline: PDF/text → Vector DB
│   ├── __init__.py
│   ├── ingest.py               # Chunking + embedding + upload to Vertex AI
│   └── gcs_loader.py           # Load documents from Google Cloud Storage
│
├── api/                        # FastAPI server (Cloud Run endpoint)
│   ├── __init__.py
│   └── main.py                 # POST /query endpoint
│
├── tests/                      # Unit + integration tests
│   ├── test_agent.py
│   └── test_retriever.py
│
├── .env.example                # Template for environment variables (never commit .env)
├── .gitignore
├── Dockerfile                  # Container definition for Cloud Run
├── requirements.txt            # Python dependencies
├── project_structure.md        # This file
└── README.md                   # Project overview for your resume/GitHub
```

## Data Flow

1. **Ingestion** (offline): Documents → GCS → chunked → embedded → Vertex AI Vector Search index
2. **Query** (online):  User → FastAPI → LangGraph Agent → retrieve from Vector Search → LLM → response

## GCP Services Used

| Service | Role |
|---|---|
| Cloud Run | Hosts the FastAPI API (serverless, auto-scales) |
| Cloud Storage (GCS) | Stores raw documents |
| Vertex AI Vector Search | High-performance vector similarity search |
| Vertex AI (Gemini) | LLM for reasoning and answer generation |
| Artifact Registry | Stores Docker images |
| Cloud Build | CI/CD — auto-builds Docker image on git push |
