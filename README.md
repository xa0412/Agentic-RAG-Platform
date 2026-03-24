# Agentic RAG Platform

An autonomous research agent that answers questions from your documents. Built with LangGraph, Gemini, and deployed on Google Cloud Run.

**Live Demo:** https://agentic-rag-platform-21171800375.us-central1.run.app/docs

---

## What It Does

Upload any PDF or text document, then ask questions about it. The agent retrieves the most relevant sections, generates a grounded answer, grades its own output, and retries if the quality is poor — all autonomously.

**Example:**
```
POST /ingest  →  upload company_report.pdf
POST /query   →  "What were the top reasons for customer churn in Q3?"
              ←  Answer with cited sources
```

---

## Architecture

```
User
 │
 ▼
FastAPI (Cloud Run)
 │
 ├── POST /ingest ──► Chunk document ──► Vertex AI Embeddings ──► Vector Store
 │
 └── POST /query ───► LangGraph Agent
                           │
                      [retrieve] ◄──────────────────┐
                           │                        │ retry if
                      [generate]               answer poor
                           │                        │
                      [check_answer] ───────────────┘
                           │
                        answer + sources
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Gemini 2.5 Flash (Google AI Studio) |
| Embeddings | Vertex AI text-embedding-004 |
| Vector Store | LangChain InMemoryVectorStore |
| API | FastAPI + Uvicorn |
| Containerisation | Docker (two-stage build) |
| Deployment | Google Cloud Run |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API playground (Swagger UI) |
| `POST` | `/ingest` | Upload a `.pdf` or `.txt` file |
| `POST` | `/query` | Ask a question, get an answer + sources |

### Query Example

```bash
curl -X POST https://agentic-rag-platform-21171800375.us-central1.run.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph?"}'
```

```json
{
  "answer": "LangGraph is a framework for building agentic workflows as state machines...",
  "sources": ["document.pdf"]
}
```

---

## Run Locally

**Prerequisites:** Python 3.11+, GCP account, Google AI Studio API key

```bash
# Clone and install
git clone https://github.com/xa0412/Agentic-RAG-Platform.git
cd Agentic-RAG-Platform
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in your values in .env

# Authenticate with GCP (for embeddings)
gcloud auth application-default login

# Start the server
uvicorn api.main:app --reload --port 8080
```

Then open `http://127.0.0.1:8080/docs`

---

## Deploy to Cloud Run

```bash
gcloud run deploy agentic-rag-platform \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --max-instances 2 \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT,GCP_REGION=us-central1,GEMINI_API_KEY=YOUR_KEY"
```

---

## Environment Variables

See `.env.example` for the full list.

| Variable | Description |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | GCP region (`us-central1` recommended) |
| `GEMINI_API_KEY` | Google AI Studio API key — [get one free](https://aistudio.google.com/apikey) |
