# Palmmind RAG API

Production-oriented FastAPI backend for document ingestion, Retrieval-Augmented Generation (RAG), and interview booking extraction.

## Project Overview

This project provides a modular backend that:

- Uploads and processes PDF/TXT documents
- Stores document metadata in PostgreSQL
- Generates embeddings with Sentence Transformers
- Uses Qdrant for vector similarity search
- Uses Redis for chat memory by `session_id`
- Runs a custom RAG pipeline (without RetrievalQAChain)
- Detects interview-booking intent and extracts structured fields via LLM

## Architecture

### High-level Flow

1. **Document Upload**
   - Client uploads PDF/TXT + `chunk_strategy`
   - API validates file and strategy
   - Text is extracted
   - Metadata is stored in PostgreSQL

2. **Chat / RAG**
   - User sends `session_id` and `message`
   - Previous chat history is loaded from Redis
   - Query is embedded
   - Top matching chunks are retrieved from Qdrant
   - Prompt is built manually
   - Prompt is sent to OpenAI for final answer
   - Response and booking extraction JSON are returned
   - Conversation turns are persisted in Redis

3. **Interview Booking Extraction**
   - LLM detects booking intent
   - Extracts `name`, `email`, `date`, `time`
   - If complete and valid, booking is saved to PostgreSQL

## Tech Stack

- **API Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **ORM / DB Layer:** SQLAlchemy (async)
- **Database:** PostgreSQL
- **Cache / Memory:** Redis
- **Vector DB:** Qdrant
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **File Parsing:** pypdf
- **LLM SDK:** OpenAI Python SDK
- **Validation:** Pydantic
- **Containerization:** Docker + Docker Compose

## Setup Instructions

### 1) Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Qdrant
- OpenAI API key

### 2) Clone and Install

```bash
git clone <your-repo-url>
cd Palmmind-document-ingestion-and-rag-api
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Environment Variables

Create a `.env` file in the project root:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db

REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 4) Run the API (Local)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API base URL: `http://localhost:8000`

## Docker Instructions

### Run with Docker Compose

```bash
docker compose up --build
```

Services started:

- API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- Qdrant: `http://localhost:6333`

### Stop Services

```bash
docker compose down
```

### Remove volumes (optional)

```bash
docker compose down -v
```

## API Documentation

Interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Health Check

- **GET** `/health`
- **Response:**

```json
{
  "status": "ok"
}
```

### Upload Document

- **POST** `/api/v1/documents/upload`
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: PDF or TXT
  - `chunk_strategy`: one of `fixed`, `recursive`, `semantic`

- **Success (201):**

```json
{
  "id": 1,
  "filename": "sample.pdf",
  "upload_date": "2026-06-26T10:45:31.250000+00:00",
  "chunk_strategy": "fixed",
  "total_chunks": 14
}
```

- **Error Codes:** `400`, `415`, `422`, `500`

### Chat

- **POST** `/api/v1/chat`
- **Request:**

```json
{
  "session_id": "session-123",
  "message": "I want to book an interview for next Monday at 10:30 AM. I am Jane Doe, jane@example.com"
}
```

- **Success (200):**

```json
{
  "answer": "Sure, I can help with that...",
  "session_id": "session-123",
  "booking": {
    "wants_booking": true,
    "name": "Jane Doe",
    "email": "jane@example.com",
    "date": "2026-06-30",
    "time": "10:30",
    "saved_booking_id": 5
  }
}
```

- **Error Codes:** `422`, `500`

## Example Requests

### 1) Upload a TXT File

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -F "file=@./sample.txt;type=text/plain" \
  -F "chunk_strategy=fixed"
```

### 2) Upload a PDF File

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -F "file=@./sample.pdf;type=application/pdf" \
  -F "chunk_strategy=recursive"
```

### 3) Ask a Chat Question

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-abc",
    "message": "Summarize the uploaded onboarding policy"
  }'
```

### 4) Interview Booking Intent Example

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "booking-001",
    "message": "Book an interview for John Smith (john@acme.com) on 2026-07-05 at 14:00"
  }'
```

## Folder Structure

```text
.
├── app
│   ├── api
│   │   └── v1
│   │       ├── dependencies
│   │       │   ├── __init__.py
│   │       │   └── database.py
│   │       ├── endpoints
│   │       │   ├── __init__.py
│   │       │   ├── chat.py
│   │       │   └── documents.py
│   │       ├── __init__.py
│   │       └── router.py
│   ├── core
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── session.py
│   ├── integrations
│   │   ├── postgres
│   │   ├── qdrant
│   │   │   ├── __init__.py
│   │   │   └── service.py
│   │   └── redis
│   │       ├── __init__.py
│   │       └── memory.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── document.py
│   │   └── interview_booking.py
│   ├── rag
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── document.py
│   │   └── interview_booking.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── interview_booking.py
│   │   └── text_extraction.py
│   └── main.py
├── alembic
│   └── versions
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Notes

- This repository currently defines models/services for ingestion and booking, but full migration scripts are not yet included.
- Ensure PostgreSQL schema/migrations are applied before production deployment.
- For production, configure secrets via secure secret management instead of plain `.env` files.
