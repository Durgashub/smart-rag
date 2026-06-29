# RAG App — Full Stack (FastAPI + React)

A document Q&A app: upload documents, ask questions, get answers grounded
in your files, and delete documents you no longer want indexed. No
database required — see the "Why no database" section below.

```
rag-app/
├── backend/      FastAPI server (the API + RAG logic)
└── frontend/     React (Vite) UI
```

## 1. Run the backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste in your OPENAI_API_KEY
uvicorn main:app --reload --port 8000
```

Leave this running. Check it's alive by visiting
http://localhost:8000/api/health — you should see `{"status":"ok"}`.

## 2. Run the frontend

In a **second** terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL it prints (usually http://localhost:5173).

## 3. Use it

- Click **+ Add document** to upload a `.txt`, `.md`, or `.pdf` file.
- Click the **×** next to any file to remove it.
- Type a question in the box at the bottom and hit **Ask**.

Every upload or delete rebuilds the search index automatically — there's
no separate "reindex" button to remember to click.

## Why no database

- **Vector search** is handled by a FAISS index file
  (`backend/vector_store/index.faiss`) — this is your search index, no
  database server needed.
- **File storage** is just the filesystem
  (`backend/docs/`) — listing files is `os.listdir()`, no "files table"
  needed.
- **Deleting** a file triggers a full index rebuild from whatever's left
  in `docs/`. This is the simplest correct approach and is fast enough
  for a documentation-sized project (tens to low hundreds of files).

If you ever need to scale to many concurrent users or thousands of
documents, that's when a real database (e.g. SQLite for metadata, or a
dedicated vector DB like ChromaDB/pgvector) starts to pay for itself.

## API reference (for reference, the frontend already calls these)

| Method | Endpoint              | Purpose                          |
|--------|------------------------|-----------------------------------|
| GET    | `/api/files`           | List uploaded documents          |
| POST   | `/api/upload`          | Upload a document (multipart)    |
| DELETE | `/api/files/{filename}`| Delete a document                |
| POST   | `/api/ask`             | `{"question": "..."}` → answer   |

## Troubleshooting

- **Frontend shows "Could not reach the backend"** → make sure
  `uvicorn` is still running on port 8000 in the other terminal.
- **CORS errors in the browser console** → confirm the frontend is
  running on port 5173 (the backend only allows that origin by default;
  see `allow_origins` in `backend/main.py` if you change ports).
- **Upload fails** → only `.txt`, `.md`, and `.pdf` files are accepted.
