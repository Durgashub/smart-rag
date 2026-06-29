# RAG Starter Project (OpenAI API + FAISS)

A minimal, working Retrieval-Augmented Generation chatbot. Comes with a
sample employee handbook so you can test it immediately, then swap in
your own docs.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add your OpenAI API key

Copy `.env.example` to `.env` and paste in your key:

```bash
cp .env.example .env
```

Then edit `.env`:
```
OPENAI_API_KEY=sk-...
```

## 3. Build the index

This reads everything in `docs/`, chunks it, embeds it, and saves a FAISS
index to `vector_store/`.

```bash
python ingest.py
```

## 4. Chat with your docs

```bash
python app.py
```

Try asking (using the included sample handbook):
- "How many vacation days do I get?"
- "Can I work fully remote?"
- "How much parental leave is offered?"

## Using your own documents

1. Drop `.txt`, `.md`, or `.pdf` files into `docs/` (delete or keep the sample).
2. Re-run `python ingest.py` to rebuild the index.
3. Run `python app.py` and ask questions.

## How it works

```
docs/*.txt|md|pdf
      │
      ▼
  chunk_text()          (ingest.py)
      │
      ▼
  embed_texts()  ──► OpenAI text-embedding-3-small
      │
      ▼
  FAISS index    ──► vector_store/index.faiss + metadata.json
      │
      ▼
  [at chat time]
  retrieve(question)    (query.py)
      │
      ▼
  build_prompt()        (app.py)
      │
      ▼
  gpt-4.1-mini  ──► answer + sources
```

## Tuning knobs worth trying once it works

- `CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py` — smaller chunks = more
  precise retrieval but less surrounding context.
- `top_k` in `app.py`'s `ask()` — how many chunks get sent to the LLM.
- `CHAT_MODEL` — try a stronger model if answers feel shallow.

## Natural next steps

- Swap the CLI for a small Streamlit or FastAPI UI.
- Show the actual chunk text alongside sources (not just filenames).
- Add re-ranking of retrieved chunks before sending them to the LLM.
- Track conversation history for follow-up questions.
