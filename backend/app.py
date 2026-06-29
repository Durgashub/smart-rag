"""
Simple command-line RAG chatbot.

Run `python ingest.py` first to build the index, then run this file:
    python app.py
"""

from dotenv import load_dotenv
from openai import OpenAI

from query import retrieve

load_dotenv()
client = OpenAI()

CHAT_MODEL = "gpt-4.1-mini"  # swap for any chat model your account has access to

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context. If the answer isn't in the context, say you don't know
rather than guessing."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"""Context:
{context}

Question:
{question}

Answer using only the context above. Cite which source(s) you used."""




def ask(question: str, top_k: int = 4):
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        print("No relevant chunks found. Did you run `python ingest.py` first?")
        return

    prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content
    sources = sorted(set(c["source"] for c in chunks))

    print("\nAnswer:")
    print(answer)
    print(f"\nSources: {', '.join(sources)}")


def main():
    print("RAG chatbot ready. Type a question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        ask(question)
        print()


if __name__ == "__main__":
    main()
