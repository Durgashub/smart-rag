"""
Quick check: confirms your OpenAI API key is valid and working.
Drop this file into your rag-project folder and run it.

Usage:
    python test_key.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your-api-key-here":
    print("No API key found. Check that .env exists in this folder and contains")
    print("a real key, e.g.  OPENAI_API_KEY=sk-...")
else:
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say 'API key works' and nothing else."}],
            max_tokens=10,
        )
        print("SUCCESS:", response.choices[0].message.content)
    except Exception as e:
        print("FAILED:", e)
