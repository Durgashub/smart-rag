import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from query import retrieve
from server import build_prompt
from server import client
from server import SYSTEM_PROMPT
from server import CHAT_MODEL

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

benchmark_path = BASE_DIR / "evaluation" / "benchmark.csv"
results_dir = BASE_DIR / "evaluation" / "results"
results_dir.mkdir(exist_ok=True)

# ----------------------------
# Load benchmark
# ----------------------------
benchmark = pd.read_csv(benchmark_path)

results = []

print("\n========== STARTING EVALUATION ==========\n")

# ----------------------------
# Run benchmark questions
# ----------------------------
for _, row in benchmark.iterrows():

    question = row["question"]
    expected_answer = row["expected_answer"]

    print(f"Processing: {question}")

    # Retrieve context
    chunks = retrieve(question, top_k=8)

    contexts = [chunk["text"] for chunk in chunks]

    print(f"Retrieved {len(contexts)} chunks")

    # Build prompt
    prompt = build_prompt(question, chunks)

    # Generate answer
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    print("✓ Answer generated\n")

    results.append(
        {
            "question": question,
            "expected_answer": expected_answer,
            "answer": answer,
            "contexts": contexts,
        }
    )

print("========== EVALUATION FINISHED ==========\n")

# ----------------------------
# Save raw results
# ----------------------------
results_df = pd.DataFrame(results)

results_file = results_dir / "results.csv"

results_df.to_csv(results_file, index=False)

print(f"Results saved to:\n{results_file}")

# ----------------------------
# Prepare Dataset for Ragas
# ----------------------------
dataset = Dataset.from_dict(
    {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["expected_answer"] for r in results],
    }
    
)
print("\nRunning Ragas evaluation...\n")

ragas_result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ],
)

ragas_df = ragas_result.to_pandas()

ragas_file = results_dir / "ragas_results.csv"

ragas_df.to_csv(ragas_file, index=False)

print(f"Ragas results saved to:\n{ragas_file}")
print("\n========== RAGAS SUMMARY ==========")

print(ragas_df.mean(numeric_only=True))

print("===================================")

print("\nDataset created successfully.")
print(dataset)