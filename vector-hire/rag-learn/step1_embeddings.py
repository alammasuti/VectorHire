import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import OPENAI_API_KEY

EMBED_MODEL = "openai/text-embedding-3-small"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

SENTENCES = [
    "Built REST APIs in Python using Django and PostgreSQL for a banking client.",
    "Deployed microservices on Kubernetes and managed Docker containers on AWS.",
    "Led a team of 6 engineers delivering a fintech payments platform.",
    "Designed responsive UIs with React, TypeScript and Redux.",
    "Wrote ETL pipelines with Spark and Airflow for a data warehouse.",
    "Hobbies include cooking pasta, hiking and photography.",
    "Managed container orchestration and CI/CD pipelines with Jenkins.",
    "Experienced in server-side development with Node.js and MongoDB.",
]


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def keyword_search(question, k=3):
    words = {w.strip("?.,").lower() for w in question.split()}
    scored = [(sum(w in s.lower() for w in words), s) for s in SENTENCES]
    return sorted(scored, reverse=True)[:k]


def vector_search(question, vectors, k=3):
    q = embed([question])[0]
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), SENTENCES[i]) for i in top]


if __name__ == "__main__":
    vectors = embed(SENTENCES)
    print(f"Each sentence became a vector of {vectors.shape[1]} numbers.\n")
    questions = sys.argv[1:] or [
        "Who has K8s experience?",
        "Someone good at backend web development",
        "Has leadership experience",
    ]
    for q in questions:
        print("=" * 70)
        print("QUESTION:", q)
        print("\n  KEYWORD search (matching words):")
        for score, s in keyword_search(q):
            print(f"    [{score} word hits] {s}")
        print("\n  VECTOR search (matching meaning):")
        for score, s in vector_search(q, vectors):
            print(f"    [{score:.2f} similarity] {s}")
        print()
