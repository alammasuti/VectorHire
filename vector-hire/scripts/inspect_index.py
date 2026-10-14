import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.resume_index_manual import get_collection


def main():
    collection = get_collection()
    print(f"Total chunks stored: {collection.count()}")

    data = collection.get(limit=5, include=["embeddings", "documents", "metadatas"])
    for i, doc_id in enumerate(data["ids"]):
        embedding = data["embeddings"][i]
        print()
        print("ID:", doc_id)
        print("Metadata:", data["metadatas"][i])
        print("Document (first 150 chars):", data["documents"][i][:150])
        print(f"Embedding (first 5 of {len(embedding)} numbers):", list(embedding[:5]))


if __name__ == "__main__":
    main()
