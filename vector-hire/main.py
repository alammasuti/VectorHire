from app.config import validate
from app.engine import init_query_engine, get_query_engine


def main():
    validate()

    print("Connecting to SQL Server and setting up AI...")
    init_query_engine()
    print("Ready.\n")

    engine = get_query_engine()

    print("Candidate Search  (powered by LlamaIndex + OpenRouter)")
    print("Type your question in plain English. Type 'quit' to exit.\n")
    print("Examples:")
    print("  - Find me backend engineers with Python skills")
    print("  - Who has more than 3 years of experience?")
    print("  - Show candidates expecting less than 80000 salary")
    print("-" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\nSearching...\n")
        try:
            response = engine.query(question)
            print("\nAnswer:")
            print(str(response))
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
