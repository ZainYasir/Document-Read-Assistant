from rag_utils import load_faiss_index, search_documents, embed_documents
from model_utils import load_phi2, generate_response


def main():
    # Step 1: Load or build the FAISS index
    print("[INFO] Loading FAISS index...")
    index, texts = load_faiss_index("/kaggle/working/Document-Read-Assistant/index/faiss_index.bin", "/kaggle/working/Document-Read-Assistant/index")

    # Step 2: Load Phi-2 model
    print("[INFO] Loading Phi-2 model...")
    model, tokenizer = load_phi2()

    # Step 3: Interactive loop
    print("\n[READY] Ask me anything. Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        # Step 4: Get top relevant documents
        context = search_documents(question, index, texts)

        # Step 5: Generate answer
        answer = generate_response(question, context, model, tokenizer)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()