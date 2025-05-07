# core.py

from rag_utils import load_faiss_index, search_documents
from model_utils import load_phi2, generate_response

# Load assets once
print("[INFO] Loading FAISS index...")
index, texts = load_faiss_index("index/faiss_index.bin", "index/docs.pkl")

print("[INFO] Loading Phi-2 model...")
model, tokenizer = load_phi2()

def ask(question: str) -> str:
    context = search_documents(question, index, texts)
    return generate_response(question, context, model, tokenizer)
