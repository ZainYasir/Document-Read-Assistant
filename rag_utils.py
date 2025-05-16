
# rag_utilities.py

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Use one consistent embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_documents(doc_paths):
    chunks = []
    for file in doc_paths:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks += chunk_text(text)

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, chunks

def build_faiss_index(embeddings, chunks, index_path, text_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(text_path, 'wb') as f:
        pickle.dump(chunks, f)

    return index, chunks

def load_faiss_index(index_path, text_path):
    if not os.path.exists(index_path) or not os.path.exists(text_path):
        files = [os.path.join("documents", f) for f in os.listdir("documents") if f.endswith(".txt")]
        embeddings, chunks = embed_documents(files)
        return build_faiss_index(embeddings, chunks, index_path, text_path)

    index = faiss.read_index(index_path)
    with open(text_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def search_documents(query, index, chunks, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return "\n".join([chunks[i] for i in indices[0]])
