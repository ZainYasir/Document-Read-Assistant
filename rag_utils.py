import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("BAAI/bge-small-en")


def embed_documents(doc_paths):
    texts = []
    for file in doc_paths:
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    return embeddings, texts


def build_faiss_index(embeddings, texts, index_path, text_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(text_path, 'wb') as f:
        pickle.dump(texts, f)

    return index, texts


def load_faiss_index(index_path, text_path):
    if not os.path.exists(index_path) or not os.path.exists(text_path):
        files = [os.path.join("documents", f) for f in os.listdir("documents") if f.endswith(".txt")]
        embeddings, texts = embed_documents(files)
        return build_faiss_index(embeddings, texts, index_path, text_path)

    index = faiss.read_index(index_path)
    with open(text_path, 'rb') as f:
        texts = pickle.load(f)
    return index, texts


def search_documents(query, index, texts, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return "\n".join([texts[i] for i in indices[0]])