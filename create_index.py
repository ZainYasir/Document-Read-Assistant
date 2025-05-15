# create_index.py

from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def rebuild_faiss_index(docs_dir, index_path, pkl_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []

    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                text = f.read()
                chunks += chunk_text(text)

    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(chunks, f)

# Call the function
rebuild_faiss_index(
    "documents",
    "index/faiss_index.bin",
    "index/docs.pkl"
)
