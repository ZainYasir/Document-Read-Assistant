from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

def rebuild_faiss_index(docs_dir, index_path, pkl_path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())

    embeddings = model.encode(docs, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)

# Call the function
rebuild_faiss_index(
    "documents",
    "index/faiss_index.bin",
    "index/docs.pkl"
)
