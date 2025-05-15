from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle


def split_text(text, max_tokens=500):
    sentences = text.split(". ")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len((chunk + sentence).split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk = sentence
        else:
            chunk += sentence + ". "

    if chunk:
        chunks.append(chunk.strip())

    return chunks


def rebuild_faiss_index(docs_dir, index_path, pkl_path):
    model = SentenceTransformer("BAAI/bge-small-en")  # Consistent model with rag_utils

    docs = []
    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = split_text(text)
                docs.extend(chunks)

    embeddings = model.encode(docs, show_progress_bar=True)

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings found. Check if your documents are too large or empty.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)


# Call the function
rebuild_faiss_index(
    "documents",
    "index/faiss_index.bin",
    "index/docs.pkl"
)
