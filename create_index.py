
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
    model = SentenceTransformer("BAAI/bge-small-en")

    docs = []
    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            full_path = os.path.join(docs_dir, fname)
            print(f"Reading file: {full_path}")
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                if len(text.strip()) == 0:
                    print(f"Warning: {fname} is empty. Skipping.")
                    continue

                chunks = split_text(text)
                print(f" → {len(chunks)} chunks from {fname}")
                docs.extend(chunks)

    if not docs:
        raise ValueError("No valid text chunks found. Are your files empty or non-textual?")

    print(f"Total chunks to embed: {len(docs)}")
    embeddings = model.encode(docs, show_progress_bar=True)

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embedding failed. Possible issue with input format or model.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)

    print("✅ FAISS index created successfully.")


# Call it
rebuild_faiss_index(
    "documents",
    "index/faiss_index.bin",
    "index/docs.pkl"
)
