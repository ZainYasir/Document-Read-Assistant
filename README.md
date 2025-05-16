# 📄 Document Read Assistant

The **Document Read Assistant** is a local semantic search engine that reads and indexes plain-text documents using embeddings. It allows fast retrieval of relevant text chunks using FAISS (Facebook AI Similarity Search).

---

## 🚀 Features

- 🔍 Splits large documents into semantic chunks
- 🧠 Embeds text using `BAAI/bge-small-en` from SentenceTransformers
- ⚡ Fast and scalable vector search with FAISS
- 🧾 Stores chunks alongside the FAISS index for efficient retrieval

---

## 📁 Project Structure

```
.
├── documents/             # Place your .txt documents here
├── index/
│   ├── faiss_index.bin    # FAISS vector index
│   └── docs.pkl           # Serialized list of text chunks
├── build_index.py         # Script to build or rebuild the FAISS index
└── README.md              # This file
```

---

## 📦 Requirements

- Python 3.8+
- `sentence-transformers`
- `faiss-cpu`
- `pickle` (standard)
- `os` (standard)

Install the required packages:

```bash
pip install sentence-transformers faiss-cpu
```

---

## 🧠 How It Works

1. Loads `.txt` documents from the `documents/` folder.
2. Splits text into chunks (default: 500 tokens max per chunk).
3. Generates dense embeddings using `BAAI/bge-small-en`.
4. Builds a FAISS index (`IndexFlatL2`) for similarity search.
5. Stores both index and chunk metadata.

---

## 🛠️ Usage

To build the FAISS index:

```bash
python build_index.py
```

Make sure all your text files are in the `documents/` folder and have the `.txt` extension.

---

## ✅ Status

- ✅ Document chunking implemented
- ✅ FAISS indexing complete
- ✅ Basic error handling for empty files
- ⏳ Query interface and RAG integration (coming next)

---

## 🧪 Next Steps

- Add a query module for searching similar text
- Integrate with a local LLM (e.g., TinyLlama) for RAG-based answering
- Web or CLI interface for document Q&A

---

## 🧑‍💻 Author

**Zain Yasir** — AI/ML Engineer  
GitHub: [ZainYasir](https://github.com/ZainYasir)