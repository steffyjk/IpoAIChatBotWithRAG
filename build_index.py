import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

DOCS_CSV = "ipo_docs.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "index_meta.pkl"

def build():
    df = pd.read_csv(DOCS_CSV)
    texts = df["text"].tolist()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; use IndexFlatL2 if you normalize differently
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    # Save metadata (original texts & other fields)
    meta = {"docs": df.to_dict(orient="records")}
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(meta, f)
    print("Index built and saved.")

if __name__ == "__main__":
    build()
