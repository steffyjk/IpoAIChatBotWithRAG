import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- config
MODEL_EMB = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
META_FILE = "index_meta.pkl"
K = 5

# load
emb_model = SentenceTransformer(MODEL_EMB)
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)
docs = meta["docs"]

app = FastAPI()

class QueryIn(BaseModel):
    query: str

def retrieve(query, k=K):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        hits.append({"score": float(score), "doc": docs[idx]})
    return hits

# Option A: Use Hugging Face transformers for generation (if you installed)
from transformers import pipeline
# choose a small free model that supports generation; adjust model_name to one you have
# e.g., "gpt2" is tiny and not great, but works for a demonstration.
# If you have a local instruct model, replace here.
# gen_model = pipeline("text-generation", model="gpt2", max_length=512)
# gen_model = pipeline("text-generation", model="gpt2", max_length=300)
gen_model = pipeline(
    "text-generation",
    model="gpt2",
)


def generate_answer_with_context(question, hits):
    # Create prompt combining context snippets.
    context_texts = "\n\n---\n\n".join([h["doc"]["text"] for h in hits])
    prompt = (
        f"You are an assistant specialized in Indian IPOs. Use the following context extracted from an IPO dataset:\n\n"
        f"{context_texts}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely, cite which IPO(s) you used from the context when relevant."
    )
    # out = gen_model(prompt, max_length=300, do_sample=False)[0]["generated_text"]
    out = gen_model(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]

    # Post-process: remove the prompt prefix, return only generated answer portion
    return out[len(prompt):].strip()

@app.post("/query")
def query(q: QueryIn):
    hits = retrieve(q.query)
    answer = generate_answer_with_context(q.query, hits)
    return {"query": q.query, "hits": hits, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
