from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from contextlib import asynccontextmanager

from app.cache_manager import SemanticCache

# Use a state dictionary or a class to avoid 'global' pollution
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    model_path = "models/gmm_model.pkl"
    index_path = "models/faiss.index"
    data_path = "data/cleaned_20ng.csv"

    if not all(os.path.exists(p) for p in [model_path, index_path, data_path]):
        raise RuntimeError("Model or Data files missing. Run training pipeline first.")

    state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    state["index"] = faiss.read_index(index_path)
    state["gmm"] = joblib.load(model_path)
    state["texts"] = pd.read_csv(data_path)["text"].tolist()
    state["cache"] = SemanticCache(threshold=0.85) # Tunable threshold
    yield
    # --- Shutdown Logic (if needed) ---
    state.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    query = request.query
    if "model" not in state:
        state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 1. Vectorize and Normalize
    # Normalizing makes the Dot Product equivalent to Cosine Similarity
    query_vec = state["model"].encode([query]).astype("float32")
    faiss.normalize_L2(query_vec) 

    # 2. Cluster Inference
    # We use GMM to find the most likely cluster for this vector
    cluster_probs = state["gmm"].predict_proba(query_vec)[0]
    cluster_id = int(np.argmax(cluster_probs))

    # 3. Cache Check (Cluster-Aware)
    cached_hit = state["cache"].get(query_vec[0], cluster_id)

    if cached_hit:
        entry, similarity = cached_hit
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query_text"],
            "similarity_score": round(float(similarity), 4),
            "result": entry["result"],
            "dominant_cluster": cluster_id
        }

    # 4. Vector Search (Cache Miss)
    # Search top 5 but we return the top 1 for the cache
    distances, indices = state["index"].search(query_vec, 5)
    
    # Check if indices are valid
    if indices[0][0] == -1:
        raise HTTPException(status_code=404, detail="No similar documents found.")
        
    top_result = state["texts"][indices[0][0]]

    # 5. Update Cache
    state["cache"].add(query, query_vec[0], top_result, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": top_result,
        "dominant_cluster": cluster_id
    }

@app.get("/cache/stats")
def cache_stats():
    c = state["cache"]
    total_entries = sum(len(v) for v in c.cache.values())
    total_reqs = c.hits + c.misses
    return {
        "total_entries": total_entries,
        "hit_count": c.hits,
        "miss_count": c.misses,
        "hit_rate": round(c.hits / total_reqs, 3) if total_reqs > 0 else 0
    }

@app.delete("/cache")
def clear_cache():
    state["cache"].cache.clear()
    state["cache"].hits = 0
    state["cache"].misses = 0
    return {"message": "Cache flushed successfully"}