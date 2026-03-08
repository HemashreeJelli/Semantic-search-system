import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

from app.cache_manager import SemanticCache


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    state["model"] = None
    state["index"] = None
    state["gmm"] = None
    state["pca"] = None
    state["texts"] = None
    state["cache"] = SemanticCache(threshold=0.85)

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


def load_resources():

    if state["model"] is None:
        state["model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    if state["index"] is None:
        state["index"] = faiss.read_index(
            os.path.join(BASE_DIR, "models/faiss.index")
        )

    if state["gmm"] is None:
        state["gmm"] = joblib.load(
            os.path.join(BASE_DIR, "models/gmm_model.pkl")
        )

    if state["pca"] is None:
        state["pca"] = joblib.load(
            os.path.join(BASE_DIR, "models/pca_model.pkl")
        )

    if state["texts"] is None:
        state["texts"] = pd.read_csv(
            os.path.join(BASE_DIR, "data/cleaned_20ng.csv")
        )["text"].tolist()


# -------------------------
# QUERY ENDPOINT
# -------------------------

@app.post("/query")
def query_endpoint(request: QueryRequest):

    load_resources()

    query = request.query

    # Embed query
    query_vec = state["model"].encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    # PCA reduction for clustering
    query_vec_reduced = state["pca"].transform(query_vec)

    # Predict cluster
    cluster_probs = state["gmm"].predict_proba(query_vec_reduced)[0]
    cluster_id = int(np.argmax(cluster_probs))

    # Check semantic cache
    cached_hit = state["cache"].get(query_vec[0], cluster_id)

    if cached_hit:
        entry, similarity = cached_hit

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query_text"],
            "similarity_score": float(similarity),
            "result": entry["result"],
            "dominant_cluster": cluster_id
        }

    # -------------------------
    # FAISS SEARCH
    # -------------------------

    K = 5
    distances, indices = state["index"].search(query_vec, K)

    results = []

    for i in range(K):

        idx = indices[0][i]

        if idx == -1:
            continue

        results.append(state["texts"][idx])

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No similar documents found."
        )

    # best similarity
    similarity = float(distances[0][0])

    # store result in cache
    state["cache"].add(
        query,
        query_vec[0],
        results,
        cluster_id
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": similarity,
        "result": results,
        "dominant_cluster": cluster_id
    }


# -------------------------
# CACHE STATS ENDPOINT
# -------------------------

@app.get("/cache/stats")
def cache_stats():

    c = state["cache"]

    total_entries = sum(len(v) for v in c.cache.values())
    total_requests = c.hits + c.misses

    return {
        "total_entries": total_entries,
        "hit_count": c.hits,
        "miss_count": c.misses,
        "hit_rate": round(c.hits / total_requests, 3) if total_requests > 0 else 0
    }


# -------------------------
# CLEAR CACHE
# -------------------------

@app.delete("/cache")
def clear_cache():

    state["cache"].cache.clear()
    state["cache"].hits = 0
    state["cache"].misses = 0

    return {
        "message": "Cache flushed successfully"
    }