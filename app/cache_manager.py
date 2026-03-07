import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, query_vec, cluster_id):
        if cluster_id not in self.cache:
            self.misses += 1
            return None
        
        cluster_entries = self.cache[cluster_id]
        if not cluster_entries:
            self.misses += 1
            return None

        for entry in cluster_entries:
            sim = cosine_similarity(
                query_vec.reshape(1,-1),
                entry["query_vec"].reshape(1,-1)
            )[0][0]

            if sim >= self.threshold:
                self.hits += 1
                return entry, sim
        
        self.misses += 1
        return None

    def add(self, query_text, query_vec, result, cluster_id):
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []

        self.cache[cluster_id].append({
            "query_text": query_text,
            "query_vec": query_vec,
            "result": result
        })
