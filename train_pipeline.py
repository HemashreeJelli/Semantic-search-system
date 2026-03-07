import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
import faiss
import numpy as np
import joblib

os.makedirs("models", exist_ok=True)

# 1. Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = pd.read_csv("data/cleaned_20ng.csv")["text"].tolist()

embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# 2. Fuzzy Clustering (GMM)
gmm = GaussianMixture(n_components=15, covariance_type="full", random_state=42)
gmm.fit(embeddings)

# 3. Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save artifacts
joblib.dump(gmm, "models/gmm_model.pkl")
faiss.write_index(index, "models/faiss.index")
np.save("models/embeddings.npy", embeddings)

print("Training pipeline completed.")