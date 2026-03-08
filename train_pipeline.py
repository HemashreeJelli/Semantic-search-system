import os
import joblib
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# --------------------------------------------------
# Setup
# --------------------------------------------------

DATA_PATH = "data/cleaned_20ng.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

print("\nLoading cleaned dataset...")

texts = pd.read_csv(DATA_PATH)["text"].tolist()

print(f"Total documents loaded: {len(texts)}")


# --------------------------------------------------
# 2. Generate Sentence Embeddings
# --------------------------------------------------

print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
).astype("float32")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)


# --------------------------------------------------
# 3. Dimensionality Reduction (for GMM)
# --------------------------------------------------

print("\nReducing dimensions using PCA...")

pca = PCA(
    n_components=50,
    random_state=42
)

embeddings_reduced = pca.fit_transform(embeddings)

print("Reduced embedding shape:", embeddings_reduced.shape)


# --------------------------------------------------
# 4. Determine Optimal Number of Clusters
# --------------------------------------------------

print("\nCalculating BIC scores for cluster range...")

cluster_range = list(range(5, 45, 5))
bic_scores = []

for k in cluster_range:

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="tied",
        random_state=42
    )

    gmm.fit(embeddings_reduced)

    bic = gmm.bic(embeddings_reduced)
    bic_scores.append(bic)

    print(f"Clusters: {k} | BIC: {bic:.2f}")


# --------------------------------------------------
# Distance-from-Chord Method
# --------------------------------------------------

def find_sweet_spot(k_values, scores):
    """
    Finds the elbow point using the Distance-from-Chord method.
    """

    k_arr = np.array(k_values)
    s_arr = np.array(scores)

    # Normalize values
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min())

    p1 = np.array([k_norm[0], s_norm[0]])
    p2 = np.array([k_norm[-1], s_norm[-1]])

    distances = []

    for i in range(len(k_norm)):
        p3 = np.array([k_norm[i], s_norm[i]])

        d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        distances.append(d)

    return k_values[np.argmax(distances)]


best_k = find_sweet_spot(cluster_range, bic_scores)

print("\n--- Mathematical Sweet Spot Result ---")
print(f"Optimal Clusters: {best_k}")
print(
    "Justification: Maximum deviation from the BIC trend line "
    "indicating the elbow point."
)


# --------------------------------------------------
# 5. Train Final GMM Model
# --------------------------------------------------

print(f"\nTraining final GMM model with {best_k} clusters...")

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="tied",
    random_state=42
)

gmm.fit(embeddings_reduced)


# --------------------------------------------------
# 6. Cluster Interpretation
# --------------------------------------------------

cluster_probs = gmm.predict_proba(embeddings_reduced)

print("\nCluster Analysis (Top 3 Docs per Cluster):")

for cluster_id in range(best_k):

    cluster_strength = cluster_probs[:, cluster_id]

    top_docs_idx = np.argsort(cluster_strength)[-3:][::-1]

    print(f"\nCluster {cluster_id}")
    print("-" * 40)

    for idx in top_docs_idx:
        print(f"[{idx}] {texts[idx][:150]}...")


# --------------------------------------------------
# 7. Build FAISS Index
# --------------------------------------------------

print("\nBuilding FAISS index for similarity search...")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("FAISS index size:", index.ntotal)


# --------------------------------------------------
# 8. Save Model Artifacts
# --------------------------------------------------

print("\nSaving trained models...")

joblib.dump(gmm, os.path.join(MODEL_DIR, "gmm_model.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.pkl"))

faiss.write_index(index, os.path.join(MODEL_DIR, "faiss.index"))

np.save(os.path.join(MODEL_DIR, "embeddings.npy"), embeddings)


print("\nPipeline completed successfully.")
print(f"Final number of clusters used: {best_k}")