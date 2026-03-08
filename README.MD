# Semantic Search System

This project implements a **semantic search system** built on the **20 Newsgroups dataset**.
It combines **vector embeddings, clustering, and semantic caching** to efficiently retrieve relevant documents from a text corpus.

The system exposes a **FastAPI REST API** and supports **Docker containerisation** for reproducible deployment.

---

# Features

* SentenceTransformer embeddings for semantic representation
* PCA dimensionality reduction for efficient clustering
* FAISS vector index for fast similarity search
* Gaussian Mixture Model (GMM) for fuzzy semantic clustering
* Cluster-aware semantic caching
* Top-K semantic document retrieval
* FastAPI REST API
* Docker containerisation for deployment
* Docker Compose support for simplified container orchestration

---

# System Architecture

The system follows a **multi-stage semantic retrieval architecture**.

### 1. Embedding Generation

Queries and documents are embedded using a SentenceTransformer model.

```
Text → SentenceTransformer → Dense Embedding
```

### 2. Dimensionality Reduction

PCA reduces the embedding space before clustering.

```
Embedding (384D) → PCA → Reduced Embedding (50D)
```

### 3. Semantic Clustering

A Gaussian Mixture Model assigns queries to semantic clusters.

### 4. Semantic Cache

The system checks a **cluster-scoped semantic cache** for previously processed queries.

### 5. Vector Search

If no cache hit occurs, FAISS retrieves the **Top-K most similar documents**.

### 6. Cache Storage

The result is stored for faster future retrieval.

---

# Retrieval Pipeline

```
Query
 ↓
SentenceTransformer
 ↓
Embedding (384D)
 ↓
Normalization
 ↓
PCA Reduction (50D)
 ↓
GMM Cluster Prediction
 ↓
Cluster-Aware Semantic Cache
     ↓            ↓
    Hit          Miss
     ↓            ↓
 Return       FAISS Search
 Result           ↓
             Top-K Retrieval
                 ↓
            Store in Cache
```

---

# Project Structure

```
semantic-search-20ng
│
├── app/
│   ├── main.py
│   └── cache_manager.py
│
├── data/
│   └── cleaned_20ng.csv
│
├── models/
│   ├── embeddings.npy
│   ├── faiss.index
│   ├── gmm_model.pkl
│   └── pca_model.pkl
│
├── preprocess.py
├── train_pipeline.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Setup

Clone the repository

```
git clone <your-repo-link>
cd semantic-search-20ng
```

Create a virtual environment

```
python -m venv venv
```

Activate the environment

### Windows

```
venv\Scripts\activate
```

### Mac / Linux

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Data Preprocessing

Prepare the dataset.

```
python preprocess.py
```

This generates the cleaned dataset:

```
data/cleaned_20ng.csv
```

---

# Train the Retrieval Pipeline

Run the training pipeline.

```
python train_pipeline.py
```

This performs:

* embedding generation
* PCA dimensionality reduction
* cluster discovery
* GMM training
* FAISS index creation

Artifacts generated in `models/`:

```
embeddings.npy   → document embeddings
faiss.index      → FAISS vector index
gmm_model.pkl    → trained GMM model
pca_model.pkl    → PCA dimensionality reduction model
```

---

# Run the API (Local Development)

Start the FastAPI server.

```
uvicorn app.main:app --reload
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## POST `/query`

Performs semantic search on the dataset and returns the most relevant documents along with clustering information and cache status.

---

## GET `/cache/stats`

Returns statistics about the semantic cache including total entries, hit count, miss count, and hit rate.

---

## DELETE `/cache`

Clears all entries from the semantic cache and resets statistics.

---

# Docker Usage

Build the Docker image

```
docker build -t semantic-search-api .
```

Run the container

```
docker run -p 8000:8000 semantic-search-api
```

Open API documentation

```
http://localhost:8000/docs
```

---

# Docker Compose Usage

You can also run the application using Docker Compose.

Build and start the container:

```
docker compose up --build
```

Run in the background:

```
docker compose up -d
```

Stop the container:

```
docker compose down
```

Open the API documentation

```
http://localhost:8000/docs
```



