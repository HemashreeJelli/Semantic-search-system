# Cluster-Aware Semantic Retrieval System (CASR)

This project implements a **semantic search system** built on the **20 Newsgroups dataset**. It combines vector embeddings, clustering, and a semantic cache to efficiently retrieve relevant documents. The system exposes a **FastAPI API service** and includes **Docker containerisation** for reproducible deployment.

---

# Features

* SentenceTransformer embeddings for semantic text representation
* FAISS vector index for fast similarity search
* Gaussian Mixture Model (GMM) for fuzzy clustering
* Cluster-aware semantic caching for repeated queries
* FastAPI REST API for retrieval
* Docker containerisation for reproducible environments

---

# System Architecture

The retrieval pipeline follows a multi-stage architecture:

1. **Embedding Generation**
   Queries and documents are embedded into dense vectors using a SentenceTransformer model.

2. **Vector Search**
   FAISS performs fast approximate nearest-neighbour search over document embeddings.

3. **Cluster Assignment**
   A Gaussian Mixture Model assigns the query to a dominant semantic cluster.

4. **Cluster-Aware Cache**
   The system checks a cluster-scoped semantic cache to determine if a similar query has already been processed.

5. **Result Retrieval**
   If no cache hit occurs, FAISS retrieves the closest document and the result is stored in the cache for future queries.

This architecture improves retrieval efficiency by combining **vector search, clustering, and semantic caching**.

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
│   ├── faiss.index
│   └── gmm_model.pkl
│
├── preprocess.py
├── train_pipeline.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Setup

Clone the repository:

```
git clone <your-repo-link>
cd semantic-search-20ng
```

Create a virtual environment:

```
python -m venv venv
```

Activate it:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Data Preprocessing

```
python preprocess.py
```

---

# Train the Retrieval Pipeline

```
python train_pipeline.py
```

Artifacts generated:

* `models/faiss.index`
* `models/gmm_model.pkl`

---

# Run the API

```
uvicorn app.main:app --reload
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

### POST `/query`

Example request:

```
{
 "query": "space shuttle launch"
}
```

---

### GET `/cache/stats`

Returns semantic cache statistics.

---

### DELETE `/cache`

Clears the semantic cache.

---

# Docker Usage

Build the image:

```
docker build -t semantic-search-api .
```

Run the container:

```
docker run -p 8000:8000 semantic-search-api
```

Access the API:

```
http://localhost:8000/docs
```

---

# Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* Scikit-learn
* NumPy
* Pandas
* Docker
