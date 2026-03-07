# Cluster-Aware Semantic Retrieval System (CASR)

This project implements a semantic search system on the **20 Newsgroups dataset**.

It includes:
- SentenceTransformer embeddings
- FAISS vector search
- Gaussian Mixture Model fuzzy clustering
- Cluster-aware semantic cache
- FastAPI API service

---

## Setup

Clone the repository:

```bash
git clone <your-repo-link>
cd semantic-search-20ng
```

Create virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run preprocessing

```bash
python preprocess.py
```

---

## Train the pipeline

```bash
python train_pipeline.py
```

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### POST `/query`

Example request:

```json
{
 "query": "space shuttle launch"
}
```

### GET `/cache/stats`

Returns cache statistics.

### DELETE `/cache`

Clears the semantic cache.