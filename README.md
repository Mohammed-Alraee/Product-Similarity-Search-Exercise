# Product Similarity Search

A microservice that returns similar products given a product ID, built on 30,000 Amazon fashion products using FastAPI and FAISS.

---

## How to Run

Install the dependencies, unzip the dataset, then start the server.

```bash
pip install -r requirements.txt
cd data && unzip archive.zip
uvicorn app:app --reload
```

Once running, open `http://localhost:8000/docs` in your browser to interact with the API.

---

## Files

`similarity.py` is the core engine — it loads the dataset, builds the FAISS index at startup, and exposes the `find_similar_products()` function. `app.py` wraps that function in a FastAPI endpoint. `Dockerfile` and `k8s.yaml` handle containerization and Kubernetes deployment with 2 replicas and health checks.

---

## Approach

### Data Exploration

Before writing any code, I explored the dataset to understand what fields were actually usable. The spec mentions `color` and `weight` as features, but neither exists in practice — `color` is missing entirely from the dataset, and every `weight` value is `999999999`, which is a null sentinel. The fields I ended up using were `meta_keywords`, `sales_price`, `rating`, `brand`, and `parent___child_category__all`.

### Features

I combined three types of features into a single vector per product. Text features from `meta_keywords` were converted using TF-IDF with bigrams and carry 50% of the weight since keywords best describe what a product is. Price and rating were normalized to a 0–1 range using MinMaxScaler and carry 30%. Brand and category were converted to numbers using the hashing trick and carry the remaining 20%. Each group is L2-normalized before combining so no single group dominates due to scale differences.

### Similarity Search

I used cosine similarity rather than Euclidean distance because it measures the angle between vectors rather than their magnitude, which works better for high-dimensional sparse data like TF-IDF. The search is powered by FAISS `IndexFlatIP`, which does exact nearest-neighbour search on L2-normalized vectors. The index is built once at startup and reused for every request, so there is no per-request preprocessing cost. At 30,000 products, exact search is fast enough. If the dataset grew significantly, switching to `IndexIVFFlat` would trade a small amount of accuracy for much faster approximate search.

### Error Handling

The API returns 404 if the product ID is not found, 422 if the parameters are invalid, and 500 for unexpected errors.
