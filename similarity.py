#Similarity Engine: 
# In this file, I focused on buidling a similarity engine based on 3 feature: 
# 1. Text (meta_keywords) - 50% weight
# 2. Numeric (sales_price, weight, rating) - 30% weight
# 3. Categorical (brand, category) - 20% weight


from __future__ import annotations #for Python 3.10+ type hinting (e.g. list[str]) without needing a future import in Python 3.11+

import json #read dataset
import re #remove currency symbolos
from pathlib import Path #Combine directory paths
from typing import List #import list type alias for type hinting

import faiss #Facebook AI Similarity Search library for efficient similarity search on large datasets
import numpy as np #numerical computing library for Python, used here for array manipulation and mathematical operations
import pandas as pd #data manipulation library for Python, used here for loading and processing the dataset
from sklearn.feature_extraction.text import TfidfVectorizer #convert text to numbers
from sklearn.preprocessing import MinMaxScaler #scales and translates each feature to a given range, here used to normalize numeric features to [0, 1]

# Read Dataset: 


DATA_PATH = Path(__file__).parent / "data" / (
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)

WEIGHT_NULL_SENTINEL = 999_999_999  # raw value used for missing weight in the dataset

# Contribution weights for each feature group (must sum to 1.0).
# The three weights control how much each feature type contributes to similarity. 
# Text matters most (50%) because product keywords best describe what something is.
TEXT_WEIGHT = 0.50
NUMERIC_WEIGHT = 0.30
CATEGORICAL_WEIGHT = 0.20


# Data loading
# Each line is one product in JSON Format. Read every line, parse it, and put its all into
# a pandas DataFrame. 

def _load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {DATA_PATH}. "
            "Please ensure the data file exists in the data directory."
        )

    records = []
    with open(DATA_PATH, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


# Feature extraction helpers
# Gets the product category (e.g. "Men's T-Shirts") from a nested dictionary field.
def _extract_category(row) -> str:
    """Return the first parent category name, or empty string."""
    val = row.get("parent___child_category__all")
    if isinstance(val, dict) and val:
        return next(iter(val))
    return ""

# Prices come as strings like "₹350.00". This strips everything except digits and the decimal point, 
# giving us 350.0. Returns None if the value is missing.
def _clean_price(val) -> float | None:
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except ValueError:
        return None

# Converts weight to a number and returns None if it's the fake 999999999 value.

def _clean_weight(val) -> float | None:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f >= WEIGHT_NULL_SENTINEL:
        return None
    return f


# Feature matrix construction

def _build_feature_matrix(df: pd.DataFrame):
    ids = df["uniq_id"].tolist()

    # --- Text features (TF-IDF on meta_keywords) ---------------------------
    texts = df["meta_keywords"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=4096, sublinear_tf=True, ngram_range=(1, 2)) #keep only the top 4096 most useful ones. Throw away the rest
    text_mat = tfidf.fit_transform(texts).toarray().astype(np.float32)  # (N, 4096)

    # --- Numeric features ---------------------------------------------------
    prices = df["sales_price"].apply(_clean_price)
    weights = df["weight"].apply(_clean_weight)
    ratings = pd.to_numeric(df["rating"], errors="coerce") #

    numeric_df = pd.DataFrame({
        "sales_price": prices,
        "weight": weights,
        "rating": ratings,
    })
    # Drop columns that are entirely null (e.g. weight is all-sentinel in this dataset)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    # Impute remaining missing with column median
    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    scaler = MinMaxScaler()
    numeric_mat = scaler.fit_transform(numeric_df).astype(np.float32)  # (N, 3)

    # --- Categorical features (brand + category) ----------------------------
    brands = df["brand"].fillna("unknown").astype(str)
    categories = df.apply(_extract_category, axis=1).fillna("unknown")

    # Use hashing trick: fast, fixed-size, no vocabulary needed
    cat_dim = 256
    cat_mat = np.zeros((len(df), cat_dim), dtype=np.float32)
    for i, (b, c) in enumerate(zip(brands, categories)):
        b_idx = hash(b) % cat_dim
        c_idx = hash(c) % cat_dim
        cat_mat[i, b_idx] += 1.0
        cat_mat[i, c_idx] += 1.0

    # --- Combine with per-group L2 normalisation then weighted concat --------
    def _l2_normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return m / norms
# All Three Features togetherne big vector per product,
# applying the weights (50/30/20). Then normalizes the whole thing so 
# every product vector has the same length — this is required for cosine 
# similarity to work correctly.
    combined = np.hstack([
        TEXT_WEIGHT * _l2_normalize(text_mat),
        NUMERIC_WEIGHT * _l2_normalize(numeric_mat),
        CATEGORICAL_WEIGHT * _l2_normalize(cat_mat),
    ])

    # Final L2 normalisation so FAISS inner-product == cosine similarity
    combined = _l2_normalize(combined)
    return combined.astype(np.float32), ids



# Index build (runs once at import time)
#FAISS IndexFlatIP: exact nearest-neighbour search using inner product.
def _build_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index


print("Building product similarity index …")
_df = _load_dataframe()
_matrix, _ids = _build_feature_matrix(_df)
_index = _build_index(_matrix)
_id_to_pos = {uid: i for i, uid in enumerate(_ids)}
print(f"Index ready — {len(_ids)} products, {_matrix.shape[1]} dimensions.")


# Public API

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    if product_id not in _id_to_pos:
        raise KeyError(f"product_id '{product_id}' not found in dataset.")

    pos = _id_to_pos[product_id]
    query = _matrix[pos : pos + 1]  # shape (1, D)

    # Request num_similar + 1 because the query product itself is always rank-0
    k = num_similar + 1
    _, indices = _index.search(query, k)

    results = []
    for idx in indices[0]:
        if idx == -1:  # FAISS returns -1 for unfilled slots
            continue
        uid = _ids[idx]
        if uid != product_id:
            results.append(uid)
        if len(results) == num_similar:
            break

    return results
