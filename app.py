from fastapi import FastAPI, HTTPException, Query
from typing import List

from similarity import find_similar_products

app = FastAPI(
    title="Product Similarity Search",
    description="Returns similar Amazon fashion products based on text, numeric, and categorical features.",
    version="1.0.0",
)


@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(..., description="uniq_id of the query product"),
    num_similar: int = Query(..., ge=1, le=100, description="Number of similar products to return"),
) -> List[str]:
    
   # Returns a ranked list of product IDs most similar to the given product_id.
   
    try:
        return find_similar_products(product_id, num_similar)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}
