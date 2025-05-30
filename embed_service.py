from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
from FlagEmbedding import BGEM3FlagModel
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_TEXTS = 100
PRECISION = 4

# Model loading
try:
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def dense_to_text(dense_vecs, precision=PRECISION):
    """Convert dense_vecs to string representation"""
    text_representations = []
    for vec in dense_vecs:
        vec_str = ','.join([f"{val:.{precision}f}" for val in vec])
        text_representations.append(vec_str)
    return text_representations

def lexical_to_sparse_format(lexical_weights, tokenizer):
    """Convert lexical_weights to [vocab_size, [index array], [value array]] format"""
    vocab_size = model.tokenizer.vocab_size
    sparse_representations = []
    for weights_dict in lexical_weights:
        if not weights_dict:
            # Empty sparse vector
            sparse_representations.append(f"[{vocab_size}, [], []]")
            continue
            
        # Extract token IDs and weights
        token_ids = [int(tid) if isinstance(tid, str) else tid for tid in weights_dict.keys()]
        weights = list(weights_dict.values())
        
        try:
            # Sort by weight (descending order) to maintain consistency
            paired_data = list(zip(token_ids, weights))
            paired_data.sort(key=lambda x: x[1], reverse=True)
            
            sorted_token_ids, sorted_weights = zip(*paired_data) if paired_data else ([], [])
            
            # Format indices array
            indices_str = '[' + ','.join(map(str, sorted_token_ids)) + ']'
            
            # Format values array with specified precision
            values_str = '[' + ','.join([f"{weight:.{PRECISION}f}" for weight in sorted_weights]) + ']'
            
            # Create sparse vector format: [vocab_size, [indices], [values]]
            sparse_format = f"[{vocab_size}, {indices_str}, {values_str}]"
            sparse_representations.append(sparse_format)
            
        except Exception as e:
            logger.warning(f"Failed to convert lexical weights to sparse format: {e}")
            sparse_representations.append(f"[{vocab_size}, [], []]")
    
    return sparse_representations



# FastAPI app definition
app = FastAPI(
    title="BGE-M3 Embedding API",
    description="API for BGE-M3 text embeddings",
    version="1.0.0"
)

# Input data structure
class TextInput(BaseModel):
    texts: List[str]
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts cannot be empty")
        if len(v) > MAX_TEXTS:
            raise ValueError(f"Too many texts. Maximum: {MAX_TEXTS}")
        return v

def embed(texts: List[str]):
    """Embedding extraction function"""
    try:
        output = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        dense = output["dense_vecs"]
        dense_text = dense_to_text(dense)
        lexical = output["lexical_weights"]
        lexical_sparse = lexical_to_sparse_format(lexical, model.tokenizer)
        
        # Create individual results for each input text
        results = []
        for i in range(len(texts)):
            results.append({
                "dense_vecs": dense_text[i],
                "sparse_vecs": lexical_sparse[i]
            })
        
        return results
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

# Endpoint definitions
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "BAAI/bge-m3"
    }

@app.post("/embed")
def get_embeddings(input: TextInput):
    """Embedding extraction endpoint"""
    try:
        logger.info(f"Processing {len(input.texts)} texts")
        results = embed(input.texts)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7999)
