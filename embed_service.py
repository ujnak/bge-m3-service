from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from FlagEmbedding import BGEM3FlagModel
import logging
import json
import os

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_TEXTS = 100
PRECISION = 4
MAX_VOCAB_SIZE = 65535  # Oracle 23ai limit
VOCAB_DICT_PATH = "global_vocab_mapping.json"  # Persistent vocabulary file

# Model loading
try:
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    original_vocab_size = model.tokenizer.vocab_size
    logger.info(f"Original vocab size: {original_vocab_size}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Global vocabulary dictionary (persistent across requests)
global_vocab_mapping: Dict[str, int] = {}
current_vocab_size = 0

def load_global_vocabulary():
    """Load global vocabulary mapping from file"""
    global global_vocab_mapping, current_vocab_size
    
    if os.path.exists(VOCAB_DICT_PATH):
        try:
            with open(VOCAB_DICT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                global_vocab_mapping = {str(k): int(v) for k, v in data['mapping'].items()}
                current_vocab_size = data['size']
                logger.info(f"Loaded global vocabulary: {current_vocab_size} tokens")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            global_vocab_mapping = {}
            current_vocab_size = 0
    else:
        logger.info("No existing vocabulary file found, starting fresh")

def save_global_vocabulary():
    """Save global vocabulary mapping to file"""
    try:
        data = {
            'mapping': global_vocab_mapping,
            'size': current_vocab_size
        }
        with open(VOCAB_DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved global vocabulary: {current_vocab_size} tokens")
    except Exception as e:
        logger.error(f"Failed to save vocabulary: {e}")

def update_global_vocabulary(new_tokens: List[str]):
    """Update global vocabulary with new tokens"""
    global global_vocab_mapping, current_vocab_size
    
    added_count = 0
    for token in new_tokens:
        token_str = str(token)
        if token_str not in global_vocab_mapping and current_vocab_size < MAX_VOCAB_SIZE:
            global_vocab_mapping[token_str] = current_vocab_size
            current_vocab_size += 1
            added_count += 1
    
    if added_count > 0:
        save_global_vocabulary()
        logger.info(f"Added {added_count} new tokens to global vocabulary")
    
    return added_count

def get_top_tokens_from_batch(lexical_weights: List[Dict], top_k: int = 1000):
    """Get top K tokens from current batch"""
    token_scores = {}
    
    # Collect all tokens with their max scores
    for weights_dict in lexical_weights:
        for token_id, weight in weights_dict.items():
            token_str = str(token_id)
            if token_str not in token_scores or weight > token_scores[token_str]:
                token_scores[token_str] = weight
    
    # Sort by score and return top K
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [token for token, _ in sorted_tokens[:top_k]]
    
    return top_tokens

def dense_to_text(dense_vecs, precision=PRECISION):
    """Convert dense_vecs to string representation"""
    text_representations = []
    for vec in dense_vecs:
        vec_str = '[' + ','.join([f"{val:.{precision}f}" for val in vec]) + ']'
        text_representations.append(vec_str)
    return text_representations

def lexical_to_sparse_format_fixed(lexical_weights: List[Dict]):
    """Convert lexical_weights to sparse format using fixed global vocabulary"""
    sparse_representations = []
    
    for weights_dict in lexical_weights:
        if not weights_dict:
            # Empty sparse vector
            sparse_representations.append(f"[{MAX_VOCAB_SIZE}, [], []]")
            continue
        
        # Map tokens to global vocabulary positions
        mapped_tokens = []
        mapped_weights = []
        
        for token_id, weight in weights_dict.items():
            token_str = str(token_id)
            if token_str in global_vocab_mapping:
                global_pos = global_vocab_mapping[token_str]
                mapped_tokens.append(global_pos)
                mapped_weights.append(weight)
        
        if not mapped_tokens:
            sparse_representations.append(f"[{MAX_VOCAB_SIZE}, [], []]")
            continue
        
        try:
            # Sort by index position (ascending order) instead of weight
            paired_data = list(zip(mapped_tokens, mapped_weights))
            paired_data.sort(key=lambda x: x[0])  # Sort by index (x[0]) instead of weight (x[1])
            
            sorted_positions, sorted_weights = zip(*paired_data) if paired_data else ([], [])
            
            # Format indices array
            indices_str = '[' + ','.join(map(str, sorted_positions)) + ']'
            
            # Format values array with specified precision
            values_str = '[' + ','.join([f"{weight:.{PRECISION}f}" for weight in sorted_weights]) + ']'
            
            # Create sparse vector format: [vocab_size, [indices], [values]]
            sparse_format = f"[{MAX_VOCAB_SIZE}, {indices_str}, {values_str}]"
            sparse_representations.append(sparse_format)
            
        except Exception as e:
            logger.warning(f"Failed to convert lexical weights to sparse format: {e}")
            sparse_representations.append(f"[{MAX_VOCAB_SIZE}, [], []]")
    
    return sparse_representations

# Load global vocabulary on startup
load_global_vocabulary()

# FastAPI app definition
app = FastAPI(
    title="BGE-M3 Embedding API with Fixed Global Vocabulary",
    description="API for BGE-M3 text embeddings with persistent global vocabulary for Oracle 23ai",
    version="2.0.0"
)

# Input data structure
class TextInput(BaseModel):
    texts: List[str]
    update_vocabulary: bool = True  # Whether to update global vocab with new tokens
    top_k_per_batch: int = 1000     # Top K tokens to consider from current batch
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts cannot be empty")
        if len(v) > MAX_TEXTS:
            raise ValueError(f"Too many texts. Maximum: {MAX_TEXTS}")
        return v

def embed(texts: List[str], update_vocabulary: bool = True, top_k_per_batch: int = 1000):
    """Embedding extraction function with fixed global vocabulary"""
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
        
        # Update global vocabulary if requested
        if update_vocabulary:
            # Get top tokens from current batch
            top_tokens = get_top_tokens_from_batch(lexical, top_k_per_batch)
            added_count = update_global_vocabulary(top_tokens)
            logger.info(f"Vocabulary update: {added_count} new tokens added")
        
        # Convert to sparse format using fixed global vocabulary
        lexical_sparse = lexical_to_sparse_format_fixed(lexical)
        
        # Create individual results for each input text
        results = []
        for i in range(len(texts)):
            results.append({
                "dense_vecs": dense_text[i],
                "sparse_vecs": lexical_sparse[i],
                "vocab_size_used": MAX_VOCAB_SIZE,
                "global_vocab_size": current_vocab_size
            })
        
        logger.info(f"Processed {len(texts)} texts using global vocabulary ({current_vocab_size} tokens)")
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
        "model": "BAAI/bge-m3",
        "original_vocab_size": original_vocab_size,
        "global_vocab_size": current_vocab_size,
        "max_vocab_size": MAX_VOCAB_SIZE
    }

@app.get("/vocabulary/info")
def vocabulary_info():
    """Get global vocabulary information"""
    return {
        "total_tokens": current_vocab_size,
        "max_capacity": MAX_VOCAB_SIZE,
        "usage_percentage": round(current_vocab_size / MAX_VOCAB_SIZE * 100, 2),
        "vocab_file": VOCAB_DICT_PATH
    }

@app.get("/vocabulary/sample")
def vocabulary_sample(limit: int = 20):
    """Get sample tokens from global vocabulary"""
    sample_tokens = list(global_vocab_mapping.items())[:limit]
    return {
        "sample_tokens": sample_tokens,
        "total_count": len(global_vocab_mapping)
    }

@app.post("/vocabulary/reset")
def reset_vocabulary():
    """Reset global vocabulary (admin function)"""
    global global_vocab_mapping, current_vocab_size
    
    global_vocab_mapping = {}
    current_vocab_size = 0
    
    if os.path.exists(VOCAB_DICT_PATH):
        os.remove(VOCAB_DICT_PATH)
    
    return {
        "status": "success",
        "message": "Global vocabulary has been reset"
    }

@app.post("/embed")
def get_embeddings(input: TextInput):
    """Embedding extraction endpoint with fixed global vocabulary"""
    try:
        logger.info(f"Processing {len(input.texts)} texts")
        results = embed(input.texts, input.update_vocabulary, input.top_k_per_batch)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7999)
