import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3')

# 語彙サイズを取得
vocab_size = model.tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# 1. dense_vecsをテキスト表現に変換
def dense_to_text(dense_vecs, precision=6):
    """dense_vecsを文字列に変換"""
    text_representations = []
    for vec in dense_vecs:
        # 各要素を指定精度で文字列化してカンマ区切りに
        vec_str = ','.join([f"{val:.{precision}f}" for val in vec])
        text_representations.append(vec_str)
    return text_representations

# 2. lexical_weightsをテキスト表現に変換
def lexical_to_text(lexical_weights, tokenizer):
    """lexical_weightsを人間が読める形式に変換"""
    text_representations = []
    for weights_dict in lexical_weights:
        token_weight_pairs = []
        for token_id, weight in weights_dict.items():
            if isinstance(token_id, str):
                token_id = int(token_id)
            # トークンIDを実際のトークンに変換
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            token_weight_pairs.append(f"{token}:{weight:.6f}")
        
        # 重み順でソート（降順）
        token_weight_pairs.sort(key=lambda x: float(x.split(':')[1]), reverse=True)
        text_representations.append('|'.join(token_weight_pairs))
    return text_representations

output = model.encode(["test"], return_dense=True, return_sparse=True)

print(dense_to_text(output["dense_vecs"]))
print(lexical_to_text(output["lexical_weights"], model.tokenizer))
