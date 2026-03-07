
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from .utils import sigmoid

_model = SentenceTransformer("all-MiniLM-L6-v2")
_model_crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def return_embeddings(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Chunks text and returns a NumPy array of embeddings.
    """
    #sentences: List[str] = chunk_text(str_text)
    
    # model.encode returns a numpy.ndarray by default
    embeddings: np.ndarray = _model.encode(sentences, normalize_embeddings=True)
    
    return embeddings

def return_crossencoded(ranked: list[dict], query: str) ->list[dict]:
    if not ranked:
        return []
   
    pairs = [[query,r["chunk_text"]] for r in ranked]
    scores = _model_crossencoder.predict(pairs)
    for r, score in zip(ranked, scores):
            r["ce_score"] = float(score)
    reranked = sorted(ranked, key=lambda x: x["ce_score"], reverse=True)
    return reranked[:3]