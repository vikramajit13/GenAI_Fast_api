
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def return_embeddings(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Chunks text and returns a NumPy array of embeddings.
    """
    #sentences: List[str] = chunk_text(str_text)
    
    # model.encode returns a numpy.ndarray by default
    embeddings: np.ndarray = _model.encode(sentences, normalize_embeddings=True)
    
    return embeddings