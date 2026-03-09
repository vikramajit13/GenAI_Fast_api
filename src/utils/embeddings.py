import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from .utils import split_sentences, cosine_similarity, normalize_text


_model = SentenceTransformer("all-MiniLM-L6-v2")
_model_crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def return_embeddings(
    sentences: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Chunks text and returns a NumPy array of embeddings.
    """
    # sentences: List[str] = chunk_text(str_text)

    # model.encode returns a numpy.ndarray by default
    embeddings: np.ndarray = _model.encode(sentences, normalize_embeddings=True)

    return embeddings


def return_crossencoded(ranked: list[dict], query: str) -> list[dict]:
    if not ranked:
        return []

    pairs = [[query, r["chunk_text"]] for r in ranked]
    scores = _model_crossencoder.predict(pairs)
    for r, score in zip(ranked, scores):
        r["ce_score"] = float(score)
    reranked = sorted(ranked, key=lambda x: x["ce_score"], reverse=True)
    return reranked[:3]


def return_topk_sentences(reranked: list[dict], query: str, k: int = 3):
    # 1. Flatten sentences and track their metadata
    flat_sentences = []
    metadata = []
    
    for r in reranked:
        sentences = split_sentences(r["chunk_text"])
        for i, text in enumerate(sentences):
            flat_sentences.append(text)
            metadata.append({"chunk_index": r["idx"], "sentence_index": i, "text": text})

    if not flat_sentences:
        return []

    # 2. Batch embed sentences (Massive performance boost)
    # Assuming return_embeddings can handle a list; if not, wrap in a single loop
    query_embed = return_embeddings(query)
    sentence_embeds = return_embeddings(flat_sentences) 
    
    # 3. Calculate scores and build results
    results = []
    for i, emb in enumerate(sentence_embeds):
        score = cosine_similarity(emb, query_embed)
        results.append({**metadata[i], "score": float(score)})

    # 4. Sort and return top K
    return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

def return_top_sentences(reranked: list[dict], query: str, top_n: int = 5) -> list[dict]:
    sentence_rows = []

    # flatten sentences with metadata
    for r in reranked:
        chunk_index = r["idx"]
        for sentence_index, text in enumerate(split_sentences(r["chunk_text"])):
            sentence_rows.append(
                {
                    "chunk_index": chunk_index,
                    "sentence_index": sentence_index,
                    "text": text,
                }
            )

    if not sentence_rows:
        return []

    # batch embed all sentences at once
    sentence_texts = [row["text"] for row in sentence_rows]
    sentence_embeds = return_embeddings(sentence_texts)
    query_embed = return_embeddings([query])[0]

    scored = []
    for row, emb in zip(sentence_rows, sentence_embeds):
        score = cosine_similarity(query_embed, emb)
        scored.append(
            {
                **row,
                "score": float(score),
            }
        )

    # sort by descending similarity
    scored.sort(key=lambda x: x["score"], reverse=True)

    # exact normalized dedupe
    seen = set()
    unique = []
    for row in scored:
        norm = normalize_text(row["text"])
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(row)
        if len(unique) >= top_n:
            break

    return unique