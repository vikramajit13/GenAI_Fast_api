

# have a list of common words
# find unique words that are repeating in the sentence
# 
from collections import Counter
import re
from math import log

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "be","been","being","that","this","it","as","at","by","from","but","not","all",
    "they","their","them","we","you","your","i","our","us"
}

def tokenize(text: str) -> list[str]:
    # keep numbers and % as tokens, drop most punctuation
    tokens = re.findall(r"[a-zA-Z]+|\d+%?|\d+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def build_idf(chunks: list[str]) -> dict[str, float]:
    N = len(chunks)
    df = Counter()
    for ch in chunks:
        df.update(set(tokenize(ch)))
    # smooth IDF
    return {t: log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

def keyword_score_idf(query: str, chunk: str, idf: dict[str, float]) -> float:
    q = set(tokenize(query))
    c = set(tokenize(chunk))
    overlap = q.intersection(c)
    if not q:
        return 0.0
    # sum weights of overlapping terms
    num = sum(idf.get(t, 0.0) for t in overlap)
    den = sum(idf.get(t, 0.0) for t in q) + 1e-6
    return num / den
    