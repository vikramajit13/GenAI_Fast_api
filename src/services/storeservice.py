from ..core.store import RAGStore
import numpy as np
from ..utils.utils import chunk_text, cosine_similarity, tokenize, clean_query
from ..utils.embeddings import return_embeddings
from ..utils.retrieval_tfidf import build_idf, keyword_score_idf_tokens
from ..utils.invoke_ollama import query_with_context

import faiss


class RagService:
    def __init__(self, stores: dict[str, RAGStore]):
        self.stores = stores

    def add_to_store(self, store_name: str, content: str):
        store = self._get_store(store_name)
        return store.add_document(content)

    def search_in_store(self, store_name: str, query_text: str):
        store = self._get_store(store_name)
        return store.query(query_text)

    def _get_store(self, name: str) -> RAGStore:
        if name not in self.stores:
            # You can decide to auto-create or raise an error
            self.stores[name] = RAGStore(name)
        return self.stores[name]

    def ingest_and_index(self, store_name: str, text: str) -> dict:
        store = self._get_store(store_name)

        # 1) store raw doc
        store.add_document(text)

        # 2) chunk
        chunks = chunk_text(text, target_chars=800, overlap_sents=1, min_chars=200)

        # 3) embed chunks (single call)
        emb = return_embeddings(chunks)  # returns np.ndarray (n, dim) normalized
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        # 4) build idf + tokens once
        idf = build_idf(chunks)
        tokens = [set(tokenize(c)) for c in chunks]

        # 5) save index
        store.index.chunks = chunks
        store.index.embeddings = emb
        store.index.idf = idf
        store.index.tokens = tokens
        store.index.faiss = index
        store.index.dim = emb.shape[1]

        return {"store": store_name, "num_chunks": len(chunks)}

    def retrieve(
        self, store_name: str, query: str, k: int = 5, use_hybrid: bool = True
    ) -> list[dict]:
        store = self._get_store(store_name)
        
        idx = store.index

        if not idx.chunks or idx.embeddings is None:
            return []

        # embed query
        q_clean = clean_query(query)
        #q_emb = return_embeddings([q_clean])[0]  # shape (dim,)
        q_emb = np.asarray(return_embeddings([query])[0]).astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_emb)

        results = []
       
        for i, ch in enumerate(idx.chunks):
            cos = float(cosine_similarity(q_emb, idx.embeddings[i]))

            kw = (
                float(keyword_score_idf_tokens(q_clean, idx.tokens[i], idx.idf))
                if use_hybrid
                else 0.0
            )
            bonus = 0.0  # keep 0 for now; add anchor bonus later

            final = (0.75 * cos + 0.20 * kw + bonus) if use_hybrid else cos

            results.append(
                {
                    "idx": i,
                    "cosine": cos,
                    "keyword": kw,
                    "bonus": bonus,
                    "final": final,
                    "preview": ch.replace("\n", " ")[:160],
                }
            )

        results.sort(key=lambda x: x["final"], reverse=True)
        return results[:k]

    def answer(
        self,
        store_name: str,
        query: str,
        k: int = 3,
        use_hybrid: bool = True,
        required_terms: list[str] | None = None,
    ) -> dict:
        ranked = self.retrieve(store_name, query, k=k, use_hybrid=use_hybrid)
        ranked = [r for r in ranked if r["final"] >= 0.30]
        ranked_for_llm = ranked[:2]  # or 3
        indices = [r["idx"] for r in ranked_for_llm]

        store = self._get_store(store_name)
        selected_chunks = [store.index.chunks[i] for i in indices]

        # evidence gate (simple)
        evidence_ok = True
        if required_terms:
            evidence_ok = all(
                any(t.lower() in c.lower() for c in selected_chunks)
                for t in required_terms
            )

        if not evidence_ok:
            return {
                "answer": "I don't know based on the provided context.",
                "used_indices": indices,
                "evidence_ok": False,
                "retrieval": ranked,
            }

        ans = query_with_context(query, store.index.chunks, indices, trace=False)
        return {
            "answer": ans,
            "used_indices": indices,
            "evidence_ok": True,
            "retrieval": ranked,
        }
