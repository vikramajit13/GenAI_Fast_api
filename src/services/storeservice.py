from ..core.store import RAGStore
import numpy as np
from ..utils.utils import (
    chunk_text,
    tokenize,
    clean_query,
    to_pgvector_str,
    to_vec_literal,
)
from ..utils.embeddings import (
    return_embeddings,
    return_crossencoded,
    return_topk_sentences,
    return_top_sentences
)
from ..utils.retrieval_tfidf import (
    build_idf,
)
from ..utils.invoke_ollama import query_with_context, get_lexical_query
from ..core.db import get_pool
from asyncpg.exceptions import PostgresError
#import faiss
import asyncio
from typing import Any


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

    # def ingest_and_index(self, store_name: str, text: str) -> dict:
    #     store = self._get_store(store_name)

    #     # 1) store raw doc
    #     store.add_document(text)

    #     # 2) chunk
    #     chunks = chunk_text(text, target_chars=800, overlap_sents=1, min_chars=200)

    #     # 3) embed chunks (single call)
    #     emb = return_embeddings(chunks)  # returns np.ndarray (n, dim) normalized
    #     emb = np.ascontiguousarray(emb, dtype=np.float32)

    #     index = faiss.IndexFlatIP(emb.shape[1])
    #     index.add(emb)

    #     # 4) build idf + tokens once
    #     idf = build_idf(chunks)
    #     tokens = [set(tokenize(c)) for c in chunks]

    #     # 5) save index
    #     store.index.chunks = chunks
    #     store.index.embeddings = emb
    #     store.index.idf = idf
    #     store.index.tokens = tokens
    #     store.index.faiss = index
    #     store.index.dim = emb.shape[1]

    #     return {"store": store_name, "num_chunks": len(chunks)}

    async def answer(
        self,
        store_name: str,
        query: str,
        k: int = 3,
        use_hybrid: bool = True,
    ) -> dict:
        ranked = await self.retreive_from_db(
            store_name, query, k=k, use_hybrid=use_hybrid
        )
        # ranked = [r for r in ranked if r["rrf"] >= 0.01]
        # ranked_for_llm = ranked[:3]
        reranked = return_crossencoded(ranked, query)
        sentences = return_top_sentences(reranked, query)
        print("sentences :::", sentences)

        ans = query_with_context(query, sentences, trace=False)
        return {
            "store": store_name,
            "answer": ans,
            "retrieval": sentences,
        }

    async def index_and_store_pg_vector(self, name: str, text: str) -> dict:

        # not adding anything in store here but keeping the same signature, Store will be populated by retrieve.
        chunks = chunk_text(text, target_chars=800, overlap_sents=1, min_chars=200)
        emb = np.asarray(return_embeddings(chunks), dtype=np.float32)

        rows = [
            (name, i, chunks[i], to_pgvector_str(emb[i])) for i in range(len(chunks))
        ]

        sql = """
            INSERT INTO rag_chunks(doc_id, chunk_index, chunk_text, embedding, tsv)
            VALUES($1, $2, $3, $4::vector, to_tsvector('english', $3))
            ON CONFLICT (doc_id, chunk_index)
            DO UPDATE SET chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv
            """
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(sql, rows)
        except PostgresError as e:
            print("db error occured", e)
        except Exception as e:
            print("some error occured", e)

        return {"store": name, "num_chunks": len(chunks)}

    async def retreive_from_db(
        self, name: str, query: str, k: int = 5, use_hybrid: bool = True
    ) -> list[dict]:

        q_clean = clean_query(query)
        q_emb_lex = get_lexical_query(q_clean) or q_clean
        q_emb = return_embeddings([q_clean])[0]
        q_emb_param = to_vec_literal(q_emb)

        rows = await self.get_search_results(q_emb_param, q_emb_lex, name, k)

        if not rows:
            return []

        results = []
        return [
            {
                "idx": int(r["chunk_index"]),
                "rrf": float(r["rrf_score"]),
                "chunk_text": r["chunk_text"],
            }
            for r in rows
        ]

    async def get_search_results(
        self, q_emb: str, lex_query: str, name: str, k: int
    ) -> list[Any]:
        sql = """
       WITH
        vec AS (
        SELECT
            chunk_index,
            chunk_text,
            row_number() OVER (ORDER BY embedding <=> $1::vector) AS r_vec
        FROM rag_chunks
        WHERE doc_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        ),
        lex AS (
        SELECT
            chunk_index,
            chunk_text,
            row_number() OVER (ORDER BY ts_rank_cd(tsv, q.query) DESC) AS r_lex
        FROM rag_chunks
        CROSS JOIN (SELECT websearch_to_tsquery('english', $4) AS query) q
        WHERE doc_id = $2
            AND tsv @@ q.query
        ORDER BY ts_rank_cd(tsv, q.query) DESC
        LIMIT $3
        ),
        fused AS (
        SELECT
            COALESCE(vec.chunk_index, lex.chunk_index) AS chunk_index,
            COALESCE(vec.chunk_text, lex.chunk_text) AS chunk_text,
            vec.r_vec,
            lex.r_lex,
            (CASE WHEN vec.r_vec IS NOT NULL THEN 1.0 / ($5 + vec.r_vec) ELSE 0 END) +
            (CASE WHEN lex.r_lex IS NOT NULL THEN 1.0 / ($5 + lex.r_lex) ELSE 0 END)
            AS rrf_score
        FROM vec
        FULL OUTER JOIN lex USING (chunk_index)
        )
        SELECT chunk_index, chunk_text, r_vec, r_lex, rrf_score
        FROM fused
        ORDER BY rrf_score DESC
        LIMIT $6;
        """

        # sql_with_cte = """
        # WITH q AS (SELECT websearch_to_tsquery('english', $1) AS query)
        # SELECT chunk_index, chunk_text,
        #     ts_rank_cd(tsv, q.query) AS lex_score
        # FROM rag_chunks, q
        # WHERE doc_id = $2 AND tsv @@ q.query
        # ORDER BY lex_score DESC
        # LIMIT $3;
        # """
        print("lex_query", lex_query)
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, q_emb, name, k, lex_query, 60, 6)

            return rows
        except PostgresError as e:
            print("some error occured", e)
            raise
