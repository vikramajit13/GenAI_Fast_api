from ..core.store import RAGStore
import numpy as np
from ..utils.utils import (
    chunk_text,
    clean_query,
    to_pgvector_str,
    to_vec_literal,
    normalize_text
)
from ..utils.embeddings import (
    return_embeddings,
    return_crossencoded,
    return_top_sentences,
    select_evidence,
    rewrite_query
)

from ..utils.invoke_ollama import query_with_context, get_lexical_query
from ..core.db import get_pool
from asyncpg.exceptions import PostgresError

# import faiss
import asyncio
from typing import Any


class RagService:
    

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
        if not ranked:
            return {"answer": None, "reason": "no_retrieval_results", "retrieval": []}
        # ranked = [r for r in ranked if r["rrf"] >= 0.01]
        # ranked_for_llm = ranked[:3]
        reranked = return_crossencoded(ranked, query)
        sentences = return_top_sentences(reranked, query)
        sentences = [r for r in sentences if r["score"] >= 0.25]
        if not sentences:
            return {
                "answer": None,
                "reason": "insufficient_evidence",
                "retrieval": ranked,
            }
            
        try:
            ans = await query_with_context(query, sentences, trace=False)
            return {
                "store": store_name,
                "answer": ans,
                "retrieval": sentences,
            }
        except Exception as e:
            return {
                "answer": None,
                "reason": "llm_unavailable",
                "error": str(e),
                "retrieval": ranked,
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
        
    async def orchestrate_answer(self,doc_id: str, query: str, max_retries: int = 2):

        attempt = 0
        current_query = query
        seen_queries = {normalize_text(query)} 

        while attempt <= max_retries:

            ranked = await self.retreive_from_db(doc_id, current_query)

            if not ranked:
                return {
                    "path": "refuse",
                    "reason": "no_retrieval_results",
                    "answer": "no Randed",
                    "sources": [],
                    "retrieval" :None,
                    "store": doc_id
                }

            sentences = select_evidence(ranked, current_query)

            if sentences:
                answer = await query_with_context(query, sentences)

                return {
                    "path": "direct_answer" if attempt == 0 else "retry_retrieval",
                    "answer": answer,
                    "retrieval": sentences,
                    "attempts": attempt + 1,
                     "store": doc_id
                }

            if attempt == max_retries:
                break

            # rewrite query for retry
         
            rewritten = rewrite_query(query)
            if not rewritten:
                break
            if normalize_text(rewritten) in seen_queries:
                break
            current_query = rewritten
            seen_queries.add(normalize_text(rewritten))
            attempt += 1

        return {
            "path": "refuse",
            "reason": "insufficient_evidence",
            "answer": "failed last return",
            "sources": [],
            "attempts": attempt + 1,
            "retrieval" :None,
            "store": doc_id
        }
