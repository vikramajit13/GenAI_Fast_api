# class RAGStore:
#     def __init__(self, name:str):
#         self.name= name
#         self.data:list[str] = []

#     def add_document(self, text: str):
#         self.data.append(text)
#         return f"Added to {self.name}"

#     def get_all(self):
#         return self.data

#     def query(self, search_term: str):
#         # Placeholder for actual vector search logic
#         return [doc for doc in self.data if search_term.lower() in doc.lower()]


# stores: dict[str, RAGStore] = {}

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class RAGIndex:
    chunks: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # shape: (n_chunks, dim)
    idf: dict[str, float] = field(default_factory=dict)
    tokens: list[set[str]] = field(default_factory=list)
    faiss: np.float32 | None = None
    dim: np.ndarray | None = None


class RAGStore:
    def __init__(self, name: str):
        self.name = name
        self.raw_docs: list[str] = []
        self.index = RAGIndex()
        self.faiss = None
        self.dim = None

    def add_document(self, text: str):
        self.raw_docs.append(text)
        return f"Added to {self.name}"

    def get_all(self):
        return self.raw_docs


stores: dict[str, RAGStore] = {}
