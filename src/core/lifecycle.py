from contextlib import asynccontextmanager
from fastapi import FastAPI
from .store import stores, RAGStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Initialize singleton stores
    stores["doc"] = RAGStore("doc")
    stores["images"] = RAGStore("images")
    yield
    # Cleanup: Ensure resources are released
    stores.clear()