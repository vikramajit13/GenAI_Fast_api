from contextlib import asynccontextmanager
from fastapi import FastAPI
from .store import stores, RAGStore
from .db import init_db_pool, close_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Initialize singleton stores
    stores["doc"] = RAGStore("doc")
    stores["images"] = RAGStore("images")
    await init_db_pool()
    yield
    # Cleanup: Ensure resources are released
    stores.clear()
    await close_db_pool()