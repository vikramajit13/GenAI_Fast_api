import os
import asyncpg
from typing import Optional
from .config import settings

_pool: Optional[asyncpg.Pool] = None


async def init_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL
            or "postgresql://postgres:postgres@localhost:5432/postgres",
            min_size=1,
            max_size=10,
            command_timeout=60,
        )

    return _pool


async def close_db_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db pool in not initialised")
    return _pool
