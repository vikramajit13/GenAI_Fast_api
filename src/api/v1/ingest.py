from fastapi import APIRouter, Depends

from ...core.dependencies import get_rag_service

from ...schemas.ingest_models import Ingest

router = APIRouter()


@router.post("/ingest/")
async def ingest_doc(name: str, doc: Ingest, service=Depends(get_rag_service)):
    out = await service.index_and_store_pg_vector(name, doc.doc)
    return out


@router.get("/ingest/search")
async def search_data(name: str, q: str, service=Depends(get_rag_service)):
    results = service.search_in_store(name, q)
    return {"store": name, "results": results}
