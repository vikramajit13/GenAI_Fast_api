from fastapi import APIRouter, Depends

from ...core.dependencies import get_rag_service

from ...schemas.ingest_models import Ingest

router = APIRouter()

@router.post("/ingest/")
async def ingest_doc(name:str, doc: Ingest, service = Depends(get_rag_service)):
    return {"message": service.add_to_store(name, doc)}

@router.get("/ingest/search")
async def search_data(name: str, q: str, service = Depends(get_rag_service)):
    results = service.search_in_store(name, q)
    return {"store": name, "results": results}

