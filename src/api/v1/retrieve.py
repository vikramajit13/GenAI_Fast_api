from fastapi import APIRouter, Depends
from ...core.dependencies import get_rag_service
from ...schemas.query_models import RetrieveRequest

router = APIRouter()

@router.post("/retrieve/")
async def retrieve(name: str, req: RetrieveRequest, service = Depends(get_rag_service)):
    results = await service.retreive_from_db(name, req.query, k=req.k, use_hybrid=req.use_hybrid)
    return {"store": name, "results": results}



