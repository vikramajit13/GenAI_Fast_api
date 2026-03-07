from fastapi import APIRouter, Depends
from ...core.dependencies import get_rag_service
from ...schemas.query_models import AnswerRequest

router = APIRouter()

@router.post("/answer/")
async def answer(name: str, req: AnswerRequest, service = Depends(get_rag_service)):
    out = await service.answer(name, req.query, k=req.k, use_hybrid=req.use_hybrid)
    return {"store": name, **out}