from fastapi import APIRouter, Depends
from ...core.dependencies import get_rag_service
from ...schemas.query_models import AnswerRequest, AnswerResponse

router = APIRouter()

@router.post("/answer/",response_model=AnswerResponse)
async def answer(name: str, req: AnswerRequest, service = Depends(get_rag_service)):
    #out = await service.answer(name, req.query, k=req.k, use_hybrid=req.use_hybrid)
    out = await service.orchestrate_answer(name, req.query)
    return {"store": name, **out}