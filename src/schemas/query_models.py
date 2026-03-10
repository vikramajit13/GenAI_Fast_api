from pydantic import BaseModel, Field
from typing import Optional, List

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    use_hybrid: bool = True

class AnswerRequest(BaseModel):
    query: str =  Field(..., min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)
    use_hybrid: bool = True
    
class RetrievalObject(BaseModel):
    chunk_index: int
    sentence_index: int
    text: str
    score: float
    
class AnswerResponse(BaseModel):
    store: str
    answer: str
    retrieval: list[RetrievalObject]  # Example fields—match your 'out' dict structure