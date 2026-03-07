from pydantic import BaseModel
from typing import Optional, List

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    use_hybrid: bool = True

class AnswerRequest(BaseModel):
    query: str
    k: int = 6
    use_hybrid: bool = True