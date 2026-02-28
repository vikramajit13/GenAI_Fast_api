from .store import stores
from ..services.storeservice import RagService

def get_rag_service() -> RagService:
    return RagService(stores=stores)