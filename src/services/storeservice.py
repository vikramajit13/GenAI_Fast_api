from ..core.store import RAGStore

class RagService:
    def __init__(self, stores: dict[str, RAGStore]):
        self.stores = stores

    def add_to_store(self, store_name: str, content: str):
        store = self._get_store(store_name)
        return store.add_document(content)

    def search_in_store(self, store_name: str, query_text: str):
        store = self._get_store(store_name)
        return store.query(query_text)

    def _get_store(self, name: str) -> RAGStore:
        if name not in self.stores:
            # You can decide to auto-create or raise an error
            self.stores[name] = RAGStore(name)
        return self.stores[name]
        

