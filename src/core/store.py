

class RAGStore:
    def __init__(self, name:str):
        self.name= name
        self.data =list[str] = []
    
    def add_document(self, text: str):
        self.data.append(text)
        return f"Added to {self.name}"
    
    def get_all(self):
        return self.data
    
    def query(self, search_term: str):
        # Placeholder for actual vector search logic
        return [doc for doc in self.data if search_term.lower() in doc.lower()]
    
    
stores: dict[str, RAGStore] = {}