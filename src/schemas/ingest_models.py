from pydantic import BaseModel

class Ingest(BaseModel):
    doc: str