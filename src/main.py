from fastapi import FastAPI
from src.api.v1.ingest import router

app = FastAPI()

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Hello World"}