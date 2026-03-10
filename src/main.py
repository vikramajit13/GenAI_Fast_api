from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from .core.lifecycle import lifespan
from .core.exceptionhandler import validation_exception_handler
from .api.v1.ingest import router as ingest_router
from .api.v1.retrieve import router as retrieve_router
from .api.v1.answer import router as answer_router

app = FastAPI(lifespan=lifespan)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(ingest_router, prefix="/api/v1")
app.include_router(retrieve_router, prefix="/api/v1")
app.include_router(answer_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Hello World"}
