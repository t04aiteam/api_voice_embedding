"""API Xac thuc giong noi (Qwen3 Voice Verification) — FastAPI entrypoint.

Serves speaker-embedding extraction and voice verification backed by
marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B. Intended for a trusted internal
network — no authentication is enforced (see README.md).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from routes.voice import router as voice_router
from schemas.voice import HealthResponse
from services import model as model_service

MODEL_ID = os.getenv("MODEL_ID", "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()  # warm up once at startup, not per-request
    yield


app = FastAPI(
    title="API Xac thuc giong noi (Qwen3 Voice Verification)",
    description=f"Speaker embedding extraction and 1:1 voice verification backed by {MODEL_ID}.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(voice_router)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=MODEL_ID, device=model_service.resolve_device())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6422)