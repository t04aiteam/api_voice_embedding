"""API Xac thuc giong noi (Qwen3 Voice Verification) — FastAPI entrypoint.

Serves speaker-embedding extraction and voice verification backed by
marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B. Intended for a trusted internal
network — no authentication is enforced (see README.md).
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# .env must be in the environment before the services read their settings at
# import time. Real environment variables still win (override=False).
load_dotenv(override=False)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from routes.voice import router as voice_router  # noqa: E402
from schemas.voice import HealthResponse  # noqa: E402
from services import model as model_service  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice_api")

MODEL_ID = os.getenv("MODEL_ID", "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B")
# Seconds a client should wait before retrying after a 503.
OOM_RETRY_AFTER_SECONDS = int(os.getenv("OOM_RETRY_AFTER_SECONDS", "10"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model up once at startup, not per request.

    Args:
        app: The FastAPI application being started.

    Yields:
        Control to the running application.
    """
    model_service.load_model()
    yield


app = FastAPI(
    title="API Xac thuc giong noi (Qwen3 Voice Verification)",
    description=f"Speaker embedding extraction and 1:1 voice verification backed by {MODEL_ID}.",
    version="1.1.0",
    lifespan=lifespan,
)
app.include_router(voice_router)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Report liveness, the configured model id and the inference device.

    Returns:
        The health payload.
    """
    return HealthResponse(
        status="ok", model=MODEL_ID, device=model_service.resolve_device()
    )


@app.exception_handler(model_service.InferenceOutOfMemory)
async def out_of_memory_handler(
    request: Request, exc: model_service.InferenceOutOfMemory
):
    """Turn an inference OOM into a retryable 503 with a Retry-After header.

    Args:
        request: The failing request.
        exc: The out-of-memory error raised by the model service.

    Returns:
        A 503 JSON response.
    """
    logger.error(
        "%s %s: out of memory on %s (%s)",
        request.method,
        request.url.path,
        exc.device,
        type(exc.cause).__name__,
    )
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(OOM_RETRY_AFTER_SECONDS)},
        content={
            "detail": (
                f"Server out of memory on {exc.device} while processing this request. "
                f"Retry in {OOM_RETRY_AFTER_SECONDS}s or send a shorter recording."
            ),
            "retry_after_seconds": OOM_RETRY_AFTER_SECONDS,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the traceback server-side and return a generic 500 to the client.

    Args:
        request: The failing request.
        exc: The unhandled exception.

    Returns:
        A 500 JSON response naming only the exception type.
    """
    logger.exception(
        "%s %s: unhandled %s", request.method, request.url.path, type(exc).__name__
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {type(exc).__name__}. See the service log."
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "6422"))
    )
