import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, evaluate, query, upload
from app.config import settings
from app.db.session import init_db

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="RAG API",
    description="Document upload + question answering over your own documents.",
    version="0.1.0",
)

# Only the frontend origin(s) listed in .env may call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_directories_exist()
    init_db()
    logger.info("Startup complete. Environment: %s", settings.environment)


app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evaluate.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["system"])
def health_check() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn

    # loop="asyncio" instead of uvicorn's default uvloop — required for
    # ragas's nest_asyncio patch to work (see module docstring above).
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio",
    )