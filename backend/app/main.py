import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.db.engine import init_global_db
from app.api.projects_router import router as projects_router
from app.api.documents_router import router as documents_router
from app.api.analyze_router import router as analyze_router
from app.api.structure_router import router as structure_router
from app.api.translation_router import router as translation_router
from app.api.glossary_router import router as glossary_router
from app.api.qa_router import router as qa_router
from app.api.layout_router import router as layout_router
from app.api.export_router import router as export_router
from app.api.system_router import router as system_router
from app.api.events_router import router as events_router
from app.api.entities_router import router as entities_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize global DB
    init_global_db()
    print(f"[{settings.APP_NAME}] Backend initialized successfully.")
    yield
    # Shutdown
    print(f"[{settings.APP_NAME}] Backend shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(projects_router)
app.include_router(documents_router)
app.include_router(analyze_router)
app.include_router(structure_router)
app.include_router(translation_router)
app.include_router(glossary_router)
app.include_router(qa_router)
app.include_router(layout_router)
app.include_router(export_router)
app.include_router(system_router)
app.include_router(events_router)
app.include_router(entities_router)

# Mount project assets and static files
if settings.STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(settings.STATIC_DIR / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = settings.STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(settings.STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
