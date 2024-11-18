from contextlib import asynccontextmanager
from fastapi import (
    File,
    HTTPException,
    Query,
    UploadFile,
    Form,
    FastAPI,
    Request,
    WebSocket,
)
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import requests
import io

from .db.db_instance import init_db
from .transcription.router import transcription_router, get_pipeline
from .vad.router import get_vad_model, vad_router
from .bgm_separation.router import get_bgm_separation_inferencer, bgm_separation_router
from .common.config_loader import read_env, load_server_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization
    load_server_config()
    read_env("DB_URL")  # Place .env file into /configs/.env
    init_db()
    transcription_pipeline = get_pipeline()
    vad_inferencer = get_vad_model()
    bgm_separation_inferencer = get_bgm_separation_inferencer()
    yield
    # Right before server shutdown
    transcription_pipeline = None
    vad_inferencer = None
    bgm_separation_inferencer = None


backend_app = FastAPI(
    title="Whisper-WebUI Backend",
    description=f"""
    # Whisper-WebUI Backend
    Whisper-WebUI server with fastapi. Docs are available via SwaggerUI:    
    """,
    version="0.0.1",
    lifespan=lifespan
)
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],  # Disable DELETE
    allow_headers=["*"],
)
backend_app.include_router(transcription_router)
backend_app.include_router(vad_router)
backend_app.include_router(bgm_separation_router)


@backend_app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    """
    Redirect to the documentation.
    """
    return "/docs"
