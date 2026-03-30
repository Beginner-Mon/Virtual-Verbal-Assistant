"""Unified main API endpoint for the complete three-service pipeline.

This module is now the thin FastAPI composition layer. Business logic lives in
services/* and HTTP handlers live in routers/*.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config.settings import get_main_api_settings
from routers import answer_router, compatibility_router, health_router, sessions_router
from utils.logger import get_logger

logger = get_logger(__name__)
SETTINGS = get_main_api_settings()

app = FastAPI(
    title="Unified Multi-Service Pipeline API",
    description="Single entry point that fans out to AgenticRAG (port 8000) and DART (port 5001)",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(answer_router)
app.include_router(compatibility_router)
app.include_router(health_router)
app.include_router(sessions_router)


if __name__ == "__main__":
    logger.info(f"Starting Unified Pipeline API on {SETTINGS.main_api_host}:{SETTINGS.main_api_port}...")
    logger.info("")
    logger.info("Requires these services to be running:")
    logger.info(f"  AgenticRAG : {SETTINGS.agentic_rag_url}  (Windows, firstconda env)")
    logger.info(f"  DART       : {SETTINGS.dart_url}  (WSL/Linux, DART env)")
    logger.info(f"  SpeechLLM  : {SETTINGS.tts_url}")
    logger.info("")
    logger.info(f"Frontend: POST http://localhost:{SETTINGS.main_api_port}/answer")
    logger.info("")

    uvicorn.run(
        app,
        host=SETTINGS.main_api_host,
        port=SETTINGS.main_api_port,
        log_level="info",
    )
