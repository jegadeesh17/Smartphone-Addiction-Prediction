"""Modular FastAPI application entrypoint for Smartphone Addiction Analytical Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.health import router as health_router
from src.api.predict import router as predict_router
from src.config import Settings, get_settings
from src.inference import get_inference_engine

logger = logging.getLogger("src.main")
ROOT_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event context manager to warm up inference engine on application startup."""
    settings = get_settings()
    logger.info(
        "Starting application (host=%s, port=%d, debug=%s, mock=%s)",
        settings.APP_HOST,
        settings.APP_PORT,
        settings.APP_DEBUG,
        settings.USE_MOCK_MODEL,
    )
    try:
        if settings.USE_MOCK_MODEL:
            logger.info("Initializing mock inference engine.")
            get_inference_engine(use_mock=True)
        else:
            logger.info("Warming up LightGBM inference engine from %s", settings.MODEL_PATH)
            get_inference_engine(
                use_mock=False,
                model_path=settings.MODEL_PATH,
                priors_path=settings.PRIORS_PATH,
            )
            logger.info("LightGBM inference engine warmed up successfully.")
    except FileNotFoundError as exc:
        logger.error("Model or priors artifact missing during startup: %s (AC-1.9)", exc)
    except Exception as exc:
        logger.error("Unexpected error during inference warmup: %s", exc)

    yield

    logger.info("Shutting down application.")


def create_app() -> FastAPI:
    """Factory function to build and configure the root FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Smartphone Addiction Analytical Platform",
        description="Modular FastAPI backend for live tabular ML inference and behavioral diagnostics.",
        version="1.0.0",
        debug=settings.APP_DEBUG,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Validation error on %s %s: %s", request.method, request.url.path, exc.errors()
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_exception_handler(
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        logger.error(
            "Missing artifact on %s %s: %s (AC-1.9)", request.method, request.url.path, exc
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service Unavailable",
                "detail": str(exc),
            },
        )

    @app.exception_handler(HTTPException)
    async def generic_http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    # Mount Routers
    app.include_router(health_router)
    app.include_router(predict_router)

    # Mount Static Files (if directory exists)
    for sdir in [
        ROOT_DIR / "src" / "static",
        ROOT_DIR / "app" / "static",
        ROOT_DIR / "static",
    ]:
        if sdir.is_dir():
            app.mount("/static", StaticFiles(directory=str(sdir)), name="static")
            break

    # Optional root page serving if HTML exists
    @app.get("/", include_in_schema=False, response_model=None)
    @app.get("/app", include_in_schema=False, response_model=None)
    async def serve_index() -> Union[FileResponse, JSONResponse]:
        for candidate in [
            ROOT_DIR / "src" / "templates" / "index.html",
            ROOT_DIR / "src" / "static" / "index.html",
            ROOT_DIR / "app" / "static" / "index.html",
        ]:
            if candidate.is_file():
                return FileResponse(candidate)
        return JSONResponse(
            content={
                "message": "Smartphone Addiction Prediction API",
                "version": "1.0.0",
                "docs": "/docs",
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run("src.main:app", host=cfg.APP_HOST, port=cfg.APP_PORT, reload=cfg.APP_DEBUG)
