"""
Top-level FastAPI entry point.

Start with:  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.service.schemas import ThemeSongRequest, ThemeSongResponse, ErrorResponse
from src.pipeline import run_pipeline

app = FastAPI(
    title="Boock Theme Song Generator",
    version="1.0.0",
    description="Generates 20-second original songs from lyrics, genre, and emotion.",
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "boock-theme-song"}


@app.post(
    "/theme-song/render",
    response_model=ThemeSongResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def render(request: ThemeSongRequest):
    try:
        return run_pipeline(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            status="failed",
            error=f"Request validation error: {exc}",
        ).model_dump(),
    )
