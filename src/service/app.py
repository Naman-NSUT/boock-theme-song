from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.service.schemas import ThemeSongRequest, ThemeSongResponse, ErrorResponse
from src.pipeline import run_pipeline

app = FastAPI(title="Boock Theme Song Generator", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "boock-theme-song"}


@app.post("/theme-song/render", response_model=ThemeSongResponse)
def render(request: ThemeSongRequest):
    try:
        response = run_pipeline(request)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
