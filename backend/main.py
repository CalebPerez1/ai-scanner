"""FastAPI server for AIScan.

Exposes two endpoints:
  POST /api/scan   — run all scanners against a project and return findings.
  GET  /api/health — liveness check.

Run with:
    uvicorn backend.main:app --reload
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.models import ScanResult
from backend.core.risk_engine import _is_git_url, scan_project


app = FastAPI(
    title="AIScan",
    description="AI supply chain security scanner API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    """Request body for POST /api/scan.

    Attributes:
        project_path: Local directory path or GitHub repository URL to scan.
        llm_endpoint_url: Optional LLM endpoint to probe for prompt injection.
        project_name: Optional display name override for the report.
    """

    project_path: str
    llm_endpoint_url: Optional[str] = None
    project_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    """Liveness check.

    Returns:
        A JSON object confirming the service is running.
    """
    return {"status": "healthy"}


@app.post("/api/scan", response_model=None)
async def scan(request: ScanRequest) -> dict:
    """Run all scanner modules against a project and return findings.

    Validates that the supplied path exists locally or looks like a GitHub
    URL before starting the scan. Returns the full ScanResult serialized to
    JSON on success.

    Args:
        request: Scan parameters — project path, optional LLM endpoint, and
            optional project name override.

    Returns:
        A JSON-serialized ScanResult.

    Raises:
        HTTPException 400: If the project_path is not a valid local directory
            and does not look like a GitHub URL.
        HTTPException 500: If the scan raises an unexpected exception.
    """
    # Validate the target before touching the network or filesystem at scale.
    if not _is_git_url(request.project_path):
        local = Path(request.project_path)
        if not local.exists():
            raise HTTPException(
                status_code=400,
                detail=f"project_path '{request.project_path}' does not exist and is not a valid repository URL.",
            )
        if not local.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"project_path '{request.project_path}' is not a directory.",
            )

    try:
        result: ScanResult = await scan_project(
            request.project_path,
            project_name=request.project_name,
            llm_endpoint_url=request.llm_endpoint_url,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scan failed: {exc}",
        ) from exc

    # Serialize via Pydantic so enum values become strings and datetime is ISO 8601.
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Static frontend — mounted last so /api/* routes take priority
# ---------------------------------------------------------------------------

_FRONTEND_BUILD = Path(__file__).parent.parent / "frontend" / "build"
if _FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_BUILD), html=True), name="frontend")
