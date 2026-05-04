"""
MOFTransformer API server.

Wraps LocalMOFTransformerAPI (or StubMOFTransformerAPI) behind HTTP endpoints
so that the moftransformer process can run independently from ChatMOF.

Start:
    uvicorn chatmof.moftransformer_api.server:app --host 0.0.0.0 --port 8000

    # Use stub backend (no moftransformer required):
    CHATMOF_MOFTRANSFORMER_BACKEND=stub uvicorn chatmof.moftransformer_api.server:app ...

    # Use real moftransformer:
    CHATMOF_MOFTRANSFORMER_BACKEND=local uvicorn chatmof.moftransformer_api.server:app ...
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatmof.moftransformer_api import get_api

app = FastAPI(
    title="MOFTransformer API",
    description="HTTP interface to MOFTransformer prediction and CIF preparation.",
    version="1.0.0",
)


# ── request / response schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    cif_paths: List[str]       # absolute paths to .cif files on the server's filesystem
    model_dir: str             # absolute path to the property model directory
    verbose: bool = False


class PredictResponse(BaseModel):
    cif_id: List[str]
    regression_logits: Optional[List[float]] = None
    classification_logits_index: Optional[List[int]] = None


class PrepareDataRequest(BaseModel):
    cif_path: str              # absolute path to the generated .cif file
    save_dir: str              # absolute path to the output directory


class PrepareDataResponse(BaseModel):
    success: bool


class PropertiesResponse(BaseModel):
    properties: List[str]


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "backend": type(get_api()).__name__}


@app.get("/properties", response_model=PropertiesResponse)
def get_properties() -> PropertiesResponse:
    return PropertiesResponse(properties=get_api().get_predictable_properties())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    data_list = [Path(p) for p in req.cif_paths]
    missing = [p for p in data_list if not p.exists()]
    if missing:
        raise HTTPException(status_code=422, detail=f"CIF files not found: {missing}")

    try:
        result = get_api().predict(data_list, Path(req.model_dir), verbose=req.verbose)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        cif_id=result["cif_id"],
        regression_logits=result.get("regression_logits"),
        classification_logits_index=result.get("classification_logits_index"),
    )


@app.post("/prepare_data", response_model=PrepareDataResponse)
def prepare_data(req: PrepareDataRequest) -> PrepareDataResponse:
    try:
        ok = get_api().prepare_data(Path(req.cif_path), Path(req.save_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PrepareDataResponse(success=ok)
