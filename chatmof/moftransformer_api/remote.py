"""
RemoteMOFTransformerAPI — HTTP client that calls a running server.py instance.

Usage:
    # In chatmof/config.py:
    config['moftransformer_backend'] = 'remote'
    config['moftransformer_server_url'] = 'http://localhost:8000'

    # Or via environment variables:
    CHATMOF_MOFTRANSFORMER_BACKEND=remote
    CHATMOF_MOFTRANSFORMER_SERVER_URL=http://localhost:8000
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import httpx

from chatmof.moftransformer_api.interface import MOFTransformerAPI

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 300  # seconds — prediction can be slow on large batches


class RemoteMOFTransformerAPI(MOFTransformerAPI):
    """
    Calls the MOFTransformer HTTP server instead of running inference locally.
    The server process (server.py) holds the model in memory and handles
    prediction requests independently from ChatMOF.
    """

    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = os.environ.get("CHATMOF_MOFTRANSFORMER_SERVER_URL")
        if base_url is None:
            from chatmof.config import config
            base_url = config.get("moftransformer_server_url", DEFAULT_URL)
        self.base_url = base_url.rstrip("/")

    # ── public API ─────────────────────────────────────────────────────────────

    def get_predictable_properties(self) -> List[str]:
        resp = self._get("/properties")
        return resp["properties"]

    def predict(
        self,
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
    ) -> Dict[str, List[Any]]:
        payload = {
            "cif_paths": [str(p) for p in data_list],
            "model_dir": str(model_dir),
            "verbose": verbose,
        }
        resp = self._post("/predict", payload)
        result: Dict[str, List[Any]] = {"cif_id": resp["cif_id"]}
        if resp.get("regression_logits") is not None:
            result["regression_logits"] = resp["regression_logits"]
        if resp.get("classification_logits_index") is not None:
            result["classification_logits_index"] = resp["classification_logits_index"]
        return result

    def prepare_data(
        self,
        cif_path: Path,
        save_dir: Path,
        logger=None,
        eg_logger=None,
    ) -> bool:
        payload = {"cif_path": str(cif_path), "save_dir": str(save_dir)}
        resp = self._post("/prepare_data", payload)
        return resp["success"]

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _get(self, path: str) -> Dict:
        try:
            r = httpx.get(f"{self.base_url}{path}", timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach MOFTransformer server at {self.base_url}. "
                "Start it with: uvicorn chatmof.moftransformer_api.server:app"
            )

    def _post(self, path: str, payload: Dict) -> Dict:
        try:
            r = httpx.post(f"{self.base_url}{path}", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach MOFTransformer server at {self.base_url}. "
                "Start it with: uvicorn chatmof.moftransformer_api.server:app"
            )
