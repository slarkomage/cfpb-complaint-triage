"""FastAPI inference service for CFPB complaint triage."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from cfpb_complaint_triage.commands import compose_config
from cfpb_complaint_triage.data.download import ensure_data
from cfpb_complaint_triage.data.schema import InferenceRequest
from cfpb_complaint_triage.production.infer_onnx import ONNXProductClassifier


@lru_cache(maxsize=1)
def _build_classifier() -> tuple[ONNXProductClassifier, int]:
    cfg = compose_config([])
    ensure_data(cfg)
    bundle_dir = Path(cfg.export.bundle_dir)
    max_length = int(cfg.train.max_length)
    top_k = int(cfg.infer.top_k)
    classifier = ONNXProductClassifier(bundle_dir=bundle_dir, max_length=max_length)
    return classifier, top_k


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(title="CFPB Complaint Triage API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Healthcheck endpoint."""
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: InferenceRequest) -> dict:
        """Predict product label and top-k probabilities."""
        try:
            classifier, top_k = _build_classifier()
            response = classifier.infer_text(text=payload.text, top_k=top_k)
            return response.model_dump()
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app
