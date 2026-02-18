"""Pydantic schemas for inference I/O."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Single-text inference request."""

    text: str = Field(..., min_length=1)


class ProbabilityItem(BaseModel):
    """Top-k probability record."""

    label: str
    p: float


class InferenceResponse(BaseModel):
    """Single-text inference response."""

    product: str
    proba_topk: list[ProbabilityItem]


class BatchInputRow(BaseModel):
    """Batch row input schema."""

    text: str


class BatchOutputRow(BaseModel):
    """Batch row output schema."""

    pred_product: str
    pred_proba_topk_json: str
