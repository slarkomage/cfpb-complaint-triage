"""MLflow pyfunc wrapper for ONNXRuntime product classifier."""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from cfpb_complaint_triage.production.infer_onnx import infer_dataframe


class CFPBOnnxPyfuncModel(mlflow.pyfunc.PythonModel):
    """Serve ONNX bundle through MLflow pyfunc interface."""

    def __init__(self, bundle_dir: str, max_length: int = 256, top_k: int = 3) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.max_length = max_length
        self.top_k = top_k

    def load_context(self, context) -> None:
        if context is not None and "bundle_dir" in context.artifacts:
            self.bundle_dir = Path(context.artifacts["bundle_dir"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        del context
        if "text" not in model_input.columns:
            raise ValueError("Input DataFrame must include 'text' column")
        return infer_dataframe(
            bundle_dir=self.bundle_dir,
            texts=model_input["text"],
            max_length=self.max_length,
            top_k=self.top_k,
        )
