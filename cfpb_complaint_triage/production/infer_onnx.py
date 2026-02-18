"""ONNXRuntime inference without torch runtime dependency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from cfpb_complaint_triage.data.io import read_table, write_table
from cfpb_complaint_triage.data.schema import InferenceResponse, ProbabilityItem


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class ONNXProductClassifier:
    """ONNXRuntime wrapper with tokenizer and label map."""

    def __init__(self, bundle_dir: Path, max_length: int) -> None:
        self.bundle_dir = bundle_dir
        self.max_length = max_length
        model_path = bundle_dir / "model.onnx"
        label_map_path = bundle_dir / "label_maps.json"
        tokenizer_name_path = bundle_dir / "tokenizer_name.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label maps not found: {label_map_path}")
        if not tokenizer_name_path.exists():
            raise FileNotFoundError(f"Tokenizer name not found: {tokenizer_name_path}")

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.label_maps = json.loads(label_map_path.read_text(encoding="utf-8"))
        self.id_to_label = {
            int(key): value for key, value in self.label_maps["id_to_label"].items()
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_path.read_text(encoding="utf-8").strip()
        )

    def infer_text(self, text: str, top_k: int) -> InferenceResponse:
        """Run single-text inference and return typed response."""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        probabilities = softmax(logits)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_items = [
            ProbabilityItem(label=self.id_to_label[int(index)], p=float(probabilities[index]))
            for index in top_indices
        ]
        return InferenceResponse(product=top_items[0].label, proba_topk=top_items)


def infer_text(cfg, text: str) -> dict:
    """Convenience function for single text inference."""
    classifier = ONNXProductClassifier(Path(cfg.export.bundle_dir), int(cfg.train.max_length))
    response = classifier.infer_text(text=text, top_k=int(cfg.infer.top_k))
    return response.model_dump()


def infer_batch(cfg, input_path: str, output_path: str, text_column: str | None = None) -> str:
    """Batch inference for CSV/Parquet input."""
    classifier = ONNXProductClassifier(Path(cfg.export.bundle_dir), int(cfg.train.max_length))
    input_file = Path(input_path)
    output_file = Path(output_path)
    dataframe = read_table(input_file)
    input_column = text_column or cfg.infer.input_text_column
    if input_column not in dataframe.columns:
        raise ValueError(f"Input text column not found: {input_column}")

    prediction_labels = []
    prediction_probas = []
    for text in dataframe[input_column].astype(str).tolist():
        response = classifier.infer_text(text=text, top_k=int(cfg.infer.top_k))
        prediction_labels.append(response.product)
        prediction_probas.append(json.dumps([item.model_dump() for item in response.proba_topk]))

    dataframe["pred_product"] = prediction_labels
    dataframe["pred_proba_topk_json"] = prediction_probas
    write_table(dataframe, output_file)
    return str(output_file)


def infer_dataframe(
    bundle_dir: Path, texts: pd.Series, max_length: int, top_k: int
) -> pd.DataFrame:
    """Inference helper for MLflow pyfunc."""
    classifier = ONNXProductClassifier(bundle_dir=bundle_dir, max_length=max_length)
    products = []
    topk_json = []
    for text in texts.astype(str).tolist():
        response = classifier.infer_text(text=text, top_k=top_k)
        products.append(response.product)
        topk_json.append(json.dumps([item.model_dump() for item in response.proba_topk]))
    return pd.DataFrame({"product": products, "proba_topk_json": topk_json})
