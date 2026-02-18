"""Export trained checkpoint to ONNX and inference bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from cfpb_complaint_triage.training.model import DistilBertClassifier
from cfpb_complaint_triage.utils.paths import processed_data_dir


def export_onnx(cfg) -> Path:
    """Export model checkpoint to ONNX and create inference bundle."""
    checkpoint_path = Path(cfg.export.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_map_path = processed_data_dir() / cfg.data.label_map_filename
    label_maps = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_labels = int(label_maps["num_labels"])

    model = DistilBertClassifier.load_from_checkpoint(
        str(checkpoint_path),
        pretrained_model_name=cfg.model.pretrained_model_name,
        num_labels=num_labels,
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        warmup_ratio=float(cfg.train.warmup_ratio),
        total_training_steps=1,
        dropout=float(cfg.model.dropout),
    )
    model.eval()
    model.to("cpu")

    max_length = int(cfg.train.max_length)
    dummy_input_ids = torch.ones((1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

    onnx_path = Path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes if bool(cfg.export.dynamic_axes) else None,
        opset_version=int(cfg.export.opset),
    )

    bundle_dir = Path(cfg.export.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_model_path = bundle_dir / "model.onnx"
    shutil.copyfile(onnx_path, bundle_model_path)
    external_data_path = Path(f"{onnx_path}.data")
    if external_data_path.exists():
        shutil.copyfile(external_data_path, bundle_dir / "model.onnx.data")
    (bundle_dir / "label_maps.json").write_text(
        json.dumps(label_maps, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "tokenizer_name.txt").write_text(
        cfg.model.pretrained_model_name,
        encoding="utf-8",
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    with (bundle_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(resolved_cfg, file_obj, sort_keys=False)

    return onnx_path
