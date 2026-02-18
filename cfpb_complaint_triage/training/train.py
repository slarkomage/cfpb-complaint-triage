"""Training pipeline entrypoint for Lightning model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightning as L
import mlflow
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from cfpb_complaint_triage.data.download import ensure_data
from cfpb_complaint_triage.data.preprocess import preprocess_data
from cfpb_complaint_triage.training.datamodule import CFPBDataModule
from cfpb_complaint_triage.training.metrics import confusion_matrix_array
from cfpb_complaint_triage.training.model import DistilBertClassifier
from cfpb_complaint_triage.utils.git import get_commit_id
from cfpb_complaint_triage.utils.logging import (
    init_mlflow,
    log_resolved_config,
    save_confusion_matrix,
    save_curve_plot,
)
from cfpb_complaint_triage.utils.paths import (
    artifacts_dir,
    plots_dir,
    processed_data_dir,
    project_root,
)
from cfpb_complaint_triage.utils.seeding import seed_everything


def _flatten_dict(values: dict, prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, full_key))
        else:
            flattened[full_key] = str(value)
    return flattened


def _collect_predictions(model: DistilBertClassifier, dataloader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    device = model.device
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(pred_labels.tolist())
            targets.extend(labels.tolist())
    return np.asarray(targets), np.asarray(predictions)


def train_pipeline(cfg):
    """Train DistilBERT model and log MLflow artifacts."""
    seed_everything(int(cfg.preprocess.seed))
    ensure_data(cfg)
    preprocess_data(cfg)

    label_map_path = processed_data_dir() / cfg.data.label_map_filename
    label_maps = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_labels = int(label_maps["num_labels"])

    data_module = CFPBDataModule(cfg)
    data_module.setup("fit")
    train_steps = int(np.ceil(len(data_module.train_dataset) / int(cfg.train.batch_size)))
    total_training_steps = max(1, train_steps * int(cfg.train.epochs))

    model = DistilBertClassifier(
        pretrained_model_name=cfg.model.pretrained_model_name,
        num_labels=num_labels,
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        warmup_ratio=float(cfg.train.warmup_ratio),
        total_training_steps=total_training_steps,
        dropout=float(cfg.model.dropout),
    )

    checkpoint_dir = artifacts_dir() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/f1_macro",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val/f1_macro",
        mode="max",
        patience=int(cfg.train.early_stopping_patience),
    )

    trainer = L.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        deterministic=True,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=1,
        logger=False,
    )

    if cfg.logging.enable:
        init_mlflow(cfg)
        with mlflow.start_run(run_name=cfg.logging.run_name):
            resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(_flatten_dict(resolved_cfg_dict))
            mlflow.set_tag("git_commit_id", get_commit_id(project_root()))
            log_resolved_config(cfg)
            trainer.fit(model, datamodule=data_module)
            trainer.test(
                model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
            )
            _save_training_artifacts(cfg, model, data_module, label_maps)
            _copy_best_checkpoint(checkpoint_callback.best_model_path)
            mlflow.log_artifact(str(artifacts_dir() / "checkpoints" / "best.ckpt"), "checkpoints")
    else:
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
        _save_training_artifacts(cfg, model, data_module, label_maps)
        _copy_best_checkpoint(checkpoint_callback.best_model_path)


def _save_training_artifacts(cfg, model, data_module, label_maps: dict) -> None:
    del cfg
    plot_directory = plots_dir()
    plot_directory.mkdir(parents=True, exist_ok=True)

    epoch_count = min(len(model.history_train_loss), len(model.history_val_loss))
    epoch_indices = list(range(1, epoch_count + 1))
    if epoch_count > 0:
        loss_path = plot_directory / "loss_curve.png"
        f1_path = plot_directory / "f1_curve.png"
        save_curve_plot(
            output_path=loss_path,
            x_values=epoch_indices,
            train_values=model.history_train_loss[:epoch_count],
            val_values=model.history_val_loss[:epoch_count],
            title="Loss Curve",
            y_label="Loss",
        )
        save_curve_plot(
            output_path=f1_path,
            x_values=epoch_indices,
            train_values=model.history_train_f1[:epoch_count],
            val_values=model.history_val_f1[:epoch_count],
            title="Macro F1 Curve",
            y_label="Macro F1",
        )
        if mlflow.active_run():
            mlflow.log_artifact(str(loss_path), "plots")
            mlflow.log_artifact(str(f1_path), "plots")

    val_targets, val_preds = _collect_predictions(model, data_module.val_dataloader())
    conf_matrix = confusion_matrix_array(val_targets, val_preds)
    labels = [
        label_maps["id_to_label"][str(index)] for index in range(len(label_maps["id_to_label"]))
    ]
    conf_path = plot_directory / "confusion_matrix_product.png"
    save_confusion_matrix(conf_matrix, labels, conf_path, title="Product Confusion Matrix")
    if mlflow.active_run():
        mlflow.log_artifact(str(conf_path), "plots")


def _copy_best_checkpoint(best_checkpoint: str) -> None:
    destination = artifacts_dir() / "checkpoints" / "best.ckpt"
    if best_checkpoint:
        source_path = Path(best_checkpoint).resolve()
        destination_path = destination.resolve()
        if source_path == destination_path:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
