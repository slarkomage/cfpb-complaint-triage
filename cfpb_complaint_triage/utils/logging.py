"""Logging and MLflow helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf


def init_mlflow(cfg: Any) -> None:
    """Configure and validate MLflow tracking URI."""
    if not cfg.logging.enable:
        return
    tracking_uri = cfg.logging.mlflow_uri
    mlflow.set_tracking_uri(tracking_uri)
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
    except Exception as error:
        if cfg.logging.allow_local_fallback:
            local_uri = f"file://{(Path.cwd() / 'mlruns').resolve()}"
            mlflow.set_tracking_uri(local_uri)
            mlflow.set_experiment(cfg.logging.experiment_name)
            return
        raise RuntimeError(
            f"MLflow server is unavailable at {tracking_uri}. "
            "Set logging.allow_local_fallback=true to use local mlruns."
        ) from error
    mlflow.set_experiment(cfg.logging.experiment_name)


def log_resolved_config(cfg: Any, artifact_name: str = "resolved_config.yaml") -> str:
    """Log fully resolved Hydra config as text artifact."""
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow.log_text(resolved, artifact_name)
    return resolved


def save_curve_plot(
    output_path: Path,
    x_values: list[int],
    train_values: list[float],
    val_values: list[float],
    title: str,
    y_label: str,
) -> None:
    """Save train/validation curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, train_values, label="train")
    plt.plot(x_values, val_values, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(
    matrix,
    labels: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        matrix,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
