"""LightningModule for DistilBERT product classification."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import AutoModel, get_linear_schedule_with_warmup


class DistilBertClassifier(L.LightningModule):
    """DistilBERT encoder with custom classification head."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        total_training_steps: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_labels)
        self.val_acc = MulticlassAccuracy(num_classes=num_labels)
        self.test_acc = MulticlassAccuracy(num_classes=num_labels)

        self.train_f1_macro = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.val_f1_macro = MulticlassF1Score(num_classes=num_labels, average="macro")
        self.test_f1_macro = MulticlassF1Score(num_classes=num_labels, average="macro")

        self.train_f1_weighted = MulticlassF1Score(num_classes=num_labels, average="weighted")
        self.val_f1_weighted = MulticlassF1Score(num_classes=num_labels, average="weighted")
        self.test_f1_weighted = MulticlassF1Score(num_classes=num_labels, average="weighted")

        self.history_train_loss: list[float] = []
        self.history_val_loss: list[float] = []
        self.history_train_f1: list[float] = []
        self.history_val_f1: list[float] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return logits

    def _step(self, batch, stage: str):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_acc(preds, labels)
            self.train_f1_macro(preds, labels)
            self.train_f1_weighted(preds, labels)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "train/f1_macro", self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "train/f1_weighted",
                self.train_f1_weighted,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        elif stage == "val":
            self.val_acc(preds, labels)
            self.val_f1_macro(preds, labels)
            self.val_f1_weighted(preds, labels)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val/f1_weighted",
                self.val_f1_weighted,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        else:
            self.test_acc(preds, labels)
            self.test_f1_macro(preds, labels)
            self.test_f1_weighted(preds, labels)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "test/f1_macro", self.test_f1_macro, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "test/f1_weighted",
                self.test_f1_weighted,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return loss

    def training_step(self, batch, batch_idx: int):
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        del batch_idx
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        del batch_idx
        return self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics
        if "train/loss" in metric and "train/f1_macro" in metric:
            self.history_train_loss.append(float(metric["train/loss"].cpu().item()))
            self.history_train_f1.append(float(metric["train/f1_macro"].cpu().item()))

    def on_validation_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics
        if "val/loss" in metric and "val/f1_macro" in metric:
            self.history_val_loss.append(float(metric["val/loss"].cpu().item()))
            self.history_val_f1.append(float(metric["val/f1_macro"].cpu().item()))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        warmup_steps = int(self.hparams.total_training_steps * float(self.hparams.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(self.hparams.total_training_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
