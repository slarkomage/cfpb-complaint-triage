"""LightningDataModule for CFPB text classification."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cfpb_complaint_triage.data.io import read_table


class TextClassificationDataset(Dataset):
    """Torch dataset wrapping tokenized text and labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        text_column: str = "consumer_complaint_narrative",
        label_column: str = "label_id",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            str(row[self.text_column]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.label_column in row:
            item["labels"] = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return item


class CFPBDataModule(L.LightningDataModule):
    """Load train/val/test splits for DistilBERT classifier."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        processed_dir = Path(self.cfg.data.processed_dir)
        train_df = read_table(processed_dir / self.cfg.data.train_filename)
        val_df = read_table(processed_dir / self.cfg.data.val_filename)
        test_df = read_table(processed_dir / self.cfg.data.test_filename)
        max_length = int(self.cfg.train.max_length)
        text_column = "consumer_complaint_narrative"
        self.train_dataset = TextClassificationDataset(
            train_df, self.tokenizer, max_length, text_column
        )
        self.val_dataset = TextClassificationDataset(
            val_df, self.tokenizer, max_length, text_column
        )
        self.test_dataset = TextClassificationDataset(
            test_df, self.tokenizer, max_length, text_column
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.train.num_workers),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.train.num_workers),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.train.num_workers),
        )
