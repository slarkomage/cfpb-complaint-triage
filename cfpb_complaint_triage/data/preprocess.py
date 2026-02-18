"""Preprocess CFPB complaints for model training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cfpb_complaint_triage.data.io import read_table, write_table


def _normalize_columns(dataframe: pd.DataFrame, cfg) -> pd.DataFrame:
    rename_map = {
        cfg.data.id_column: "complaint_id",
        cfg.data.text_column: "consumer_complaint_narrative",
        cfg.data.target_column: "product",
    }
    optional_columns = {
        cfg.data.issue_column: "issue",
        cfg.data.timely_response_column: "timely_response",
    }
    for source_name, target_name in optional_columns.items():
        if source_name in dataframe.columns:
            rename_map[source_name] = target_name
    return dataframe.rename(columns=rename_map)


def preprocess_data(cfg) -> dict[str, Path]:
    """Preprocess raw data into train/val/test splits and label maps."""
    raw_path = Path(cfg.data.raw_path)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataframe = read_table(raw_path)
    dataframe = _normalize_columns(dataframe, cfg)
    required_columns = {"complaint_id", "consumer_complaint_narrative", "product"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.dropna(subset=["consumer_complaint_narrative", "product"])
    dataframe["consumer_complaint_narrative"] = dataframe["consumer_complaint_narrative"].astype(
        str
    )
    dataframe["product"] = dataframe["product"].astype(str)
    dataframe = dataframe[dataframe["consumer_complaint_narrative"].str.strip().str.len() > 0]

    top_products = (
        dataframe["product"].value_counts().head(int(cfg.data.top_k_products)).index.tolist()
    )
    dataframe = dataframe[dataframe["product"].isin(top_products)].copy()

    max_rows = int(cfg.data.max_rows)
    if max_rows > 0 and len(dataframe) > max_rows:
        dataframe = dataframe.sample(max_rows, random_state=int(cfg.preprocess.seed))

    dataframe["label_id"] = dataframe["product"].astype("category").cat.codes
    label_to_id = (
        dataframe[["product", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("product")["label_id"]
        .to_dict()
    )
    id_to_label = {int(value): key for key, value in label_to_id.items()}

    train_ratio = float(cfg.preprocess.train_ratio)
    val_ratio = float(cfg.preprocess.val_ratio)
    test_ratio = float(cfg.preprocess.test_ratio)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=(1.0 - train_ratio),
        random_state=int(cfg.preprocess.seed),
        stratify=dataframe["label_id"],
    )
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=int(cfg.preprocess.seed),
        stratify=temp_df["label_id"],
    )

    train_path = processed_dir / cfg.data.train_filename
    val_path = processed_dir / cfg.data.val_filename
    test_path = processed_dir / cfg.data.test_filename
    write_table(train_df.reset_index(drop=True), train_path)
    write_table(val_df.reset_index(drop=True), val_path)
    write_table(test_df.reset_index(drop=True), test_path)

    label_map_path = processed_dir / cfg.data.label_map_filename
    label_map_payload = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_labels": len(label_to_id),
    }
    label_map_path.write_text(json.dumps(label_map_payload, indent=2), encoding="utf-8")
    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "label_map_path": label_map_path,
    }
