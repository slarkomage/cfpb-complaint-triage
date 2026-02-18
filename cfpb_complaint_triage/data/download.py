"""Dataset download and fallback orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from cfpb_complaint_triage.data.dvc_utils import try_dvc_pull
from cfpb_complaint_triage.data.io import write_table
from cfpb_complaint_triage.utils.paths import project_root, raw_data_dir


def download_data(cfg) -> Path:
    """Download raw data from configured source URL."""
    source_url = cfg.data.source_url
    destination = Path(cfg.data.raw_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(source_url, timeout=120)
    response.raise_for_status()
    if source_url.endswith(".parquet"):
        destination.write_bytes(response.content)
        return destination
    if source_url.endswith(".json"):
        payload = response.json()
        dataframe = pd.DataFrame(payload)
        write_table(dataframe, destination)
        return destination
    destination.write_text(response.text, encoding="utf-8")
    if destination.suffix.lower() != ".csv":
        dataframe = pd.read_csv(destination)
        destination = destination.with_suffix(".parquet")
        write_table(dataframe, destination)
    return destination


def create_synthetic_data(cfg) -> Path:
    """Create tiny synthetic dataset for smoke execution."""
    destination = Path(cfg.data.raw_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    product_texts = {
        "Credit card": "I was charged fee and credit card interest was wrong.",
        "Mortgage": "My mortgage payment and escrow account are incorrect.",
        "Debt collection": "Collector contacted me and reported invalid debt details.",
        "Checking or savings account": "My checking account transfer failed unexpectedly.",
    }
    row_index = 0
    for product_name, text_template in product_texts.items():
        for repeat_index in range(30):
            records.append(
                {
                    "complaint_id": f"synthetic-{row_index}",
                    "consumer_complaint_narrative": f"{text_template} Case {repeat_index}.",
                    "product": product_name,
                    "issue": "Synthetic issue",
                    "timely_response": "Yes" if repeat_index % 2 == 0 else "No",
                }
            )
            row_index += 1
    dataframe = pd.DataFrame(records)
    write_table(dataframe, destination)
    metadata_path = raw_data_dir() / "synthetic_fallback.json"
    metadata_path.write_text(
        json.dumps({"synthetic_fallback": True, "rows": len(dataframe)}, indent=2),
        encoding="utf-8",
    )
    print("synthetic fallback: generated tiny dataset for smoke run")
    return destination


def ensure_data(cfg) -> Path:
    """Ensure raw data exists via DVC pull, download, or synthetic fallback."""
    destination = Path(cfg.data.raw_path)
    if destination.exists():
        return destination

    project = project_root()
    if try_dvc_pull(project):
        if destination.exists():
            return destination

    try:
        return download_data(cfg)
    except Exception as download_error:
        print(f"download failed: {download_error}")
        return create_synthetic_data(cfg)
