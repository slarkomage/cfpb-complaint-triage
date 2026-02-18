#!/usr/bin/env bash
set -euo pipefail

poetry install
poetry run pre-commit install
poetry run pre-commit run -a

poetry run python -m cfpb_complaint_triage.commands train \
  data.max_rows=800 \
  train.epochs=1 \
  logging.enable=false

poetry run python -m cfpb_complaint_triage.commands export_onnx

poetry run python -m cfpb_complaint_triage.commands infer_text \
  "I am reporting unauthorized fees on my credit card."

echo "self_check completed successfully"
