#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-artifacts/model.onnx}"
PLAN_PATH="${2:-artifacts/model.plan}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec was not found. Install TensorRT first."
  exit 1
fi

mkdir -p "$(dirname "${PLAN_PATH}")"
trtexec --onnx="${ONNX_PATH}" --saveEngine="${PLAN_PATH}" --fp16
echo "TensorRT engine exported to ${PLAN_PATH}"
