"""Print commands for MLflow model serving."""


def print_serving_commands() -> None:
    """Print serving commands for MLflow pyfunc model."""
    commands = [
        'poetry run python -c "import mlflow.pyfunc as m; '
        "from cfpb_complaint_triage.production.mlflow_pyfunc import CFPBOnnxPyfuncModel; "
        "m.save_model(path='artifacts/mlflow_onnx_model', "
        "python_model=CFPBOnnxPyfuncModel(bundle_dir='artifacts/infer_bundle'))\"",
        "MLFLOW_TRACKING_URI=http://127.0.0.1:8080 poetry run mlflow models serve "
        "-m artifacts/mlflow_onnx_model -p 5001 --no-conda",
        "curl -X POST http://127.0.0.1:5001/invocations "
        "-H 'Content-Type: application/json' "
        '-d \'{"dataframe_records": [{"text": "I was charged incorrect fees"}]}\'',
    ]
    print("\n".join(commands))
