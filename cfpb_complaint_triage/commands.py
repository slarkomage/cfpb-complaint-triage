"""Public Fire CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cfpb_complaint_triage.data.download import ensure_data
from cfpb_complaint_triage.data.preprocess import preprocess_data
from cfpb_complaint_triage.production.export_onnx import export_onnx
from cfpb_complaint_triage.production.infer_onnx import infer_batch, infer_text
from cfpb_complaint_triage.production.serving_instructions import print_serving_commands
from cfpb_complaint_triage.training.baseline import baseline_train
from cfpb_complaint_triage.training.train import train_pipeline
from cfpb_complaint_triage.utils.paths import config_dir


def compose_config(overrides: list[str] | None = None):
    """Compose hydra config using config directory."""
    config_directory = str(config_dir().resolve())
    with initialize_config_dir(config_dir=config_directory, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


class CFPBTriageCommands:
    """Command collection exposed through python-fire."""

    def print_config(self, *overrides: str) -> str:
        cfg = compose_config(list(overrides))
        rendered = OmegaConf.to_yaml(cfg, resolve=True)
        print(rendered)
        return rendered

    def ensure_data(self, *overrides: str) -> str:
        cfg = compose_config(list(overrides))
        path = ensure_data(cfg)
        print(str(path))
        return str(path)

    def preprocess(self, *overrides: str) -> dict[str, str]:
        cfg = compose_config(list(overrides))
        ensure_data(cfg)
        paths = preprocess_data(cfg)
        payload = {key: str(value) for key, value in paths.items()}
        print(payload)
        return payload

    def train(self, *overrides: str) -> None:
        cfg = compose_config(list(overrides))
        train_pipeline(cfg)
        print("training completed")

    def baseline_train(self, *overrides: str) -> None:
        cfg = compose_config(list(overrides))
        baseline_train(cfg)

    def export_onnx(self, *overrides: str) -> str:
        cfg = compose_config(list(overrides))
        onnx_path = export_onnx(cfg)
        print(str(onnx_path))
        return str(onnx_path)

    def infer_text(self, text: str, *overrides: str) -> dict[str, Any]:
        cfg = compose_config(list(overrides))
        ensure_data(cfg)
        prediction = infer_text(cfg, text=text)
        print(prediction)
        return prediction

    def infer_batch(
        self,
        input_path: str,
        output_path: str,
        text_column: str | None = None,
        *overrides: str,
    ) -> str:
        cfg = compose_config(list(overrides))
        ensure_data(cfg)
        output = infer_batch(
            cfg, input_path=input_path, output_path=output_path, text_column=text_column
        )
        print(output)
        return output

    def serving_instructions(self) -> None:
        print_serving_commands()

    def api_instructions(self) -> None:
        print(
            "poetry run uvicorn cfpb_complaint_triage.production.api:create_app "
            "--factory --host 0.0.0.0 --port 8000"
        )
        print("curl -X GET http://127.0.0.1:8000/health")
        print(
            "curl -X POST http://127.0.0.1:8000/predict "
            "-H 'Content-Type: application/json' "
            '-d \'{"text": "I was charged incorrect fees"}\''
        )

    def self_check(self) -> str:
        script_path = Path("scripts") / "self_check.sh"
        print(str(script_path))
        return str(script_path)


def main() -> None:
    """Main Fire entrypoint."""
    fire.Fire(CFPBTriageCommands)


if __name__ == "__main__":
    main()
