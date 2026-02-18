"""Project path helpers based on pathlib."""

from pathlib import Path


def project_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """Return python package root directory."""
    return Path(__file__).resolve().parents[1]


def config_dir() -> Path:
    """Return Hydra config directory."""
    return project_root() / "configs"


def data_dir() -> Path:
    """Return data directory."""
    return project_root() / "data"


def raw_data_dir() -> Path:
    """Return raw data directory."""
    return data_dir() / "raw"


def processed_data_dir() -> Path:
    """Return processed data directory."""
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    """Return artifacts directory."""
    return project_root() / "artifacts"


def plots_dir() -> Path:
    """Return plots directory."""
    return project_root() / "plots"
