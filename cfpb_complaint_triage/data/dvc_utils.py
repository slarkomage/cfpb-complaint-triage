"""DVC helper utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path


def try_dvc_pull(project_root: Path) -> bool:
    """Try to run `dvc pull`, return True on success."""
    command = ["dvc", "pull"]
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0
