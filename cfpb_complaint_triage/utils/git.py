"""Git helpers."""

import subprocess
from pathlib import Path


def get_commit_id(cwd: Path | None = None) -> str:
    """Return git commit id if available."""
    command = ["git", "rev-parse", "HEAD"]
    try:
        output = subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"
