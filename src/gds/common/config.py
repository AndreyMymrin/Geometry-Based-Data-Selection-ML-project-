from __future__ import annotations

from pathlib import Path
from typing import Any

from gds.common.io import read_yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    return read_yaml(Path(config_path))

