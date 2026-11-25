from pathlib import Path
from typing import Any
import yaml

from .paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)