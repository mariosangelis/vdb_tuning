"""
Central config loader for the vdb_tuning experiments.

Reads config.yaml (machine-specific paths/flags) from the repo root so that
all Python scripts share a single source of truth. Use:

    from config_loader import CONFIG, get_path
    PG_INSTALL   = CONFIG["paths"]["pg_install"]
    SNAPSHOT_DIR = Path(get_path("snapshot_dir"))

The config.yaml lives at the repository root (vdb-tuning-main/config.yaml).
This module locates it by walking up from its own directory.
"""
import os
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to read config.yaml. Install with: pip install pyyaml"
    ) from e


def _find_config():
    """Locate config.yaml.

    Order:
      1. $VDB_CONFIG env var (explicit override)
      2. config.yaml next to this file (repo root)
      3. walk up parent dirs looking for config.yaml
    """
    env = os.environ.get("VDB_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VDB_CONFIG points to missing file: {env}")

    here = Path(__file__).resolve().parent
    candidate = here / "config.yaml"
    if candidate.is_file():
        return candidate

    for parent in here.parents:
        c = parent / "config.yaml"
        if c.is_file():
            return c

    raise FileNotFoundError(
        "config.yaml not found. Place it at the repo root (vdb-tuning-main/config.yaml) "
        "or set VDB_CONFIG=/path/to/config.yaml"
    )


CONFIG_PATH = _find_config()
with open(CONFIG_PATH, "r") as _f:
    CONFIG = yaml.safe_load(_f)


def get_path(key):
    """Return a path value from the 'paths' section by key."""
    try:
        return CONFIG["paths"][key]
    except KeyError as e:
        raise KeyError(f"paths.{key} not found in {CONFIG_PATH}") from e
