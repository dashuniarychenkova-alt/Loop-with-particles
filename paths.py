"""Utilities for resolving resource and configuration paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "SpaceParticles"


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else fallback


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Base directory containing packaged resources."""
    if is_frozen():
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            return Path(bundle_dir)
    return Path(__file__).resolve().parent


def resource_path(relative: str | Path) -> Path:
    """Resolve `relative` inside the resource root."""
    return resource_root() / Path(relative)


def resource_file(relative: str | Path) -> Path:
    """Resolve file path that might be wrapped into an extra directory."""
    path = resource_path(relative)
    if path.is_dir():
        candidate = path / Path(relative).name
        if candidate.exists():
            return candidate
    return path


def user_config_path() -> Path:
    """Writable location for user config in packaged builds."""
    if sys.platform == "win32":
        base = _env_path("APPDATA", Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = _env_path("XDG_CONFIG_HOME", Path.home() / ".config")
    return base / APP_NAME / "config.json"
