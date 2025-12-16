import json
from pathlib import Path

import configloader

from paths import is_frozen, resource_file, user_config_path
from singleton import singleton


@singleton
class ConfigLoader(object):
    def __init__(self):
        self._loader = configloader.ConfigLoader()
        self._default_path = resource_file("config.json")
        self._path = self._resolve_path()
        self.update()

    def update(self):
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._loader.update_from_json_file(f)
        except FileNotFoundError:
            if self._path != self._default_path:
                self._ensure_user_config(self._path)
                with self._path.open("r", encoding="utf-8") as f:
                    self._loader.update_from_json_file(f)
            else:
                raise
        # Backfill missing keys from the default config to stay compatible
        self._apply_defaults()

    def __getitem__(self, item):
        return self._loader[item]

    def get(self, item, default=None):
        """Return config value or default when key is absent."""
        return self._loader.get(item, default)

    def set(self, key: str | tuple, value):
        """
        Change record with key in config.
        It's not implemented by __setitem__ for config safety
        :param key: If key is str then changing cfg[key].
        If key is tuple (key_1, ..., key_n) then changing cfg[key_1][...][key_n]
        :param value: New value
        """

        if isinstance(key, str):
            if key not in self._loader:
                raise ValueError('key not in config keys')
            self._loader[key] = value
        else:
            to_update = self._loader
            for k in key[:-1]:
                if k not in to_update:
                    raise ValueError('key not in config keys')
                to_update = to_update[k]

            if key[-1] not in to_update:
                raise ValueError('key not in config keys')
            to_update[key[-1]] = value

        with self._path.open("w", encoding="utf-8") as f:
            json.dump(obj=dict(self._loader), fp=f, ensure_ascii=False, indent=2, separators=(',', ': '))

    def _resolve_path(self) -> Path:
        if is_frozen():
            user_path = user_config_path()
            self._ensure_user_config(user_path)
            return user_path
        return self._default_path

    def _ensure_user_config(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            target_path.write_text(self._default_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _apply_defaults(self) -> None:
        """Populate missing keys from the bundled default config."""
        try:
            with self._default_path.open("r", encoding="utf-8") as f:
                defaults = json.load(f)
        except Exception:
            return
        if not isinstance(defaults, dict):
            return
        for key, value in defaults.items():
            if key not in self._loader:
                self._loader[key] = value

