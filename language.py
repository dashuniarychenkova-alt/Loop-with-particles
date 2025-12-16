from pathlib import Path

import configloader

from config import ConfigLoader
from paths import resource_file
from singleton import singleton


@singleton
class Language(object):
    def __init__(self):
        self.lang, self.lang_path = self._current_language()
        self._loader = configloader.ConfigLoader()
        self.update()

    def update(self):
        with self.lang_path.open("r", encoding="utf-8") as f:
            self._loader.update_from_json_file(f)

    def reload(self):
        self.lang, self.lang_path = self._current_language()
        self.update()

    def __getitem__(self, item):
        return self._loader[item]

    @staticmethod
    def _current_language() -> tuple[str, Path]:
        cfg = ConfigLoader()
        lang = cfg['language']
        rel_path = cfg['language_files'][lang]
        return lang, resource_file(rel_path)
