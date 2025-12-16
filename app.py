import pygame
try:
    import screeninfo
except Exception:  # pragma: no cover - fallback for platforms without screeninfo deps
    screeninfo = None

import config
import language
from authors_screen import AuthorsScreen
from demo_screen import DemoScreen
from menu_screen import MenuScreen
from theory_screen import TheoryScreen
from tutorial_screen import TutorialScreen


class App:
    MIN_WINDOW = (1024, 640)
    WINDOW_SCALE = 0.86

    def __init__(self):
        pygame.init()
        self.monitor = self._detect_monitor()
        self.window_size = (self.monitor.width, self.monitor.height)
        # Prefer using SCALED when available (pygame 2+) to handle HiDPI / Retina displays
        # This prevents blurry UI on macOS by letting SDL/Pygame manage high-DPI scaling.
        self.display_flags = pygame.FULLSCREEN
        if hasattr(pygame, 'SCALED'):
            self.display_flags |= pygame.SCALED
        self.screen = pygame.display.set_mode(self.window_size, self.display_flags)

        self.clock = pygame.time.Clock()
        self._config = config.ConfigLoader()
        self.onboarding_enabled = bool(self._config['show_onboarding'])

        self.menu_screen = MenuScreen(self)
        self.authors_screen = AuthorsScreen(self)
        self.theory_screen = TheoryScreen(self)
        self.demo_screen = DemoScreen(self)
        self.tutorial_screen = TutorialScreen(self)

        # Onboarding state flags shared across screens
        self.onboarding_menu_done = False
        self.onboarding_demo_done = False
        self.onboarding_demo_pending = False

        self._screens = (
            self.menu_screen,
            self.authors_screen,
            self.theory_screen,
            self.demo_screen,
            self.tutorial_screen,
        )
        self._notify_resize(self.window_size)

        self.active_screen = self.menu_screen

    # ------------------------------------------------------------------ Locale
    def set_language(self, language_code: str) -> None:
        """Persist the selected language and notify screens to refresh text."""
        cfg = config.ConfigLoader()
        available = cfg['language_files']
        if language_code not in available:
            raise ValueError(f"Unknown language code: {language_code!r}")

        current = cfg['language']
        if current == language_code:
            return

        cfg.set('language', language_code)
        language.Language().reload()

        for screen in self._screens:
            handler = getattr(screen, "on_language_change", None)
            if callable(handler):
                handler()
        self._notify_resize(self.window_size)

    def toggle_language(self) -> None:
        cfg = config.ConfigLoader()
        current = cfg['language']
        available = list(cfg['language_files'])
        if len(available) < 2:
            return
        current_index = available.index(current)
        next_lang = available[(current_index + 1) % len(available)]
        self.set_language(next_lang)

    def _get_initial_window_size(self, monitor) -> tuple[int, int]:
        width = int(monitor.width * self.WINDOW_SCALE)
        height = int(monitor.height * self.WINDOW_SCALE)
        min_w, min_h = self.MIN_WINDOW
        return max(min_w, width), max(min_h, height)

    def _notify_resize(self, size: tuple[int, int]) -> None:
        for screen in self._screens:
            if hasattr(screen, "on_window_resize"):
                screen.on_window_resize(size)

    def handle_resize(self, size: tuple[int, int]) -> None:
        min_w, min_h = self.MIN_WINDOW
        width = max(min_w, size[0])
        height = max(min_h, size[1])
        resized = (width, height)
        if resized == self.window_size:
            return
        self.window_size = resized
        # Re-create the display with the same flags so SCALED (if enabled) continues to apply
        self.screen = pygame.display.set_mode(self.window_size, self.display_flags)
        self._notify_resize(self.window_size)

    def _detect_monitor(self):
        """Detect monitor size even when screeninfo is unavailable."""
        if screeninfo is not None:
            try:
                monitor = screeninfo.get_monitors()[0]
                monitor.width = int(getattr(monitor, "width", self.MIN_WINDOW[0]))
                monitor.height = int(getattr(monitor, "height", self.MIN_WINDOW[1]))
                return monitor
            except Exception:
                pass

        info = pygame.display.Info()

        class _Monitor:
            width = App.MIN_WINDOW[0]
            height = App.MIN_WINDOW[1]

        fallback = _Monitor()
        if getattr(info, "current_w", 0) and getattr(info, "current_h", 0):
            fallback.width = int(info.current_w)
            fallback.height = int(info.current_h)
        return fallback

    def run(self):
        """Запуск основного цикла игры."""
        while True:
            # Mouse and keyboard events handling
            self.active_screen._check_events()
            self.active_screen._update_screen()
            # Отображение последнего прорисованного экрана.
            pygame.display.flip()

            fps = self._config['FPS']
            self.clock.tick(fps)


if __name__ == '__main__':
    app = App()
    app.run()
