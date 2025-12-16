from __future__ import annotations

from pathlib import Path
import io
from typing import List, Tuple

import pygame

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

import language
from button import Button
from ui_base import ResponsiveScreen, get_font
from paths import resource_file


class TheoryScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()

        self.background_color = (240, 244, 255)
        self.card_color = (255, 255, 255)
        self.card_border = (215, 220, 235)
        self.shadow_color = (18, 24, 60, 55)
        self.text_color = (34, 38, 52)
        self.muted_text = (110, 118, 134)
        self.accent_color = (72, 104, 255)

        self.pdf_paths = {
            "rus": "_internal/theory/theory_ru.pdf",
            "eng": "_internal/theory/theory_en.pdf",
        }

        self.lang_code = self.lang.lang
        self.original_pages: List[pygame.Surface] = []
        self.scaled_pages: List[Tuple[pygame.Surface, pygame.Rect]] = []
        self.status_lines: List[str] = []
        self.status_surfaces: List[Tuple[pygame.Surface, pygame.Rect]] = []

        self.content_rect: pygame.Rect | None = None
        self.page_index = 0

        self.title_font: pygame.font.Font | None = None
        self.body_font: pygame.font.Font | None = None
        self.label_font: pygame.font.Font | None = None

        self.prev_button: Button | None = None
        self.menu_button: Button | None = None
        self.next_button: Button | None = None

        self.page_label_surface: pygame.Surface | None = None
        self.page_label_rect: pygame.Rect | None = None

        self.last_size = self.app.window_size
        self.pdf_zoom_factor = 1.0
        self._load_pdf_pages()
        self._relayout(self.app.window_size)

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        self.last_size = size
        width, height = size
        width = max(width, 900)
        height = max(height, 620)

        if self.lang.lang != self.lang_code:
            self.lang_code = self.lang.lang
            self._load_pdf_pages()

        margin_x = max(24, int(width * 0.03))
        margin_top = max(24, int(height * 0.04))
        margin_bottom = max(90, int(height * 0.12))
        content_width = max(840, width - 2 * margin_x)
        content_height = max(460, height - margin_top - margin_bottom)

        self.content_rect = pygame.Rect(
            margin_x - 18,
            margin_top - 18,
            content_width + 36,
            content_height + 42,
        )

        title_size = max(30, int(content_width * 0.052))
        body_size = max(20, int(content_width * 0.032))
        label_size = max(18, int(content_width * 0.028))

        self.title_font = get_font(title_size, bold=True)
        self.body_font = get_font(body_size)
        self.label_font = get_font(label_size, bold=False)

        self._scale_pages(content_width, content_height)
        self._build_status_surfaces(
            margin_x=margin_x,
            margin_top=margin_top,
            available_height=content_height,
            content_width=content_width,
        )
        self._build_buttons(width, height)
        self._update_page_label()

    def _scale_pages(self, content_width: int, content_height: int) -> None:
        if not self.original_pages:
            self.scaled_pages = []
            return

        target_width = content_width
        target_height = content_height
        padding = 24
        self.scaled_pages = []

        for surface in self.original_pages:
            sw, sh = surface.get_size()
            rw = max(1, target_width - padding)
            rh = max(1, target_height - padding)
            scale = min(rw / sw, rh / sh)
            scale *= self.pdf_zoom_factor
            new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
            scaled = pygame.transform.smoothscale(surface, new_size)
            rect = scaled.get_rect()
            if self.content_rect:
                rect.center = self.content_rect.center
            else:
                rect.center = (new_size[0] // 2, new_size[1] // 2)
            self.scaled_pages.append((scaled, rect))

        self.page_index = min(self.page_index, len(self.scaled_pages) - 1) if self.scaled_pages else 0

    def _build_status_surfaces(
        self,
        *,
        margin_x: int,
        margin_top: int,
        available_height: int,
        content_width: int,
    ) -> None:
        self.status_surfaces = []
        if not self.status_lines or self.body_font is None:
            return

        line_spacing = max(6, int(self.body_font.get_linesize() * 0.25))
        line_height = self.body_font.get_linesize()
        total_height = len(self.status_lines) * line_height + (len(self.status_lines) - 1) * line_spacing
        start_y = margin_top + max(0, (available_height - total_height) // 2)
        center_x = margin_x + content_width // 2

        current_y = start_y
        for line in self.status_lines:
            surface = self.body_font.render(line, True, self.text_color)
            rect = surface.get_rect()
            rect.centerx = center_x
            rect.y = current_y
            self.status_surfaces.append((surface, rect))
            current_y += line_height + line_spacing

    def _build_buttons(self, width: int, height: int) -> None:
        button_height = max(60, int(height * 0.09))
        nav_width = max(110, int(width * 0.1))
        menu_width = max(280, int(width * 0.3))
        bottom_margin = max(40, int(height * 0.05))
        y = height - bottom_margin - button_height

        button_font = ["SF Pro Display", "Segoe UI", "Arial"]
        button_kwargs = {
            "font": button_font,
            "fontSize": max(24, int(button_height * 0.45)),
            "bold": True,
            "border_radius": 18,
        }

        if self.content_rect is None:
            base_x = 42
            self.content_rect = pygame.Rect(base_x, base_x, width - base_x * 2, height - base_x * 3)

        self.prev_button = Button(
            self.app,
            "<",
            (self.content_rect.left, y),
            (nav_width, button_height),
            self.last_page,
            button_color=(245, 247, 252),
            text_color=self.accent_color,
            border_color=(210, 214, 230),
            shadow_offset=6,
            shadow_color=(0, 0, 0, 40),
            **button_kwargs,
        )

        self.next_button = Button(
            self.app,
            ">",
            (self.content_rect.right - nav_width, y),
            (nav_width, button_height),
            self.next_page,
            button_color=(245, 247, 252),
            text_color=self.accent_color,
            border_color=(210, 214, 230),
            shadow_offset=6,
            shadow_color=(0, 0, 0, 40),
            **button_kwargs,
        )

        left_limit = self.content_rect.left + nav_width + 20
        right_limit = self.content_rect.right - nav_width - 20 - menu_width
        center_x = (width - menu_width) // 2
        menu_x = max(left_limit, center_x)
        if right_limit >= left_limit:
            menu_x = min(menu_x, right_limit)
        self.menu_button = Button(
            self.app,
            self.lang["btn_menu"],
            (menu_x, y),
            (menu_width, button_height),
            self.to_menu,
            button_color=self.accent_color,
            text_color=(255, 255, 255),
            shadow_offset=9,
            shadow_color=(64, 99, 255, 110),
            **button_kwargs,
        )

    def _update_page_label(self) -> None:
        if self.label_font is None or not self.scaled_pages:
            self.page_label_surface = None
            self.page_label_rect = None
            return

        total = len(self.scaled_pages)
        current = self.page_index + 1
        templates = {
            "rus": "Страница {current}/{total}",
            "eng": "Page {current}/{total}",
        }
        text = templates.get(self.lang.lang, templates["eng"]).format(current=current, total=total)
        self.page_label_surface = self.label_font.render(text, True, self.muted_text)
        rect = self.page_label_surface.get_rect()
        rect.centerx = self.content_rect.centerx if self.content_rect else self.app.screen.get_rect().centerx
        rect.bottom = (self.menu_button.rect.top - 16) if self.menu_button else self.app.screen.get_rect().height - 48
        self.page_label_rect = rect

    # ------------------------------------------------------------------ Loading helpers
    def _load_pdf_pages(self) -> None:
        self.original_pages = []
        self.scaled_pages = []
        self.status_lines = []
        self.page_index = 0

        relative = self.pdf_paths.get(self.lang_code, self.pdf_paths["eng"])
        path = resource_file(relative)

        if fitz is None:
            self.status_lines = self._status_message("missing_lib", relative)
            return

        if not path.exists():
            self.status_lines = self._status_message("missing_file", relative)
            return

        try:
            doc = fitz.open(path)
        except Exception as exc:  # pragma: no cover
            self.status_lines = self._status_message("load_error", str(exc))
            return

        try:
            scale = fitz.Matrix(2.0, 2.0)
            for page in doc:
                pix = page.get_pixmap(matrix=scale)
                image_bytes = pix.tobytes("png")
                buffer = io.BytesIO(image_bytes)
                try:
                    surface = pygame.image.load(buffer)
                finally:
                    buffer.close()
                if surface.get_flags() & pygame.SRCALPHA:
                    surface = surface.convert_alpha()
                else:
                    surface = surface.convert()
                self.original_pages.append(surface)
        finally:
            doc.close()

        if not self.original_pages:
            self.status_lines = self._status_message("empty_pdf", str(path))

    def _status_message(self, key: str, detail: str) -> List[str]:
        templates = {
            "missing_lib": {
                "rus": [
                    "Не удалось отобразить PDF.",
                    "Установите пакет pymupdf: pip install pymupdf.",
                ],
                "eng": [
                    "Cannot render PDF pages.",
                    "Install the pymupdf package: pip install pymupdf.",
                ],
            },
            "missing_file": {
                "rus": [
                    "Файл теории не найден:",
                    "{detail}",
                    "Скомпилируйте theory_ru.tex и theory_en.tex и поместите PDF в _internal/theory/.",
                ],
                "eng": [
                    "Theory PDF is missing:",
                    "{detail}",
                    "Compile theory_en.tex and theory_ru.tex, then place PDFs into _internal/theory/.",
                ],
            },
            "load_error": {
                "rus": [
                    "Ошибка при чтении PDF:",
                    "{detail}",
                ],
                "eng": [
                    "Error reading PDF:",
                    "{detail}",
                ],
            },
            "empty_pdf": {
                "rus": [
                    "PDF не содержит страниц: {detail}",
                ],
                "eng": [
                    "PDF has no pages: {detail}",
                ],
            },
        }

        messages = templates.get(key, templates["load_error"])
        selected = messages.get(self.lang_code, messages["eng"])
        return [line.format(detail=detail) for line in selected]

    # ---------------------------------------------------------------- Buttons
    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def last_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._update_page_label()

    def next_page(self):
        if self.page_index < len(self.scaled_pages) - 1:
            self.page_index += 1
            self._update_page_label()

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        self.screen.fill(self.background_color)

        if self.content_rect:
            shadow_rect = self.content_rect.move(14, 18)
            shadow_surface = pygame.Surface(self.content_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=26)
            self.screen.blit(shadow_surface, shadow_rect.topleft)
            pygame.draw.rect(self.screen, self.card_color, self.content_rect, border_radius=24)
            pygame.draw.rect(self.screen, self.card_border, self.content_rect, width=2, border_radius=24)

        if self.scaled_pages:
            surface, rect = self.scaled_pages[self.page_index]
            self.screen.blit(surface, rect)
        elif self.status_surfaces:
            for surface, rect in self.status_surfaces:
                self.screen.blit(surface, rect)

        for button in (self.prev_button, self.menu_button, self.next_button):
            if button:
                button.draw_button()

        if self.page_label_surface and self.page_label_rect:
            self.screen.blit(self.page_label_surface, self.page_label_rect)

    # ---------------------------------------------------------------- Events
    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)

    def _check_buttons(self, mouse_position):
        for button in (self.prev_button, self.menu_button, self.next_button):
            if button and button.rect.collidepoint(mouse_position):
                button.command()
