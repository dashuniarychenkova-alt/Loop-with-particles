from __future__ import annotations

from typing import List, Tuple

import pygame

import language
from button import Button
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient


class TutorialScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()

        self.primary_color = (72, 104, 255)
        self.text_color = (35, 38, 46)
        self.muted_text = (96, 105, 124)

        self.background: pygame.Surface | None = None
        self.card_rect: pygame.Rect | None = None
        self.card_padding = 0
        self.text_layout: List[Tuple[pygame.Surface, Tuple[int, int]]] = []
        self.title_font = get_font(30, bold=True)
        self.heading_font = get_font(22, bold=True)
        self.body_font = get_font(18)
        self.back_button: Button | None = None
        self.scroll_offset = 0
        self.scroll_view_height = 0
        self.content_height = 0
        self.scroll_max_offset = 0
        self.scroll_step = 48

        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self._relayout(self.app.window_size)

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        width = max(width, 900)
        height = max(height, 650)
        self.background = build_vertical_gradient(size, (230, 236, 255), (247, 248, 254))

        outer_padding = max(40, width // 18)
        content_width = min(960, width - outer_padding * 2)
        content_width = max(content_width, min(420, width - outer_padding * 2))
        card_padding = max(24, content_width // 18)
        inner_width = max(180, content_width - card_padding * 2)
        content = self._resolve_tutorial_content()
        layout, content_height = self._build_tutorial_lines(inner_width, content)
        button_height = 60
        button_margin = max(20, outer_padding // 2)
        card_top = outer_padding + 10
        available_card_space = height - card_top - outer_padding - button_height - button_margin
        min_card_height = card_padding * 2 + 120
        max_card_height = max(min_card_height, available_card_space)
        card_height = min(content_height + card_padding * 2, max_card_height)
        card_height = max(card_height, min_card_height)
        self.content_height = content_height
        self.scroll_view_height = max(0, card_height - card_padding * 2)
        self.scroll_max_offset = max(0, self.content_height - self.scroll_view_height)
        self.scroll_offset = min(self.scroll_offset, self.scroll_max_offset)
        card_left = (width - content_width) // 2
        self.card_rect = pygame.Rect(card_left, card_top, content_width, card_height)
        self.card_padding = card_padding
        self.text_layout = layout
        self._build_back_button(
            width,
            height,
            card_top,
            card_height,
            outer_padding,
            button_height,
            button_margin,
        )

    def _build_back_button(
        self,
        width: int,
        height: int,
        card_top: int,
        card_height: int,
        outer_padding: int,
        button_height: int,
        button_margin: int,
    ) -> None:
        button_width = min(360, max(260, width // 4))
        button_x = (width - button_width) // 2
        min_y = card_top + card_height + button_margin
        button_y = min(height - outer_padding - button_height, min_y)
        self.back_button = Button(
            self.app,
            self.lang['btn_menu'],
            (button_x, button_y),
            (button_width, button_height),
            self.to_menu,
            font=["SF Pro Display", "Segoe UI", "Arial"],
            fontSize=max(22, int(button_height * 0.42)),
            bold=True,
            button_color=self.primary_color,
            text_color=(255, 255, 255),
            border_radius=18,
            shadow_offset=8,
            shadow_color=(64, 99, 255, 120),
        )

    def _build_tutorial_lines(
        self,
        max_width: int,
        content: dict,
    ) -> Tuple[List[Tuple[pygame.Surface, Tuple[int, int]]], int]:
        max_width = max(120, max_width)
        layout: List[Tuple[pygame.Surface, Tuple[int, int]]] = []
        y_offset = 0
        max_bottom = 0
        title_text = str(content.get('title', '') or '')
        if title_text:
            title_surface = self.title_font.render(title_text, True, self.text_color)
            layout.append((title_surface, (0, y_offset)))
            max_bottom = max(max_bottom, y_offset + title_surface.get_height())
            y_offset += title_surface.get_height() + max(12, max_width // 18)
        heading_gap = max(10, max_width // 28)
        line_gap = max(6, max_width // 60)
        section_gap = max(18, max_width // 20)
        sections = content.get('sections', [])
        for index, section in enumerate(sections):
            heading = str(section.get('heading', '') or '')
            if heading:
                heading_surface = self.heading_font.render(heading, True, self.primary_color)
                layout.append((heading_surface, (0, y_offset)))
                max_bottom = max(max_bottom, y_offset + heading_surface.get_height())
                y_offset += heading_surface.get_height() + heading_gap
            items = section.get('items', [])
            for item in items:
                wrapped_lines = self._wrap_text_lines(str(item), self.body_font, max_width)
                for line in wrapped_lines:
                    if not line:
                        y_offset += self.body_font.get_linesize() // 2
                        max_bottom = max(max_bottom, y_offset)
                        continue
                    surface = self.body_font.render(line, True, self.muted_text)
                    layout.append((surface, (0, y_offset)))
                    max_bottom = max(max_bottom, y_offset + surface.get_height())
                    y_offset += surface.get_height() + line_gap
            if index != len(sections) - 1:
                y_offset += section_gap
                max_bottom = max(max_bottom, y_offset)
        content_height = max(0, max_bottom)
        return layout, content_height

    def _wrap_text_lines(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        if not text:
            return ['']
        lines: List[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                lines.append('')
                continue
            lines.extend(self._wrap_single_line(stripped, font, max_width))
        return lines

    def _wrap_single_line(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        max_width = max(80, max_width)
        bullet_prefix = ''
        indent = ''
        body = text
        if text.startswith('- '):
            bullet_prefix = '- '
            body = text[2:].strip()
            indent = '  '
        words = body.split()
        if not words:
            return [bullet_prefix.strip() or '']
        lines: List[str] = []
        current_line = bullet_prefix + words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}" if current_line else word
            if font.size(candidate)[0] <= max_width:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = f"{indent}{word}" if indent else word
        lines.append(current_line)
        return lines

    def _resolve_tutorial_content(self) -> dict:
        if self.lang.lang == 'rus':
            fallback = {
                'title': 'Краткий туториал',
                'sections': [
                    {
                        'heading': 'Главные действия',
                        'items': ['- Нажмите «Демонстрация», чтобы открыть сцену.'],
                    }
                ],
            }
        else:
            fallback = {
                'title': 'Quick tutorial',
                'sections': [
                    {
                        'heading': 'Main actions',
                        'items': ['- Press "Demonstration" to open the scene.'],
                    }
                ],
            }
        try:
            raw_content = self.lang['menu_tutorial']
        except KeyError:
            raw_content = None
        if not isinstance(raw_content, dict):
            return fallback
        title = str(raw_content.get('title') or fallback['title'])
        sections_data = raw_content.get('sections')
        parsed_sections: List[dict] = []
        if isinstance(sections_data, list):
            for entry in sections_data:
                if not isinstance(entry, dict):
                    continue
                heading = str(entry.get('heading', '') or '').strip()
                items_raw = entry.get('items', [])
                if not heading or not isinstance(items_raw, list):
                    continue
                items = [str(item).strip() for item in items_raw if isinstance(item, str) and item.strip()]
                if items:
                    parsed_sections.append({'heading': heading, 'items': items})
        if not parsed_sections:
            parsed_sections = [{'heading': sec['heading'], 'items': list(sec['items'])} for sec in fallback['sections']]
        return {'title': title, 'sections': parsed_sections}

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        assert self.background is not None
        self.screen.blit(self.background, (0, 0))
        if self.card_rect is not None:
            self._draw_card(self.card_rect)
            base_x = self.card_rect.left + self.card_padding
            base_y = self.card_rect.top + self.card_padding - self.scroll_offset
            inner_rect = None
            prev_clip = None
            if self.scroll_view_height > 0:
                inner_rect = pygame.Rect(
                    base_x,
                    self.card_rect.top + self.card_padding,
                    max(0, self.card_rect.width - self.card_padding * 2),
                    self.scroll_view_height,
                )
                prev_clip = self.screen.get_clip()
                self.screen.set_clip(inner_rect)
            for surface, (offset_x, offset_y) in self.text_layout:
                self.screen.blit(surface, (base_x + offset_x, base_y + offset_y))
            if inner_rect is not None and prev_clip is not None:
                self.screen.set_clip(prev_clip)
        if self.back_button:
            self.back_button.draw_button()

    def _draw_card(self, rect: pygame.Rect) -> None:
        shadow_rect = rect.copy()
        shadow_rect.x += 12
        shadow_rect.y += 16
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (15, 22, 58, 60), shadow_surface.get_rect(), border_radius=28)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=26)
        pygame.draw.rect(self.screen, (215, 220, 235), rect, width=2, border_radius=26)

    # ---------------------------------------------------------------- Events
    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                if self.back_button and self.back_button.rect.collidepoint(mouse_position):
                    self.back_button.command()
            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_by(-event.y * self.scroll_step)

    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _scroll_by(self, delta: float) -> None:
        if self.scroll_max_offset <= 0:
            return
        self.scroll_offset = min(max(self.scroll_offset + delta, 0), self.scroll_max_offset)
