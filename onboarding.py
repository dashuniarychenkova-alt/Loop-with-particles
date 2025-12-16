from __future__ import annotations

import pygame
from typing import Callable, Iterable, Optional

from ui_base import get_font, calc_scale


class OnboardingGuide:
    """Lightweight step-by-step overlay that dims the UI except highlighted areas."""

    def __init__(
        self,
        app,
        *,
        overlay_color: tuple[int, int, int, int] = (10, 14, 32, 200),
        accent_color: tuple[int, int, int] = (98, 181, 255),
    ) -> None:
        self.app = app
        self.overlay_color = overlay_color
        self.accent_color = accent_color
        self.accent_soft = tuple(min(255, int(c * 1.08)) for c in accent_color)
        self._scale = 1.0
        self.title_font = get_font(26, bold=True)
        self.body_font = get_font(18)
        self.hint_font = get_font(16)
        self.active: bool = False
        self.steps: list[dict] = []
        self.index: int = 0
        self.on_complete: Optional[Callable[[bool], None]] = None
        self._hint_text = ""

    # ------------------------------------------------------------------ Public API
    def start(
        self,
        steps: Iterable[dict],
        *,
        on_complete: Optional[Callable[[bool], None]] = None,
        hint: str = "",
    ) -> None:
        self.steps = list(steps)
        self.index = 0
        self.on_complete = on_complete
        self._hint_text = hint
        self.active = bool(self.steps)
        self._refresh_fonts(self._scale)

    def stop(self, aborted: bool = False) -> None:
        callback = self.on_complete
        self.active = False
        self.steps = []
        self.index = 0
        self.on_complete = None
        if callback:
            callback(aborted)

    def next(self) -> None:
        if not self.active:
            return
        self.index += 1
        if self.index >= len(self.steps):
            self.stop(aborted=False)

    def prev(self) -> None:
        if not self.active:
            return
        self.index = max(0, self.index - 1)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the event is consumed by the overlay."""
        if not self.active:
            return False
        if event.type in (pygame.QUIT, pygame.VIDEORESIZE):
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_RIGHT):
                self.next()
                return True
            if event.key in (pygame.K_LEFT, pygame.K_BACKSPACE):
                self.prev()
                return True
            if event.key == pygame.K_ESCAPE:
                self.stop(aborted=True)
                return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.next()
                return True
            if event.button in (3, 4):  # right-click or wheel back for previous
                self.prev()
                return True
        return True

    # ------------------------------------------------------------------ Drawing
    def draw(self) -> None:
        if not self.active or not self.steps:
            return
        screen = self.app.screen
        width, height = screen.get_size()
        scale = calc_scale((width, height), min_scale=0.7, max_scale=1.25)
        if abs(scale - self._scale) > 1e-3:
            self._refresh_fonts(scale)
        step = self.steps[self.index]
        rects = self._resolve_rects(step)
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill(self.overlay_color)
        for rect in rects:
            pygame.draw.rect(overlay, (0, 0, 0, 0), rect, border_radius=step.get('radius', 16))
            pygame.draw.rect(overlay, self.accent_soft, rect, width=2, border_radius=step.get('radius', 16))
        screen.blit(overlay, (0, 0))
        self._draw_callout(step, rects)

    # ------------------------------------------------------------------ Helpers
    def _resolve_rects(self, step: dict) -> list[pygame.Rect]:
        rects: list[pygame.Rect] = []
        targets = step.get('rects')
        if targets is None and 'rect' in step:
            targets = [step['rect']]
        if not targets:
            return rects
        padding = step.get('padding', int(18 * self._scale))
        for target in targets:
            rect = target() if callable(target) else target
            if rect is None:
                continue
            rect = pygame.Rect(rect)
            rect = rect.inflate(padding * 2, padding * 2)
            rects.append(rect)
        return rects

    def _draw_callout(self, step: dict, rects: list[pygame.Rect]) -> None:
        screen = self.app.screen
        width, height = screen.get_size()
        title = step.get('title', '')
        body = step.get('body', '')
        placement = step.get('placement', 'right')
        max_width = min(int(520 * self._scale), int(width * 0.48))
        card_padding = max(14, int(18 * self._scale))
        text_color = (235, 240, 252)
        card_color = (20, 26, 46, 235)

        title_surface = self.title_font.render(title, True, self.accent_soft)
        body_surfaces = self._wrap_text(body, self.body_font, text_color, max_width - card_padding * 2)
        hint_text = self._hint_text or step.get(
            'hint',
            'ЛКМ — дальше, ПКМ/Backspace — назад, Esc — пропустить',
        )
        hint_surface = self.hint_font.render(hint_text, True, (180, 192, 212))

        content_width = max(title_surface.get_width(), *(surf.get_width() for surf in body_surfaces), hint_surface.get_width())
        content_width = min(max_width - card_padding * 2, content_width)
        card_width = max(content_width + card_padding * 2, max_width // 2)
        card_height = title_surface.get_height() + card_padding
        for surf in body_surfaces:
            card_height += surf.get_height() + 6
        card_height += hint_surface.get_height() + card_padding * 2

        target_rect = rects[0] if rects else pygame.Rect(width // 2, height // 2, 1, 1)
        card_rect = pygame.Rect(0, 0, card_width, card_height)
        card_rect = self._place_callout(card_rect, target_rect, placement, width, height)

        card_surface = pygame.Surface((card_rect.width, card_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(card_surface, card_color, card_surface.get_rect(), border_radius=16)
        pygame.draw.rect(card_surface, self.accent_soft, card_surface.get_rect(), width=2, border_radius=16)

        y = card_padding
        card_surface.blit(title_surface, (card_padding, y))
        y += title_surface.get_height() + 10
        for surf in body_surfaces:
            card_surface.blit(surf, (card_padding, y))
            y += surf.get_height() + 6
        y += 12
        card_surface.blit(hint_surface, (card_padding, y))

        screen.blit(card_surface, card_rect.topleft)

    def _place_callout(
        self, card_rect: pygame.Rect, target_rect: pygame.Rect, placement: str, width: int, height: int
    ) -> pygame.Rect:
        card = card_rect.copy()
        gap = max(12, int(18 * self._scale))
        if placement == 'left':
            card.right = max(gap, target_rect.left - gap)
            card.centery = target_rect.centery
        elif placement == 'top':
            card.centerx = target_rect.centerx
            card.bottom = max(gap, target_rect.top - gap)
        elif placement == 'bottom':
            card.centerx = target_rect.centerx
            card.top = min(height - card.height - gap, target_rect.bottom + gap)
        else:  # default right
            card.left = min(width - card.width - gap, target_rect.right + gap)
            card.centery = target_rect.centery

        card.left = max(gap, min(card.left, width - card.width - gap))
        card.top = max(gap, min(card.top, height - card.height - gap))
        return card

    def _wrap_text(
        self, text: str, font: pygame.font.Font, color: tuple[int, int, int], max_width: int
    ) -> list[pygame.Surface]:
        words = text.replace('\n', ' \n ').split(' ')
        lines: list[str] = []
        current = ""
        for word in words:
            if word == "\n":
                lines.append(current.rstrip())
                current = ""
                continue
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            lines = [""]
        return [font.render(line, True, color) for line in lines]

    def _refresh_fonts(self, scale: float) -> None:
        self._scale = scale
        title_size = max(18, int(26 * scale))
        body_size = max(14, int(18 * scale))
        hint_size = max(12, int(16 * scale))
        self.title_font = get_font(title_size, bold=True)
        self.body_font = get_font(body_size)
        self.hint_font = get_font(hint_size)
