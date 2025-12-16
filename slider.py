from __future__ import annotations

import pygame

from ui_base import get_font


class ParamSlider:
    """Slider control rendered as a compact card with label and value."""

    def __init__(
        self,
        app,
        name: str,
        rect: tuple[int, int, int, int],
        bounds: tuple[float, float],
        step: float,
        name_par: str,
        dec_number: int,
        initial_pos: float,
        **kwargs,
    ):
        self.app = app
        self.screen = app.screen
        self.name = name
        self.name_par = name_par
        self.min_val, self.max_val = bounds
        self.step = step
        self.decimals = dec_number
        self.dec_round = (
            (lambda x: int(round(x, 0)))
            if dec_number == 0
            else (lambda x: round(x, dec_number))
        )
        self.value_suffix = kwargs.get('value_suffix', '')

        self.card_rect = pygame.Rect(rect)
        self.padding = kwargs.get('padding', 18)
        self.card_color = kwargs.get('card_color', (255, 255, 255))
        self.card_border = kwargs.get('card_border', (214, 220, 235))
        self.shadow_color = kwargs.get('shadow_color', (15, 22, 58, 45))
        self.label_font = get_font(kwargs.get('label_size', 18), bold=True)
        self.value_font = get_font(kwargs.get('value_size', 20), bold=True)

        self.track_height = kwargs.get('track_height', 8)
        track_left = self.card_rect.left + self.padding
        track_right = self.card_rect.right - self.padding
        track_width = max(1, track_right - track_left)
        track_y = self.card_rect.bottom - self.padding - self.track_height // 2
        self.track_rect = pygame.Rect(
            track_left,
            track_y - self.track_height // 2,
            track_width,
            self.track_height,
        )
        self.track_color = kwargs.get('track_color', (215, 219, 232))
        self.fill_color = kwargs.get('fill_color', (72, 104, 255))
        self.knob_color = kwargs.get('knob_color', (72, 104, 255))
        self.knob_hover_color = kwargs.get('knob_hover_color', (52, 82, 230))

        initial_value = self._value_from_ratio(initial_pos)
        self.slider = _SliderImpl(self, initial_value)

    # ------------------------------------------------------------------ Helpers
    def _value_from_ratio(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return self.min_val + ratio * (self.max_val - self.min_val)

    def _snap_value(self, value: float) -> float:
        snapped = value
        if self.step and self.step > 0:
            steps = round((snapped - self.min_val) / self.step)
            snapped = self.min_val + steps * self.step
        return max(self.min_val, min(self.max_val, snapped))

    # ------------------------------------------------------------------ Drawing
    def draw_check(self, params: dict) -> None:
        """Render the card and update ``params`` with the current value."""
        self.screen = self.app.screen
        self._draw_card_background()
        raw_value = self.slider.get_value()
        value = self._snap_value(raw_value)
        if abs(value - raw_value) > 1e-9:
            self.slider.set_value(value)
        display_value = self.dec_round(value)
        params[self.name_par] = display_value
        self._draw_labels(display_value)
        self.slider.draw(self.screen)

    def _draw_card_background(self) -> None:
        shadow_rect = self.card_rect.copy()
        shadow_rect.x += 6
        shadow_rect.y += 8
        shadow_surface = pygame.Surface(self.card_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            shadow_surface,
            self.shadow_color,
            shadow_surface.get_rect(),
            border_radius=16,
        )
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.card_color, self.card_rect, border_radius=14)
        pygame.draw.rect(self.screen, self.card_border, self.card_rect, width=2, border_radius=14)

    def _draw_labels(self, value: float) -> None:
        label_surface = self.label_font.render(self.name, True, (58, 64, 82))
        label_rect = label_surface.get_rect()
        label_rect.topleft = (self.card_rect.left + self.padding, self.card_rect.top + self.padding)
        self.screen.blit(label_surface, label_rect)

        if self.decimals == 0:
            value_text = f"{int(value)}"
        else:
            value_text = f"{value:.{self.decimals}f}"
        if self.value_suffix:
            value_text = f"{value_text} {self.value_suffix}"
        value_surface = self.value_font.render(value_text, True, (27, 32, 42))
        value_rect = value_surface.get_rect()
        value_rect.top = label_rect.bottom + max(4, self.padding // 2)
        value_rect.left = self.card_rect.left + self.padding
        self.screen.blit(value_surface, value_rect)

    # ------------------------------------------------------------------ API
    def getValue(self) -> float:
        raw_value = self.slider.get_value()
        value = self._snap_value(raw_value)
        if abs(value - raw_value) > 1e-9:
            self.slider.set_value(value)
        return self.dec_round(value)

    def set_value(self, value: float) -> None:
        self.slider.set_value(value)

    def set_label(self, name: str) -> None:
        self.name = name


class _SliderImpl:
    def __init__(self, owner: ParamSlider, initial_value: float) -> None:
        self.owner = owner
        self.hovered = False
        self.grabbed = False
        self.screen = owner.screen
        self.min = owner.min_val
        self.max = owner.max_val
        self.track = owner.track_rect
        self.knob_radius = max(10, owner.track_height * 2)
        diameter = self.knob_radius * 2
        self.button_rect = pygame.Rect(0, 0, diameter, diameter)
        self.set_value(initial_value)

    def draw(self, surface: pygame.Surface) -> None:
        track = self.track
        center = (track.left, track.centery)
        pygame.draw.line(surface, self.owner.track_color, center, (track.right, track.centery), track.height)
        pygame.draw.line(
            surface,
            self.owner.fill_color,
            center,
            (self.button_rect.centerx, track.centery),
            track.height,
        )
        knob_color = self.owner.knob_hover_color if (self.hovered or self.grabbed) else self.owner.knob_color
        shadow_surface = pygame.Surface((self.knob_radius * 2, self.knob_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            shadow_surface,
            (0, 0, 0, 55),
            (self.knob_radius, self.knob_radius),
            self.knob_radius,
        )
        surface.blit(
            shadow_surface,
            (self.button_rect.centerx - self.knob_radius, self.button_rect.centery - self.knob_radius + 3),
        )
        pygame.draw.circle(surface, knob_color, self.button_rect.center, self.knob_radius)

    def move_slider(self, mouse_pos: tuple[int, int]) -> None:
        pos = mouse_pos[0]
        if pos < self.track.left:
            pos = self.track.left
        if pos > self.track.right:
            pos = self.track.right
        self.button_rect.centerx = pos
        self.button_rect.centery = self.track.centery

    def get_value(self) -> float:
        val_range = max(1, self.track.width)
        button_val = self.button_rect.centerx - self.track.left
        ratio = button_val / val_range
        return self.min + ratio * (self.max - self.min)

    def set_value(self, value: float) -> None:
        if self.max <= self.min:
            self.button_rect.centerx = self.track.left
            return
        clamped = max(self.min, min(self.max, value))
        ratio = (clamped - self.min) / (self.max - self.min)
        self.button_rect.centerx = int(round(self.track.left + ratio * self.track.width))
        self.button_rect.centery = self.track.centery
