from __future__ import annotations

from typing import List, Tuple

import pygame

import language
from button import Button
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient
from paths import resource_file


class AuthorsScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()
        self.assets_dir = "_internal/images"

        self.primary_color = (72, 104, 255)
        self.text_color = (33, 36, 44)
        self.muted_text = (96, 105, 124)

        self.background: pygame.Surface | None = None
        self.header_surfaces: List[Tuple[pygame.Surface, pygame.Rect]] = []
        self.author_cards: List[dict] = []
        self.menu_button: Button | None = None

        self.logo_raw = pygame.image.load(resource_file(f"{self.assets_dir}/msu_logo.jpg")).convert()
        self.author_images = {
            'author_daria': pygame.image.load(resource_file(f"{self.assets_dir}/daria_photo.jpg")).convert(),
        }

        self.authors = self._collect_authors()
        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self.authors = self._collect_authors()
        self._relayout(self.app.window_size)

    def _collect_authors(self) -> List[dict]:
        keys = sorted(key for key in self.lang._loader.keys() if key.startswith('author'))
        if not keys:
            keys = ['author_daria']
        authors = []
        for key in keys:
            name = self.lang[key] if key in self.lang._loader else key.replace('author_', '').title()
            image = self.author_images.get(key)
            authors.append({'name': name, 'image_key': key, 'image': image})
        return authors

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        width = max(width, 900)
        height = max(height, 650)
        self.background = build_vertical_gradient(size, (235, 240, 255), (249, 250, 254))

        self._build_header(width, height)
        self._build_cards(width, height)
        self._build_menu_button(width, height)

    def _build_header(self, width: int, height: int) -> None:
        outer_padding = max(48, width // 18)
        top = outer_padding
        lines = [
            (self.lang['university_name'], max(26, width // 34), True, self.primary_color),
            (self.lang['faculty_name'], max(22, width // 38), False, self.text_color),
            (self.lang['lecturer'], max(20, width // 42), False, self.muted_text),
            (self.lang['supervisor'], max(20, width // 42), False, self.muted_text),
        ]
        current_y = top
        self.header_surfaces = []

        for text, size, bold, color in lines:
            font = get_font(size, bold=bold)
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            rect.x = outer_padding
            rect.y = current_y
            self.header_surfaces.append((surface, rect))
            current_y = rect.bottom + max(14, size // 4)

        logo_size = max(120, min(180, width // 8))
        self.logo_surface = pygame.transform.smoothscale(self.logo_raw, (logo_size, logo_size))
        self.logo_position = (width - outer_padding - logo_size, outer_padding)
        self.cards_top = current_y + max(30, logo_size // 6)

    def _build_cards(self, width: int, height: int) -> None:
        outer_padding = max(48, width // 18)
        gap = max(28, width // 40)
        max_columns = 3 if len(self.authors) >= 3 and width > 1600 else 2
        columns = min(max_columns, max(1, len(self.authors)))
        cards_area_width = width - outer_padding * 2
        card_width = (cards_area_width - gap * (columns - 1)) // columns
        card_start_x = outer_padding
        if columns == 1:
            single_card_limit = min(cards_area_width, max(520, cards_area_width // 2))
            if card_width > single_card_limit:
                card_width = single_card_limit
            card_start_x = outer_padding + max(0, (cards_area_width - card_width) // 2)
        card_height = max(380, int(card_width * 1.1))

        available_height = height - self.cards_top - outer_padding - 120
        if card_height > available_height:
            card_height = max(320, available_height)

        self.author_cards = []
        for index, author in enumerate(self.authors):
            column = index % columns
            row = index // columns
            if columns == 1:
                card_x = card_start_x
            else:
                card_x = outer_padding + column * (card_width + gap)
            card_y = self.cards_top + row * (card_height + gap)
            rect = pygame.Rect(card_x, card_y, card_width, card_height)

            card = {'rect': rect, 'photo': None, 'texts': []}
            inner_padding = max(28, card_width // 12)
            photo_size = min(card_width - inner_padding * 2, int(card_height * 0.55))
            if photo_size > 0:
                photo_surface = self._build_author_photo(author, photo_size)
                photo_rect = photo_surface.get_rect()
                photo_rect.centerx = rect.centerx
                photo_rect.y = rect.y + inner_padding
                card['photo'] = (photo_surface, photo_rect)
                text_top = photo_rect.bottom + inner_padding // 2
            else:
                text_top = rect.y + inner_padding

            name_font_size = max(24, min(64, int(card_width * 0.08)))
            name_font = get_font(name_font_size, bold=True)
            name_surface = name_font.render(author['name'], True, self.text_color)
            name_rect = name_surface.get_rect()
            name_rect.centerx = rect.centerx
            name_rect.y = text_top
            card['texts'].append((name_surface, name_rect))

            tagline = 'Автор' if self.lang.lang == 'rus' else 'Author'
            tagline_font_size = max(18, min(30, int(card_width * 0.05)))
            tagline_font = get_font(tagline_font_size)
            tagline_surface = tagline_font.render(tagline, True, self.muted_text)
            tagline_rect = tagline_surface.get_rect()
            tagline_rect.centerx = rect.centerx
            tagline_rect.y = name_rect.bottom + 6
            card['texts'].append((tagline_surface, tagline_rect))

            self.author_cards.append(card)

        if self.author_cards:
            last_card = self.author_cards[-1]['rect']
            self.cards_bottom = last_card.bottom
        else:
            self.cards_bottom = self.cards_top

    def _build_author_photo(self, author: dict, size: int) -> pygame.Surface:
        if author['image'] is not None:
            return pygame.transform.smoothscale(author['image'], (size, size))

        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surface, (240, 243, 255), (0, 0, size, size), border_radius=size // 5)
        pygame.draw.rect(surface, (215, 222, 245), (0, 0, size, size), width=2, border_radius=size // 5)
        initials = ''.join(part[0] for part in author['name'].split()[:2]).upper()
        font = get_font(max(24, size // 3), bold=True)
        text_surface = font.render(initials, True, (72, 104, 255))
        text_rect = text_surface.get_rect(center=(size // 2, size // 2))
        surface.blit(text_surface, text_rect)
        return surface

    def _build_menu_button(self, width: int, height: int) -> None:
        outer_padding = max(48, width // 18)
        button_width = min(320, max(240, width // 5))
        button_height = 64
        button_x = (width - button_width) // 2
        button_y = height - outer_padding - button_height
        self.menu_button = Button(
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

    # ---------------------------------------------------------------- Events
    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                if self.menu_button and self.menu_button.rect.collidepoint(mouse_position):
                    self.menu_button.command()

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        self.screen.blit(self.background, (0, 0))
        for surface, rect in self.header_surfaces:
            self.screen.blit(surface, rect)
        self.screen.blit(self.logo_surface, self.logo_position)

        for card in self.author_cards:
            self._draw_card(card)

        if self.menu_button:
            self.menu_button.draw_button()

    def _draw_card(self, card: dict) -> None:
        rect: pygame.Rect = card['rect']
        shadow_rect = rect.copy()
        shadow_rect.x += 10
        shadow_rect.y += 14
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (18, 25, 64, 55), shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=20)
        pygame.draw.rect(self.screen, (216, 221, 236), rect, width=2, border_radius=20)

        if card['photo']:
            surface, photo_rect = card['photo']
            self.screen.blit(surface, photo_rect)
        for surface, text_rect in card['texts']:
            self.screen.blit(surface, text_rect)
