import sys
from typing import List, Tuple

import pygame

import language
from button import Button
from onboarding import OnboardingGuide
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient
from paths import resource_file


class MenuScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()
        self.assets_dir = "_internal/images"

        self.primary_color = (72, 104, 255)
        self.muted_text = (85, 95, 120)
        self.text_color = (35, 38, 46)

        self.background: pygame.Surface | None = None
        self.header_rect: pygame.Rect | None = None
        self.header_surfaces: List[Tuple[pygame.Surface, pygame.Rect]] = []
        self.card_rect: pygame.Rect | None = None
        self.buttons: List[Button] = []
        self.lang_button: Button | None = None

        self.logo_left_raw = pygame.image.load(resource_file(f"{self.assets_dir}/msu_logo.jpg")).convert()
        self.logo_right_raw = pygame.image.load(resource_file(f"{self.assets_dir}/ff_sign.png")).convert()
        self.logo_left_surface = self.logo_left_raw
        self.logo_right_surface = self.logo_right_raw
        self.logo_left_pos = (0, 0)
        self.logo_right_pos = (0, 0)

        self.button_specs = [
            {'label_key': 'btn_demo', 'command': self.to_demo, 'primary': True},
            {'label_key': 'btn_tutorial', 'command': self.to_tutorial, 'primary': False, 'disabled': True},
            {'label_key': 'btn_theory', 'command': self.to_theory, 'primary': False, 'disabled': True},
            {'label_key': 'btn_authors', 'command': self.to_authors, 'primary': False, 'disabled': True},
            {'label_key': 'btn_exit', 'command': self.quite_demo, 'primary': False},
        ]

        self.onboarding = OnboardingGuide(app)
        self._onboarding_started = False

        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self._relayout(self.app.window_size)

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        width = max(width, 900)
        height = max(height, 650)

        self.background = build_vertical_gradient((width, height), (232, 239, 255), (247, 248, 254))
        outer_padding = max(36, width // 20)

        header_top = max(24, int(height * 0.04))
        min_header_height = 170
        base_header_height = max(200, int(height * 0.24))
        max_header_height = int(height * 0.35)
        header_height = max(min_header_height, min(base_header_height, max_header_height))
        header_width = width - 2 * outer_padding
        gap_after_header = max(28, int(height * 0.035))
        button_count = len(self.button_specs)

        # Button/card sizing with sensible fallbacks for short viewports.
        card_width = max(420, min(int(width * 0.44), header_width))
        base_button_height = max(60, min(78, int(height * 0.08)))
        base_button_gap = max(18, base_button_height // 2)
        base_inner_padding = max(28, card_width // 10)
        min_button_height = 48
        min_button_gap = 12
        min_inner_padding = 20

        self.header_rect = pygame.Rect(outer_padding, header_top, header_width, header_height)

        min_card_height = (
            button_count * min_button_height + (button_count - 1) * min_button_gap + min_inner_padding * 2
        )
        available_for_card = height - outer_padding - (self.header_rect.bottom + gap_after_header)
        if available_for_card < min_card_height:
            deficit = min_card_height - available_for_card
            shrinkable = max(0, self.header_rect.height - min_header_height)
            shrink = min(deficit, shrinkable)
            if shrink:
                self.header_rect.height -= shrink
            available_for_card = height - outer_padding - (self.header_rect.bottom + gap_after_header)
            if available_for_card < min_card_height:
                gap_shrink = min(gap_after_header - 18, min_card_height - available_for_card)
                if gap_shrink > 0:
                    gap_after_header -= gap_shrink
                    available_for_card = height - outer_padding - (self.header_rect.bottom + gap_after_header)

        buttons_block_height = button_count * base_button_height + (button_count - 1) * base_button_gap
        card_height = buttons_block_height + base_inner_padding * 2
        scale = 1.0
        if card_height > available_for_card and available_for_card > 0:
            scale = max(0.68, min(1.0, available_for_card / card_height))
        button_height = max(min_button_height, int(base_button_height * scale))
        button_gap = max(min_button_gap, int(base_button_gap * scale))
        inner_padding = max(min_inner_padding, int(base_inner_padding * scale))
        buttons_block_height = button_count * button_height + (button_count - 1) * button_gap
        card_height = buttons_block_height + inner_padding * 2

        card_left = (width - card_width) // 2
        card_top = self.header_rect.bottom + gap_after_header
        max_card_height = height - outer_padding - card_top
        allowed_height = max_card_height
        total_needed = buttons_block_height + inner_padding * 2
        if allowed_height > 0 and total_needed > allowed_height:
            tight_scale = max(0.55, min(1.0, allowed_height / float(total_needed)))
            button_height = max(min_button_height, int(button_height * tight_scale))
            button_gap = max(min_button_gap, int(button_gap * tight_scale))
            inner_padding = max(min_inner_padding, int(inner_padding * tight_scale))
            buttons_block_height = button_count * button_height + (button_count - 1) * button_gap
            total_needed = buttons_block_height + inner_padding * 2
        card_height = min(total_needed, max_card_height) if max_card_height > 0 else total_needed
        self.card_rect = pygame.Rect(card_left, int(card_top), card_width, int(card_height))

        logo_limit = max(90, self.header_rect.height - 24)
        logo_size = max(90, min(170, self.header_rect.width // 6, logo_limit))
        self._build_header(int(logo_size))

        lang_label = "EN" if self.lang.lang == "rus" else "RU"
        lang_size = max(48, min(64, logo_size // 1.6))
        lang_x = self.header_rect.right - lang_size - max(12, outer_padding // 4)
        lang_y = self.header_rect.top + max(12, outer_padding // 4)
        self.lang_button = Button(
            self.app,
            lang_label,
            (lang_x, lang_y),
            (lang_size, lang_size),
            self.lang_change,
            font=["SF Pro Display", "Segoe UI", "Arial"],
            fontSize=max(20, int(lang_size * 0.4)),
            bold=True,
            button_color=self.primary_color,
            text_color=(255, 255, 255),
            border_radius=lang_size // 2,
            shadow_offset=6,
            shadow_color=(64, 99, 255, 110),
        )

        self._build_buttons(button_height, button_gap, inner_padding)

    def _build_header(self, logo_size: int) -> None:
        assert self.header_rect is not None
        header = self.header_rect

        self.logo_left_surface = pygame.transform.smoothscale(self.logo_left_raw, (logo_size, logo_size))
        self.logo_right_surface = pygame.transform.smoothscale(self.logo_right_raw, (logo_size, logo_size))

        vertical_center = header.centery
        side_margin = max(16, header.width // 40)
        self.logo_left_pos = (header.left + side_margin, vertical_center - logo_size // 2)
        self.logo_right_pos = (header.right - side_margin - logo_size, vertical_center - logo_size // 2)

        text_margin = max(24, int(header.width * 0.04))
        text_left = self.logo_left_pos[0] + logo_size + text_margin
        text_right = self.logo_right_pos[0] - text_margin
        text_center = (text_left + text_right) / 2
        text_width = max(200, text_right - text_left)

        gap_unit = max(6, int(header.height * 0.022))
        specs = [
            {'text': self.lang['university_name'], 'size': max(28, int(header.height * 0.13)), 'bold': True, 'color': self.text_color, 'gap': gap_unit},
            {'text': self.lang['faculty_name'], 'size': max(22, int(header.height * 0.1)), 'bold': True, 'color': self.text_color, 'gap': gap_unit * 2},
            {'text': self.lang['comp_demo'], 'size': max(20, int(header.height * 0.085)), 'bold': False, 'color': self.text_color, 'gap': gap_unit},
            {'text': self.lang['subject_name'], 'size': max(20, int(header.height * 0.085)), 'bold': False, 'color': self.text_color, 'gap': gap_unit * 2},
            {'text': self.lang['project_title'], 'size': max(24, int(header.height * 0.11)), 'bold': True, 'color': self.text_color, 'gap': gap_unit * 2},
        ]

        rendered: List[Tuple[pygame.Surface, int]] = []
        for spec in specs:
            surface = self._render_line_to_fit(
                spec['text'],
                spec['size'],
                spec['bold'],
                spec['color'],
                header.height,
                max_width=text_width,
                scale_down=True,
            )
            rendered.append((surface, spec['gap']))

        total_height = sum(surface.get_height() for surface, _ in rendered)
        total_height += sum(gap for _, gap in rendered[:-1])
        start_y = header.top + max(10, (header.height - total_height) // 2)

        self.header_surfaces = []
        current_y = start_y
        for surface, gap in rendered:
            rect = surface.get_rect()
            rect.centerx = int(text_center)
            rect.y = int(round(current_y))
            self.header_surfaces.append((surface, rect))
            current_y = rect.bottom + gap

    def _build_buttons(self, button_height: int, button_gap: int, inner_padding: int) -> None:
        assert self.card_rect is not None
        card = self.card_rect

        self.buttons = []
        for index, spec in enumerate(self.button_specs):
            button_x = card.left + inner_padding
            button_y = card.top + inner_padding + index * (button_height + button_gap)
            is_primary = spec.get('primary', False)
            label = self.lang[spec['label_key']]
            is_disabled = spec.get('disabled', False)
            button_color = self.primary_color if is_primary else (249, 250, 253)
            text_color = (255, 255, 255) if is_primary else self.text_color
            border_color = None if is_primary else (212, 217, 230)
            shadow_offset = 9 if is_primary else 5
            shadow_color = (64, 99, 255, 110) if is_primary else (0, 0, 0, 45)
            command = spec['command']
            if is_disabled:
                button_color = (235, 236, 241)
                text_color = self.muted_text
                border_color = (215, 220, 235)
                shadow_offset = 0
                shadow_color = (0, 0, 0, 0)
                command = self._noop
            button = Button(
                self.app,
                label,
                (int(button_x), int(button_y)),
                (card.width - inner_padding * 2, button_height),
                command,
                font=["SF Pro Display", "Segoe UI", "Arial"],
                fontSize=max(22, int(button_height * 0.42)),
                bold=True,
                button_color=button_color,
                text_color=text_color,
                border_radius=18,
                border_color=border_color,
                shadow_offset=shadow_offset,
                shadow_color=shadow_color,
            )
            button.label_key = spec.get('label_key', '')
            self.buttons.append(button)

    def _render_line_to_fit(
        self,
        text: str,
        base_size: int,
        bold: bool,
        color: Tuple[int, int, int],
        max_height: int,
        *,
        max_width: int | None = None,
        scale_down: bool = False,
    ) -> pygame.Surface:
        size = max(base_size, 18)
        last_surface = None
        while size >= 18:
            font = get_font(size, bold=bold)
            surface = font.render(text, True, color)
            fits_height = surface.get_height() <= max_height
            fits_width = True if max_width is None else surface.get_width() <= max_width
            if (fits_height and fits_width) or size <= 18:
                return surface
            last_surface = surface
            size -= 2 if scale_down else 1
        return last_surface if last_surface is not None else get_font(18, bold=bold).render(text, True, color)

    # ---------------------------------------------------------------- Buttons
# -------- Buttons
    def to_demo(self):
        self.app.active_screen = self.app.demo_screen

    def _disabled_feature(self):
        # Кнопка отключена — просто ничего не делаем
        pass

    def to_theory(self):
        self.app.active_screen = self.app.theory_screen

    def to_authors(self):
        self.app.active_screen = self.app.authors_screen

    def to_tutorial(self):
        self.app.active_screen = self.app.tutorial_screen

    def quite_demo(self):
        pygame.quit()
        sys.exit()

    def lang_change(self):
        self.app.toggle_language()

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        assert self.card_rect is not None
        assert self.background is not None

        self.screen.blit(self.background, (0, 0))
        self._draw_header()
        self._draw_card(self.card_rect)

        for button in self.buttons:
            button.draw_button()

        if self.lang_button:
            self.lang_button.draw_button()
        self._ensure_onboarding()
        if self.onboarding.active:
            self.onboarding.draw()

    def _draw_header(self) -> None:
        assert self.header_rect is not None
        header = self.header_rect
        shadow_rect = header.copy()
        shadow_rect.x += 10
        shadow_rect.y += 12
        shadow_surface = pygame.Surface(header.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (15, 22, 58, 50), shadow_surface.get_rect(), border_radius=32)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), header, border_radius=30)
        pygame.draw.rect(self.screen, (215, 220, 235), header, width=2, border_radius=30)

        self.screen.blit(self.logo_left_surface, self.logo_left_pos)
        self.screen.blit(self.logo_right_surface, self.logo_right_pos)
        for surface, rect in self.header_surfaces:
            self.screen.blit(surface, rect)

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
            elif self.onboarding.handle_event(event):
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)

    def _check_buttons(self, mouse_position):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_position):
                button.command()
                return
        if self.lang_button and self.lang_button.rect.collidepoint(mouse_position):
            self.lang_button.command()

    def _noop(self):
        """Placeholder command for disabled buttons."""
        return None

    # ---------------------------------------------------------------- Onboarding
    def _ensure_onboarding(self) -> None:
        if not getattr(self.app, 'onboarding_enabled', True):
            return
        if getattr(self.app, 'onboarding_menu_done', False):
            return
        if self._onboarding_started:
            return
        self._onboarding_started = True
        steps = self._build_onboarding_steps()

        def _finish(aborted: bool) -> None:
            self.app.onboarding_menu_done = True
            if not aborted:
                self.app.onboarding_demo_pending = True
                self.app.active_screen = self.app.demo_screen
                if hasattr(self.app.demo_screen, 'queue_onboarding_from_menu'):
                    self.app.demo_screen.queue_onboarding_from_menu()

        self.onboarding.start(steps, on_complete=_finish, hint=self._nav_hint_text())

    def _nav_hint_text(self) -> str:
        return (
            "ЛКМ — дальше, ПКМ/Backspace — назад, Esc — пропустить"
            if self.lang.lang == "rus"
            else "LMB — next, RMB/Backspace — back, Esc — skip"
        )

    def _build_onboarding_steps(self) -> list[dict]:
        intro_title = "Что здесь?" if self.lang.lang == "rus" else "What’s here?"
        intro_body = (
            "Главное меню: выбирайте разделы, чтобы понять тепловой градиент и симуляцию."
            if self.lang.lang == "rus"
            else "Main hub: pick a section to explore the temperature-gradient demo."
        )
        steps = [
            {
                'title': intro_title,
                'body': intro_body,
                'rect': lambda: self.card_rect,
                'placement': 'right',
                'padding': 24,
            }
        ]
        order = ['btn_exit', 'btn_authors', 'btn_theory', 'btn_tutorial', 'btn_demo']
        texts = {
            'btn_exit': (
                "Выход: завершите сессию, когда всё посмотрели."
                if self.lang.lang == "rus"
                else "Exit: close the app when you’re done."
            ),
            'btn_authors': (
                "Авторы: кто делал проект и как с нами связаться."
                if self.lang.lang == "rus"
                else "Authors: who built this and how to reach us."
            ),
            'btn_theory': (
                "Теория: формулы и физика теплового потока."
                if self.lang.lang == "rus"
                else "Theory: the physics behind the heat flow."
            ),
            'btn_tutorial': (
                "Обучение: краткое текстовое описание — пригодится, если хочется перечитать."
                if self.lang.lang == "rus"
                else "Tutorial: a quick text guide when you need a refresher."
            ),
            'btn_demo': (
                "Демонстрация: интерактивная симуляция. Пойдём туда последними."
                if self.lang.lang == "rus"
                else "Demo: the interactive simulation. We’ll open it last."
            ),
        }
        placements = {
            'btn_exit': 'left',
            'btn_authors': 'left',
            'btn_theory': 'right',
            'btn_tutorial': 'right',
            'btn_demo': 'right',
        }
        for key in order:
            try:
                title_text = self.lang[key]
            except KeyError:
                title_text = key
            steps.append(
                {
                    'title': title_text,
                    'body': texts[key],
                    'rect': lambda k=key: self._button_rect(k),
                    'placement': placements.get(key, 'right'),
                }
            )
        return steps

    def _button_rect(self, label_key: str):
        for button in self.buttons:
            if getattr(button, 'label_key', None) == label_key:
                return button.rect
        return None
