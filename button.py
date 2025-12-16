import pygame.font
import pygame

class Button():
    
    def __init__(self, app, msg, position, button_size, command = lambda: print('no action for button'), **kwargs):
        """Инициализирует атрибуты кнопки."""
        self.screen = app.screen
        self.screen_rect = self.screen.get_rect()
        self.font = kwargs.get('font', 'corbel')
        # Назначение размеров и свойств кнопок.
        self.width, self.height = button_size
        self.command = command
        self.button_color = kwargs.get('button_color', (240, 240, 240))
        self.text_color = kwargs.get('text_color', (0, 0, 0))
        self.font = pygame.font.SysFont(self.font, kwargs.get('fontSize', 36), kwargs.get('bold', True))
        self.border_radius = kwargs.get('border_radius', 10)
        self.border_color = kwargs.get('border_color')
        self.shadow_offset = kwargs.get('shadow_offset', 0)
        self.shadow_color = kwargs.get('shadow_color', (0, 0, 0, 80))

        # Построение объекта rect кнопки и выравнивание по центру экрана.
        self.rect = pygame.Rect(*position, self.width, self.height)
        
        # Сообщение кнопки создается только один раз.
        self._prep_msg(msg)
    
    def _prep_msg(self, msg):
        self.msg_image = self.font.render(msg, True, self.text_color,
                                            self.button_color)
        self.msg_image_rect = self.msg_image.get_rect()
        self.msg_image_rect.center = self.rect.center
    
    def draw_button(self):
        # Отображение пустой кнопки и вывод сообщения.
        if self.shadow_offset:
            shadow_rect = self.rect.copy()
            shadow_rect.x += self.shadow_offset
            shadow_rect.y += self.shadow_offset
            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            shadow_color = self.shadow_color if len(self.shadow_color) == 4 else (*self.shadow_color, 80)
            pygame.draw.rect(
                shadow_surface,
                shadow_color,
                shadow_surface.get_rect(),
                border_radius=self.border_radius,
            )
            self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.button_color, self.rect, border_radius=self.border_radius)
        if self.border_color:
            pygame.draw.rect(self.screen, self.border_color, self.rect, width=2, border_radius=self.border_radius)
        self.screen.blit(self.msg_image, self.msg_image_rect)
