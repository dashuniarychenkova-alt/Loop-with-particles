class ResponsiveScreen:
    """Keeps pygame Surface references in sync with the resizable window."""

    def __init__(self, app):
        self.app = app
        self.screen = app.screen

    def on_window_resize(self, size: tuple[int, int]) -> None:
        """Update cached Surface reference and notify subclasses to relayout."""
        self.screen = self.app.screen
        relayout = getattr(self, "_relayout", None)
        if callable(relayout):
            relayout(size)


def get_font(size: int, *, bold: bool = False) -> "pygame.font.Font":
    """Try to obtain a clean modern system font with sensible fallbacks."""
    import pygame

    families = [
        "SF Pro Display",
        "Helvetica Neue",
        "Avenir Next",
        "Segoe UI",
        "Roboto",
        "Arial",
        "sans-serif",
    ]
    return pygame.font.SysFont(families, size, bold=bold)


def calc_scale(
    size: tuple[int, int],
    *,
    base: tuple[int, int] = (1600, 900),
    min_scale: float = 0.5,
    max_scale: float = 1.25,
) -> float:
    """Return a clamped UI scale factor for the given window size."""
    if not size or base[0] <= 0 or base[1] <= 0:
        return 1.0
    width, height = size
    base_w, base_h = base
    raw = min(width / base_w, height / base_h)
    return max(min_scale, min(max_scale, raw))


def build_vertical_gradient(
    size: tuple[int, int], top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]
) -> "pygame.Surface":
    """Create a vertical gradient surface for backgrounds."""
    import pygame

    width, height = size
    surface = pygame.Surface((max(width, 1), max(height, 1)))
    if height <= 1:
        surface.fill(top_color)
        return surface
    for y in range(height):
        ratio = y / (height - 1)
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * ratio) for i in range(3))
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface
