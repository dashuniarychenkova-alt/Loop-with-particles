import pygame
import pygame_chart as pyc
import gc
# from simulation import expected_potential_energy, expected_kinetic_energy


class ListBuff:
    def __init__(self, size):
        self.size = size
        self.buff = [0] * size
        self.main = []
        self.ind = 0
    
    def append(self, other):
        self.buff[self.ind] = other
        self.ind += 1
        if self.ind == self.size:
            self.main.append(sum(self.buff) / self.size)
            self.ind = 0
    
    def extend(self, other):
        for item in other:
            # -1 means end of the buffer
            if item == -1:
                break
            self.append(item)
    
    def refresh(self):
        self.main = []
        self.buff = [0] * self.size
        gc.collect()

    def __len__(self):
        return len(self.main)


class Chart:
    def __init__(self, app, name, title, position, size, border_color, len_buf=10, bd_width=3,
                 const_val=None, const_legend: str = None, const_func = None):
        self.screen = app.screen
        self.name = name
        self.title = title
        self.chart = pyc.Figure(self.screen, position[0], position[1], size[0], size[1])
        self.border = pygame.Rect(*position, *size)
        self.bd_params = border_color, bd_width
        self.buf = ListBuff(len_buf)
        self.const_buf = [const_val] * len_buf if const_val else None
        self.const_legend = 'const' if const_legend is None else const_legend
        self.const_func = const_func

    @property
    def const_val(self):
        return self.const_buf[0] if self.const_buf else None

    @const_val.setter
    def const_val(self, new_val):
        self.const_buf = [new_val] * len(self.buf) if new_val else None

    def draw(self, params):
        self.buf.extend(params[self.name])
#       self._refresh_iter(params)
        if len(self.buf) > 1:
            self.chart.add_title(f'{self.title}')
            self.chart.add_legend()
            self.chart.line(self.const_legend,
                            list(range(1, len(self.buf) + 1)),
                            [self.const_func()] * len(self.buf),
                            line_width=2)
            self.chart.line(self.title, list(range(1, len(self.buf) + 1)), self.buf.main, color=(242,133,0),line_width=3)
            self.chart.draw()
            pygame.draw.rect(self.screen, self.bd_params[0], self.border, self.bd_params[1])

    def get_xlim(self):
        return self.chart.chart_area.xdata_min, self.chart.chart_area.xdata_max

    def get_ylim(self):
        return self.chart.chart_area.ydata_min, self.chart.chart_area.ydata_max

    def set_xlim(self, lim):
        self.chart.xmin = lim[0]
        self.chart.xmax = lim[1]

    def set_ylim(self, lim):
        self.chart.ymin = lim[0]
        self.chart.ymax = lim[1]

    def _refresh_iter(self, params):
        if params['is_changed']:
            self.buf.refresh()
