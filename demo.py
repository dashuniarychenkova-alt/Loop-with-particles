"""
Modified Demo module without spring particles.

This version of ``demo.py`` interfaces with the updated ``Simulation``
class that lacks spring particles.  It removes all logic associated with
the spring particle pair and renders only the gas particles.  The
parameters for the spring (``R``, ``m_spring``) are ignored.  The
potential energy arrays remain for compatibility but will contain zeroes.
"""

import math
import pygame
import numpy as np
import config
from typing import Dict, Optional
from simulation import Simulation, kB
from ui_base import get_font

# Unified simulation speed range shared with the UI.
SPEED_MIN = 0.2
SPEED_MAX = 6.0

# Flux graph visuals
FLUX_GRAPH_DEFAULT_HALF_RANGE = 60.0
FLUX_GRAPH_DEFAULT_GRID_LINES = 5
FLUX_GRID_COLOR = (222, 227, 242)
FLUX_GRAPH_FOLLOW_WINDOW = 240
FLUX_GRAPH_PADDING_FRACTION = 0.15
FLUX_GRAPH_EXTRA_PADDING = 0.25
# Time range (seconds) shown on the flux chart
FLUX_GRAPH_TIME_WINDOW = 10.0
# Graph zooming and axis shaping
FLUX_GRAPH_PARTICLE_REF = 200.0
FLUX_GRAPH_MIN_ZOOM = 0.35
FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION = 0.25
# Visual scale for displaying flux/heat numbers (multiplies raw values).
# Reduced to keep onscreen numbers in a readable range.
FLUX_GRAPH_DISPLAY_SCALE = 1.0e2

# Midplane divider visuals
MIDPLANE_WALL_THICKNESS_RATIO = 0.0024
MIDPLANE_WALL_COLOR = (0, 200, 0, 215)

FIELD_LINE_PARAM_KEYS = frozenset({'foot_radius', 'foot_distance', 'B0', 'B_background'})


class Demo:
    def __init__(
        self,
        app,
        position,
        demo_size,
        bg_color,
        border_color,
        bg_screen_color,
        params,
        wall_temp_bounds: tuple[float, float] | None = None,
    ):
        """
        Initialize a new demonstration instance.

        Parameters
        ----------
        app : App
            Reference to the parent application containing the pygame screen.
        position : tuple
            (x, y) coordinates of the top‑left corner of the simulation box.
        demo_size : tuple
            (width, height) of the simulation area in pixels.
        bg_color : tuple
            RGB background colour of the simulation area.
        border_color : tuple
            RGB colour of the border around the simulation area.
        bg_screen_color : tuple
            Colour used for the masked outer border region.
        params : dict
            Dictionary of initial simulation parameters (gamma, k, R, T, r, etc.).
        """
        self.app = app
        self.screen = app.screen
        self.bg_color = bg_color
        self.bg_screen_color = bg_screen_color
        self.bd_color = border_color
        self.position = position
        # Pygame rect describing the simulation area
        self.main = pygame.Rect(*position, *demo_size)
        # Store individual dimensions so the demo can be rectangular
        self.width, self.height = demo_size
        if wall_temp_bounds and wall_temp_bounds[1] > wall_temp_bounds[0]:
            self.wall_temp_bounds = (float(wall_temp_bounds[0]), float(wall_temp_bounds[1]))
        else:
            self.wall_temp_bounds = (100.0, 2000.0)
        self._wall_colors = {'left': (220, 70, 40), 'right': (60, 130, 255)}
        speed_default = params.get('speed', params.get('slowmo', 1.0))
        # Copy of the initial parameter values used by sliders
        # Copy of the initial parameter values used by sliders.  Remove keys
        # corresponding to unused simulation parameters (gamma, k, mass of spring,
        # spring radius, etc.) to avoid storing unneeded data.  If those keys
        # are not present, ``pop`` simply returns ``None``.  This keeps
        # ``self.params`` compact and eliminates references to unused legacy
        # spring parameters.
        self.params = dict(params)
        for unused_key in ('gamma', 'k', 'm_spring', 'R', 'R_spring', 'radius_scale'):
            self.params.pop(unused_key, None)
        self.params.pop('slowmo', None)
        # Drop legacy magnetic gradient param (unused)
        self.params.pop('magnetic_gradient', None)
        # Defaults for new footpoint and field controls
        self.params.setdefault('foot_radius', 0.12)
        self.params.setdefault('foot_distance', 0.65)
        self.params.setdefault('foot_charge_scale', 1.0)
        self.params.setdefault('foot_offset', 0.0)
        self.params.setdefault('foot_asymmetry', 0.0)
        self.params.setdefault('B_left', 1.0)
        self.params.setdefault('B_right', -1.0)
        self.params.setdefault('B0', 1.0)
        self.params.setdefault('B_background', 0.8)
        self.params.setdefault('B_grad_x', 0.0)
        self.params.setdefault('B_grad_y', 0.0)
        self.params.setdefault('transition_width', 0.12)
        self.params.setdefault('kperp_scale', 0.6)
        self.params.setdefault('kperp_contrast', 0.35)
        self.params.setdefault('dmm_min', 0.05)
        self.params.setdefault('dmm_max', 0.6)
        self.params.setdefault('dmm_dependence', 0.0)  # 0 -> |B|, 1 -> T
        self.params.setdefault('T_left', 800.0)
        self.params.setdefault('T_right', 650.0)
        self.params.setdefault('T_background', 500.0)
        self.params.setdefault('field_lines_enabled', True)
        self.params.setdefault('field_line_count', 9)
        self.params.setdefault('field_line_thickness', 2.5)
        self.params.setdefault('field_line_spread', 0.18)
        self.params.setdefault('field_line_spacing_px', 12)
        self.params.setdefault('field_line_focus_spacing_px', 6)
        self.params.setdefault('field_line_focus_radius_scale', 2.2)
        self.params.setdefault('field_line_focus_ratio', 0.35)
        self.params.setdefault('particles_to_add', 10.0)
        self.params.setdefault('heatmap_mode', 'Bz')
        self.params.setdefault('tails_enabled', False)
        self.params.setdefault('lambda_param', 0.0)
        # Ensure particle count is an integer
        try:
            self.params['r'] = int(round(float(self.params.get('r', 0))))
        except Exception:
            self.params['r'] = 0
        self.flux_display_scale: float = FLUX_GRAPH_DISPLAY_SCALE
        loader = config.ConfigLoader()
        self._configure_flux_graph(loader)

        # Scale factors for physical collisions and rendering.  Start 30% larger
        # to make particles more visible by default.
        initial_scale = float(params.get('size_scale', 1.3))
        if initial_scale < 0.1:
            initial_scale = 0.1
        self.physical_radius_scale = initial_scale
        self.draw_radius_factor = initial_scale

        # Keep track of indices of tagged (marked) particles.  These will
        # be drawn in a distinct colour and can be analysed separately.
        self.tagged_indices: list[int] = []
        # Tagged particles must stay clearly visible on projectors with
        # high brightness, so we render them in a dark brown tone instead
        # of yellow.
        self.tagged_color: tuple[int, int, int] = (48, 32, 20)
        # Visual bursts for decaying particles
        self.decay_bursts: list[dict] = []
        # Rendering helpers for highlight/dim features
        self.dim_untracked: bool = False
        self.dim_color: tuple[int, int, int] = (212, 212, 218)
        self.focus_field_color: tuple[int, int, int] = (72, 104, 255)
        self.focus_wind_color: tuple[int, int, int] = (255, 160, 90)

        # Trajectory tracking for a tagged particle
        self.trail_enabled: bool = False
        self.trail_points: list[tuple[int, int]] = []
        self.max_trail_points: int = 1500
        self._trail_requests: set[str] = set()
        self.tracked_particle_id: Optional[int] = None
        self._last_particle_screen_positions: np.ndarray | None = None
        self.last_draw_radius: int = 0
        self._magnetic_gradient: float = 0.0
        self.focus_active: bool = False
        self.focus_particle_idx: Optional[int] = None

        # Counters for wall contacts by tagged particles
        self.wall_hits = {'left': 0, 'right': 0}
        # Flux tracking across the midplane (x = 0.5)
        self.midplane_axis: str = 'x'
        self.midplane_position: float = 0.5
        self.midplane_flux_samples: list[tuple[float, float, float, float]] = []
        self.max_flux_samples: int = 2000
        self.last_flux_limits: Optional[tuple[float, float]] = None
        self.reset_wall_hit_counters()

        # Colour scaling for particle temperatures
        self.color_gamma: float = 0.75
        self._color_scale_min: float = 0.0
        self._color_scale_max: float = 1.0
        self._set_color_scale_bounds(self.wall_temp_bounds)

        # Unified speed control: slider value maps to step count and integrator scale
        self.time_scale: float = 1.0
        self.speed_factor: float = self._normalize_speed_value(speed_default)
        self._speed_steps: int = max(1, int(math.floor(self.speed_factor)))
        self.params['speed'] = self.speed_factor
        # Cached background gradient surface
        self._bg_gradient_surface: Optional[pygame.Surface] = None
        self._bg_gradient_size: Optional[tuple[int, int]] = None
        self.heatmap_surface: Optional[pygame.Surface] = None
        self._heatmap_cache_key: Optional[tuple] = None
        self.tail_history: list = []
        self.max_tail_points: int = 18
        self._field_lines_surface: Optional[pygame.Surface] = None
        self._field_line_cache_key: Optional[tuple] = None
        self._pending_foot_params: dict[str, float] = {}

        # Masses for gas particles only.  ``self.params['r']`` specifies the
        # number of gas particles; there are no spring masses in this
        # simplified model.  Masses are drawn from configuration.
        particle_count = max(0, int(self.params['r']))
        m = np.ones((particle_count,), dtype=float) * loader["R_mass"]

        # Legacy parameters (gamma, k, l_0) are passed through for API compatibility
        l_0 = loader['l_0']
        # Cache the base particle radius from the configuration.  ``R_size``
        # defines the nominal physical radius (in box units).  By
        # storing this value we can compute new radii when the user
        # adjusts the particle size via the UI.
        self.base_radius = loader["R_size"]
        # Initialize the simulation.  Multiply the base radius by our
        # physical scale factor to enlarge the physical collision size.
        # Legacy parameters may be absent in ``params`` because the UI
        # does not expose sliders for them.  Fetch them with default
        # fallbacks.  ``gamma`` and ``k`` are unused in the current
        # simulation but accepted for API compatibility.
        gamma_val = params.get('gamma', 1.0)
        k_val = params.get('k', 1.0)
        loader_dict = getattr(loader, '_loader', {})
        magnetic_field_cfg = loader_dict.get('magnetic_field', {}) if isinstance(loader_dict, dict) else {}
        solar_wind_cfg = loader_dict.get('solar_wind', {}) if isinstance(loader_dict, dict) else {}
        if isinstance(magnetic_field_cfg, dict):
            try:
                base_bg = float(magnetic_field_cfg.get('base', self.params.get('B_background', 0.8)))
                self.params['B_background'] = base_bg
            except Exception:
                pass
        # Create the simulation with legacy parameters, particle radius and counts.
        self.simulation = Simulation(
            gamma=gamma_val,
            k=k_val,
            l_0=l_0,
            R=self.base_radius * self.physical_radius_scale,
            particles_cnt=self.params['r'],
            T=self.params['T'],
            m=m,
            magnetic_field=magnetic_field_cfg,
            solar_wind=solar_wind_cfg,
        )
        # Field bounds cache for UI
        self.b_min, self.b_max = self.simulation.get_field_min_max()
        self.wind_speed = float(solar_wind_cfg.get('speed', 0.0)) if isinstance(solar_wind_cfg, dict) else 0.0
        grad_value = self.params.get('magnetic_gradient')
        if grad_value is None:
            try:
                mag_cfg = loader['magnetic_field']
                grad_value = mag_cfg.get('gradient', 0.0)
            except KeyError:
                grad_value = 0.0
        self.set_magnetic_gradient(grad_value)
        try:
            bg_b = float(self.params.get('B_background', getattr(self.simulation, '_B_background', 0.8)))
        except Exception:
            bg_b = getattr(self.simulation, '_B_background', 0.8)
        self.simulation.set_field_range(bg_b, bg_b)
        self.params['B_background'] = bg_b
        self.simulation.set_lambda_param(self.params.get('lambda_param', 0.0))
        self._apply_absorber_to_sim()
        self._update_simulation_box_aspect()
        self.midplane_axis = self.simulation.get_midplane_axis()
        self.midplane_position = self.simulation.get_midplane_position()
        # If thermal wall parameters are provided in params, set them on the simulation
        # (These keys may not exist in older configs.)
        t_left = params.get('T_left')
        t_right = params.get('T_right')
        update_kwargs: Dict[str, float] = {}
        if t_left is not None:
            update_kwargs['T_left'] = t_left
        if t_right is not None:
            update_kwargs['T_right'] = t_right
        if update_kwargs:
            self.simulation.set_params(**update_kwargs)

        # Apply the initial speed factor after the simulation is created.
        self.update_speed_factor(self.speed_factor, force=True)

    def _update_simulation_box_aspect(self) -> None:
        """Resize the simulation domain horizontally to match the viewport ratio."""
        if self.height <= 0:
            return
        # Align physical domain with full visible rectangle (no inset walls)
        self.wall_thickness_px = 0
        self.sim_left_px = self.position[0]
        self.sim_width_px = self.width
        usable_width = max(1, self.width)
        aspect = usable_width / self.height if self.height > 0 else 1.0
        if aspect <= 0:
            aspect = 1.0
        self.simulation.set_box_width(aspect)
        self._apply_absorber_to_sim()

    def _invalidate_heatmap(self) -> None:
        """Mark the cached heatmap as stale so it will be rebuilt next frame."""
        self.heatmap_surface = None
        self._heatmap_cache_key = None
        self._invalidate_field_lines()

    def _invalidate_field_lines(self) -> None:
        """Clear the cached field-line artwork so it can be recomputed."""
        self._field_lines_surface = None
        self._field_line_cache_key = None

    def _foot_centers(self) -> list[tuple[float, float]]:
        """Return footpoint centres in simulation units."""
        box_w = max(self.simulation.get_box_width(), 1e-6)
        radius = max(0.0, float(self.params.get('foot_radius', 0.12)))
        distance = max(0.02, float(self.params.get('foot_distance', 0.4)))
        distance = min(distance, box_w - 2 * radius)
        half = distance / 2.0
        asym = float(self.params.get('foot_asymmetry', 0.0))
        base_x = box_w * 0.5
        x1 = base_x - half - asym
        x2 = base_x + half + asym
        x1 = min(max(radius, x1), box_w - radius)
        x2 = min(max(radius, x2), box_w - radius)
        y_base = 0.5 + float(self.params.get('foot_offset', 0.0))
        y_base = min(max(radius, y_base), 1.0 - radius)
        return [(x1, y_base), (x2, y_base)]

    def _apply_absorber_to_sim(self) -> None:
        """Send updated absorption zones to the simulation."""
        if not getattr(self, 'simulation', None):
            return
        centers = self._foot_centers()
        radius = max(0.0, float(self.params.get('foot_radius', 0.12)))
        self.simulation.set_absorber_zones(centers, radius)
        gradient_amplitude = float(self.params.get('B0', 1.0))
        gradient_strengths = []
        for idx in range(len(centers)):
            amp = float(self.params.get('B_left', 1.0)) if idx == 0 else float(self.params.get('B_right', -1.0))
            gradient_strengths.append(gradient_amplitude * amp)
        self.simulation.set_footpoint_field(centers, radius, gradient_strengths)

    def _temperature_to_color(self, temperature: float) -> tuple[int, int, int]:
        """
        Map a wall temperature to the same blue-to-red gradient used for particles.
        """
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            temp_value = 0.0
        normalized = self._normalize_temperatures(np.array([temp_value], dtype=float))
        norm = float(normalized[0]) if normalized.size else 0.0
        norm = max(0.0, min(1.0, norm))
        red = int(round(255 * norm))
        blue = int(round(255 * (1.0 - norm)))
        return red, 0, blue

    def get_wall_color(self, side: str) -> tuple[int, int, int]:
        """Return the most recently rendered colour for the given wall."""
        return self._wall_colors.get(side, (255, 255, 255))

    def update_radius_scale(self, scale: float) -> None:
        """
        Update both the physical and visual radius scales of the particles.

        This method is intended to be called from the demo screen when
        the user changes the particle size slider.  It sets the
        ``physical_radius_scale`` and ``draw_radius_factor`` to the same
        value, updates the simulation's physical radius accordingly,
        and stores the new scale.  Changing the physical radius
        influences collision detection, while changing the draw factor
        adjusts the rendered size on screen.  Both are applied at once
        so that the user perceives the change consistently.

        Parameters
        ----------
        scale : float
            New scaling factor (1.0 means default size; higher values
            enlarge particles; lower values shrink them).
        """
        # Avoid zero or negative scales to keep a valid radius.
        if scale < 0.1:
            scale = 0.1
        # Store the new scale for both physical interactions and drawing
        self.physical_radius_scale = scale
        self.draw_radius_factor = scale
        # Compute the new physical radius using the cached base radius
        new_R = self.base_radius * self.physical_radius_scale
        # Update the simulation with the new radius
        self.simulation.set_params(R=new_R)

    def set_time_scale(self, scale: float) -> None:
        """Expose slow-motion control for the simulation loop."""
        if scale < 0.05:
            scale = 0.05
        self.time_scale = float(scale)
        self.simulation.set_time_scale(self.time_scale)

    def _normalize_speed_value(self, value: float) -> float:
        """Clamp raw speed input to the supported slider range."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1.0
        if not math.isfinite(value):
            value = 1.0
        return max(SPEED_MIN, min(SPEED_MAX, value))

    def update_speed_factor(self, raw_value: float, force: bool = False) -> float:
        """
        Update simulation stepping based on a single unified speed slider.

        The slider value controls both the number of integration steps per frame
        and the integrator time multiplier.  Values above 1.0 execute multiple
        substeps while keeping each substep stable; values below 1.0 slow the
        simulation down.
        """
        value = self._normalize_speed_value(raw_value)
        steps = max(1, int(math.floor(value)))
        step_scale = value / steps
        speed_changed = force or steps != getattr(self, '_speed_steps', 1)
        scale_changed = force or abs(step_scale - self.time_scale) > 1e-4
        self.speed_factor = value
        self._speed_steps = steps
        self.params['speed'] = value
        if speed_changed or scale_changed:
            self.set_time_scale(step_scale)
        return self.speed_factor

    def resize_viewport(self, position: tuple[int, int], demo_size: tuple[int, int]) -> None:
        """Adjust the rendering viewport to a new rectangle."""
        self.position = position
        self.main = pygame.Rect(*position, *demo_size)
        self.width, self.height = demo_size
        self.screen = self.app.screen
        self._update_simulation_box_aspect()
        self._invalidate_heatmap()


    def set_dim_untracked(self, dim: bool) -> None:
        """Enable or disable dimming of untagged particles."""
        self.dim_untracked = bool(dim)

    def set_trail_enabled(self, enabled: bool) -> None:
        """Toggle trajectory rendering for the tracked tagged particle."""
        self._set_trail_request('user', enabled)

    def set_tracked_trail_enabled(self, enabled: bool) -> None:
        """Ensure the tracked particle trail is visible while tracking."""
        self._set_trail_request('track', enabled)

    def _set_trail_request(self, source: str, enabled: bool) -> None:
        """Maintain the union of sources requesting trail rendering."""
        enabled = bool(enabled)
        prev_active = bool(self._trail_requests)
        if enabled:
            self._trail_requests.add(source)
        else:
            self._trail_requests.discard(source)
        active = bool(self._trail_requests)
        if active and not prev_active:
            self.trail_points.clear()
        elif not active and prev_active:
            self.trail_points.clear()
        self.trail_enabled = active
        if enabled:
            self._ensure_tracked_particle()

    def set_magnetic_gradient(self, gradient: float) -> None:
        """Set the current magnetic field gradient used in the in-plane push."""
        try:
            value = float(gradient)
        except (TypeError, ValueError):
            return
        self._magnetic_gradient = value
        self.simulation.set_params(magnetic_gradient=value)
        self.params['magnetic_gradient'] = value

    def set_focus_tracking(self, enabled: bool, *, reselect: bool = False) -> None:
        """Toggle the focused particle overlay that highlights field/wind vectors."""
        new_state = bool(enabled)
        self.focus_active = new_state
        if not new_state:
            self.focus_particle_idx = None
            return
        if reselect:
            if self.tracked_particle_id is not None:
                self.focus_particle_idx = int(self.tracked_particle_id)
            else:
                self._select_random_focus_particle()
        elif self.focus_particle_idx is None:
            self._select_random_focus_particle()

    def has_tagged_particles(self) -> bool:
        """Return True when at least one tagged particle exists."""
        return bool(self.tagged_indices)

    def get_wall_hit_counts(self) -> tuple[int, int]:
        """Return accumulated counts of tagged particles hitting left/right walls."""
        return self.wall_hits['left'], self.wall_hits['right']

    def reset_wall_hit_counters(self) -> None:
        """Clear stored hit counts for both walls."""
        self.wall_hits['left'] = 0
        self.wall_hits['right'] = 0
    def _sync_tagged_indices(self, particle_count: int) -> None:
        """Ensure tagged indices remain valid after particle count changes."""
        if not self.tagged_indices:
            return
        max_index = max(0, int(particle_count))
        filtered = [idx for idx in self.tagged_indices if 0 <= int(idx) < max_index]
        if len(filtered) != len(self.tagged_indices):
            self.tagged_indices = filtered
            self.reset_wall_hit_counters()
            if self.trail_points:
                self.trail_points.clear()
        if not self.tagged_indices:
            self.tracked_particle_id = None
            return
        if self.tracked_particle_id not in self.tagged_indices:
            self.tracked_particle_id = self.tagged_indices[0]
            if self.trail_points:
                self.trail_points.clear()

    def get_half_concentrations(self) -> tuple[float, float]:
        """Return particle concentration in the left and right halves of the box."""
        positions = self.simulation.r[0]
        width = max(self.simulation.get_box_width(), 1e-9)
        half = width / 2.0
        left_count = float(np.count_nonzero(positions < half))
        right_count = float(positions.size - left_count)
        half_area = max(half, 1e-9)
        left_density = left_count / half_area
        right_density = right_count / half_area
        return left_density, right_density

    def reset_measurements(self) -> None:
        """Reset visual counters, flux history and the tracked trail."""
        self.reset_wall_hit_counters()
        self.midplane_flux_samples.clear()
        self.last_flux_limits = None
        if self.trail_points:
            self.trail_points.clear()
        self.simulation.reset_midplane_statistics()
        # Also reset velocities to calm the system
        self.simulation.zero_velocities()
        self._ensure_tracked_particle()

    def clear_tagged_particles(self) -> int:
        """Remove all tagged particles that were added to the simulation."""
        removed = len(self.tagged_indices)
        if removed <= 0:
            if self.trail_points:
                self.trail_points.clear()
            self._ensure_tracked_particle()
            return 0
        target_count = max(0, int(self.simulation._n_particles) - removed)
        self.simulation.set_params(particles_cnt=target_count)
        self.params['r'] = target_count
        self.tagged_indices.clear()
        self.tracked_particle_id = None
        if self.trail_points:
            self.trail_points.clear()
        self.reset_wall_hit_counters()
        self._ensure_tracked_particle()
        return removed

    def _ensure_tracked_particle(self) -> None:
        """Ensure the tracked particle id remains valid."""
        count = int(max(0, getattr(self.simulation, '_n_particles', 0)))
        if self.tracked_particle_id is not None:
            idx = int(self.tracked_particle_id)
            if 0 <= idx < count:
                return
        if self.tagged_indices:
            self.tracked_particle_id = self.tagged_indices[0]
            if self.trail_points:
                self.trail_points.clear()
            return
        self.tracked_particle_id = None
        if self.trail_points:
            self.trail_points.clear()

    def _collect_decay_events(self, x_scale: float, y_scale: float) -> None:
        """Convert decay events from simulation space into renderable bursts."""
        events = self.simulation.consume_decay_events()
        if not events:
            return
        type_defs = self.simulation.get_type_definitions()
        for pos, t_idx in events:
            x_px = self.sim_left_px + pos[0] * x_scale
            y_px = self.position[1] + self.height - pos[1] * y_scale
            color = (200, 200, 200)
            if 0 <= t_idx < len(type_defs):
                color = type_defs[t_idx].color
            self.decay_bursts.append(
                {
                    'pos': (x_px, y_px),
                    'ttl': 10,
                    'radius': max(6, int(self.draw_radius_factor * 8)),
                    'color': color,
                }
            )

    def _render_decay_bursts(self) -> None:
        """Draw and age-out decay bursts."""
        if not self.decay_bursts:
            return
        alive = []
        for burst in self.decay_bursts:
            ttl = burst.get('ttl', 0)
            if ttl <= 0:
                continue
            pos = burst.get('pos', (0, 0))
            radius = burst.get('radius', 8)
            color = burst.get('color', (255, 255, 255))
            alpha = int(255 * ttl / 10)
            surf_size = radius * 4
            surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            center = surf_size // 2
            pygame.draw.circle(surface, (*color, alpha), (center, center), radius + 4, width=4)
            pygame.draw.circle(surface, (*color, max(50, alpha)), (center, center), radius // 2)
            top_left = (int(pos[0] - center), int(pos[1] - center))
            self.screen.blit(surface, top_left)
            burst['ttl'] = ttl - 1
            alive.append(burst)
        self.decay_bursts = alive

    def _project_particle_positions(self, positions: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        """
        Convert simulation-space positions to screen-space points and reuse the radius info.
        """
        draw_scale = self.draw_radius_factor
        phys_scale = self.physical_radius_scale if self.physical_radius_scale > 0 else 1.0
        radius_box_units = self.simulation.R * (draw_scale / phys_scale)

        box_width_units = max(self.simulation.get_box_width(), 1e-6)
        x_scale = self.sim_width_px / box_width_units
        y_scale = self.height

        positions[0] = self.sim_left_px + positions[0] * x_scale
        positions[1] = self.position[1] + self.height - positions[1] * y_scale
        positions = np.round(positions).astype(int)

        self._last_particle_screen_positions = positions.copy()
        unit_scale = min(x_scale, y_scale)
        r_radius = max(1, int(round(unit_scale * radius_box_units)))
        self.last_draw_radius = r_radius
        return positions, r_radius, x_scale, y_scale

    def _draw_field_background(self) -> None:
        """Render the plain background with heatmap, field lines, footpoints, and arrow."""
        rect = self.main
        pygame.draw.rect(self.screen, self.bg_color, rect)

        if self.params.get('field_lines_enabled', False):
            self._ensure_field_lines_surface()
            if self._field_lines_surface is not None:
                self.screen.blit(self._field_lines_surface, rect.topleft)
        else:
            self._field_lines_surface = None
            self._field_line_cache_key = None

        centers = self._foot_centers()
        self._draw_background_B_arrow(rect)
        self._draw_footpoints(rect, centers)

    def _normalize_field(self, field: np.ndarray) -> np.ndarray:
        """Normalize a field to [0, 1] using robust percentiles."""
        finite = np.isfinite(field)
        if not np.any(finite):
            return np.zeros_like(field, dtype=float)
        data = field[finite]
        low = np.percentile(data, 2)
        high = np.percentile(data, 98)
        if high <= low:
            high = low + 1e-6
        norm = (field - low) / (high - low)
        return np.clip(norm, 0.0, 1.0)

    def _colormap(self, norm: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
        """Linearly blend between two RGB colours based on ``norm`` in [0,1]."""
        norm = np.clip(norm, 0.0, 1.0)
        low_arr = np.array(low, dtype=float)
        high_arr = np.array(high, dtype=float)
        rgb = low_arr + (high_arr - low_arr) * norm[..., None]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _build_heatmap_surface(self, size: tuple[int, int]) -> Optional[pygame.Surface]:
        """Generate a heatmap surface for the selected layer."""
        width, height = size
        mode = str(self.params.get('heatmap_mode', 'Bz'))
        cache_key = (
            size,
            mode,
            float(self.params.get('foot_radius', 0.12)),
            float(self.params.get('foot_distance', 0.4)),
            float(self.params.get('foot_offset', 0.0)),
            float(self.params.get('foot_asymmetry', 0.0)),
            float(self.params.get('B_left', 1.0)),
            float(self.params.get('B_right', -1.0)),
            float(self.params.get('B0', 1.0)),
            float(self.params.get('B_background', 0.8)),
            float(self.params.get('B_grad_x', 0.0)),
            float(self.params.get('B_grad_y', 0.0)),
            float(self.params.get('transition_width', 0.12)),
            float(self.params.get('kperp_scale', 0.6)),
            float(self.params.get('kperp_contrast', 0.35)),
            float(self.params.get('dmm_min', 0.05)),
            float(self.params.get('dmm_max', 0.6)),
            float(self.params.get('dmm_dependence', 0.0)),
            float(self.params.get('T_left', 800.0)),
            float(self.params.get('T_right', 650.0)),
            float(self.params.get('T_background', 500.0)),
        )
        if self.heatmap_surface is not None and self._heatmap_cache_key == cache_key:
            return self.heatmap_surface

        box_w = max(self.simulation.get_box_width(), 1e-6)
        grid_w = max(80, min(280, width // 3))
        grid_h = max(80, min(220, height // 3))
        xs = np.linspace(0.0, box_w, grid_w, dtype=float)
        ys = np.linspace(0.0, 1.0, grid_h, dtype=float)
        xg, yg = np.meshgrid(xs, ys)

        foot_radius = max(0.01, float(self.params.get('foot_radius', 0.12)))
        peak_base = float(self.params.get('B0', 0.0))
        background_B = float(self.params.get('B_background', 0.8))
        foot_scale = float(self.params.get('foot_charge_scale', 1.0))

        centers = self._foot_centers()
        profiles = []
        Bz = np.zeros_like(xg)
        for idx, (cx, cy) in enumerate(centers):
            amp_scale = float(self.params.get('B_left', 1.0)) if idx == 0 else float(self.params.get('B_right', -1.0))
            amp = peak_base * amp_scale * foot_scale
            r2 = (xg - cx) ** 2 + (yg - cy) ** 2
            r = np.sqrt(np.maximum(r2, 1e-12))
            profile = np.exp(-((r / foot_radius) ** 4))
            profiles.append(profile)
            Bz += amp * profile

        profile_sum = np.zeros_like(xg)
        for p in profiles:
            profile_sum += p

        B_abs = np.abs(Bz) + abs(background_B)
        T_bg = max(0.0, float(self.params.get('T_background', 500.0)))
        T_field = np.full_like(xg, T_bg)
        T_left = max(0.0, float(self.params.get('T_left', 800.0)))
        T_right = max(0.0, float(self.params.get('T_right', 650.0)))
        for idx, profile in enumerate(profiles):
            target = T_left if idx == 0 else T_right
            T_field += (target - T_bg) * profile

        k_scale = max(0.0, float(self.params.get('kperp_scale', 0.6)))
        k_contrast = float(self.params.get('kperp_contrast', 0.35))
        norm_gauss = self._normalize_field(profile_sum)
        k_perp = k_scale * (1.0 + k_contrast * (norm_gauss * 2 - 1))
        k_perp = np.clip(k_perp, 0.0, None)

        dmm_min = float(self.params.get('dmm_min', 0.05))
        dmm_max = float(self.params.get('dmm_max', 0.6))
        dmm_bias = float(self.params.get('dmm_dependence', 0.0))
        norm_B = self._normalize_field(B_abs)
        norm_T = self._normalize_field(T_field)
        driver = np.clip((1.0 - dmm_bias) * norm_B + dmm_bias * norm_T, 0.0, 1.0)
        dmm = dmm_min + driver * (dmm_max - dmm_min)

        if mode.lower() == 'bz':
            max_abs = np.max(np.abs(Bz))
            if max_abs <= 1e-9:
                max_abs = 1.0
            norm_signed = np.clip(Bz / max_abs, -1.0, 1.0)
            pos = np.clip(norm_signed, 0.0, 1.0)
            neg = np.clip(-norm_signed, 0.0, 1.0)
            red = (0.25 + 0.75 * pos) * 255
            blue = (0.25 + 0.75 * neg) * 255
            green = (0.30 + 0.40 * (1.0 - np.abs(norm_signed))) * 255
            rgb = np.stack([red, green, blue], axis=2)
        elif mode.lower() in ('b', '|b|', 'absb'):
            norm = self._normalize_field(B_abs)
            rgb = self._colormap(norm, (230, 238, 255), (120, 70, 210))
        elif mode.lower() in ('kperp', 'k_⊥', 'k\\perp'):
            norm = self._normalize_field(k_perp)
            rgb = self._colormap(norm, (210, 240, 230), (0, 150, 160))
        elif mode.lower() in ('dmm', 'dμμ', 'd_mu'):
            norm = self._normalize_field(dmm)
            rgb = self._colormap(norm, (235, 240, 210), (240, 140, 70))
        else:  # Temperature
            norm = self._normalize_field(T_field)
            rgb = self._colormap(norm, (40, 70, 120), (255, 200, 80))

        rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)[::-1, :, :]
        surf_array = np.transpose(rgb_uint8, (1, 0, 2))
        surface = pygame.surfarray.make_surface(surf_array)
        surface = surface.convert()
        self.heatmap_surface = surface
        self._heatmap_cache_key = cache_key
        return surface

    def _ensure_field_lines_surface(self) -> None:
        """Rebuild the cached field-line surface when parameters change."""
        if self.main.width <= 0 or self.main.height <= 0 or not getattr(self, 'simulation', None):
            self._field_lines_surface = None
            self._field_line_cache_key = None
            return
        if not self.params.get('field_lines_enabled', False):
            self._field_lines_surface = None
            self._field_line_cache_key = None
            return
        key = self._compute_field_line_cache_key()
        if key is None:
            return
        if self._field_line_cache_key == key and self._field_lines_surface is not None:
            return
        surface = self._rebuild_field_lines_surface(key)
        self._field_lines_surface = surface
        self._field_line_cache_key = key

    def _compute_field_line_cache_key(self) -> tuple | None:
        """Return a hashable key representing the current field-line state."""
        if not getattr(self, 'simulation', None):
            return None
        centers = tuple(
            (round(coord[0], 4), round(coord[1], 4))
            for coord in self._foot_centers()
        )
        return (
            round(self.width, 2),
            round(self.height, 2),
            centers,
            round(float(self.params.get('foot_radius', 0.12)), 4),
            round(float(self.params.get('foot_distance', 0.65)), 4),
            round(float(self.params.get('B0', 1.0)), 4),
            round(float(self.params.get('B_background', 0.8)), 4),
            round(float(self.params.get('B_left', 1.0)), 4),
            round(float(self.params.get('B_right', -1.0)), 4),
            round(float(self.params.get('foot_charge_scale', 1.0)), 4),
            round(float(self.params.get('field_line_count', 9)), 3),
            round(float(self.params.get('field_line_thickness', 2.5)), 3),
            round(float(self.params.get('field_line_spread', 0.18)), 4),
            round(float(self.params.get('field_line_spacing_px', 12)), 3),
            round(float(self.params.get('field_line_focus_spacing_px', 6)), 3),
            round(float(self.params.get('field_line_focus_radius_scale', 2.2)), 3),
            round(float(self.params.get('field_line_focus_ratio', 0.35)), 3),
            round(float(self.simulation.get_box_width()), 4),
        )

    def _rebuild_field_lines_surface(self, key: tuple) -> pygame.Surface | None:
        rect = self.main
        if rect.width <= 0 or rect.height <= 0:
            return None
        lines = self._generate_field_lines()
        if not lines:
            return None
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        box_width = max(self.simulation.get_box_width(), 1e-6)
        x_scale = self.sim_width_px / box_width if box_width > 0 else 0.0
        y_scale = self.height
        color = (80, 125, 205, 220)
        line_width = max(2, int(round(float(self.params.get('field_line_thickness', 2.5)))))
        for line in lines:
            if not line:
                continue
            points = []
            for x, y in line:
                px = int(round(self.sim_left_px + x * x_scale))
                py = int(round(rect.bottom - y * y_scale))
                points.append((px - rect.left, py - rect.top))
            if len(points) >= 2:
                pygame.draw.lines(surface, color, False, points, width=line_width)
        return surface

    def _generate_field_lines(self) -> list[list[tuple[float, float]]]:
        box_width = max(self.simulation.get_box_width(), 1e-6)
        centers = self._foot_centers()
        radius = max(float(self.params.get('foot_radius', 0.12)), 0.01)

        B_background = float(self.params.get('B_background', 0.8))
        focus_radius_scale = max(1.0, float(self.params.get('field_line_focus_radius_scale', 2.2)))
        focus_ratio = max(0.0, min(1.0, float(self.params.get('field_line_focus_ratio', 0.35))))
        spacing_px = max(4.0, float(self.params.get('field_line_spacing_px', 12)))
        focus_spacing_px = max(2.0, float(self.params.get('field_line_focus_spacing_px', 6)))
        box_width = max(box_width, 1e-6)

        rect = self.main
        x_scale = self.sim_width_px / box_width if box_width > 0 else 0.0
        y_scale = self.height

        foot_charge = float(self.params.get('foot_charge_scale', 1.0))
        B0 = float(self.params.get('B0', 1.0))
        base_amps = [
            B0 * float(self.params.get('B_left', 1.0)) * foot_charge,
            B0 * float(self.params.get('B_right', -1.0)) * foot_charge,
        ]

        def _wrap_delta(dx: float, dy: float) -> tuple[float, float]:
            dx -= round(dx / box_width) * box_width
            dy -= round(dy / 1.0) * 1.0
            return dx, dy

        def _foot_sources(x: float, y: float) -> tuple[float, float]:
            vx = 0.0
            vy = 0.0
            radius_local = max(radius, 1e-3)
            for idx, center in enumerate(centers):
                if not center:
                    continue
                cx, cy = center
                if idx >= len(base_amps):
                    amp = 0.0
                else:
                    amp = base_amps[idx]
                if amp == 0.0:
                    continue
                dx = x - cx
                dy = y - cy
                dx, dy = _wrap_delta(dx, dy)
                r2 = dx * dx + dy * dy
                r = math.sqrt(max(r2, 1e-12))
                profile = math.exp(-((r / radius_local) ** 4))
                prefactor = (-4.0 * (r ** 3) / (radius_local ** 4)) * amp * profile
                if r > 1e-12:
                    prefactor /= r
                vx += prefactor * dx
                vy += prefactor * dy
            return vx, vy

        def vector_field(x: float, y: float) -> tuple[float, float]:
            fx, fy = _foot_sources(x, y)
            return (B_background + fx, fy)

        def _closest_foot_info(x: float, y: float) -> tuple[float, int | None, float, float, float, float]:
            best = (math.inf, None, 0.0, 0.0, 0.0, 0.0)
            for idx, center in enumerate(centers):
                if not center:
                    continue
                cx, cy = center
                dx = x - cx
                dy = y - cy
                dx, dy = _wrap_delta(dx, dy)
                dist = math.hypot(dx, dy)
                if dist < best[0]:
                    best = (dist, idx, dx, dy, cx, cy)
            return best

        def _focus_state(x: float, y: float) -> dict[str, float | bool | tuple[float, float] | int | None]:
            vx, vy = vector_field(x, y)
            source_x = vx - B_background
            source_y = vy
            source_mag = math.hypot(source_x, source_y)
            uniform = max(abs(B_background), 1e-6)
            ratio = source_mag / uniform
            dist, idx, dx, dy, cx, cy = _closest_foot_info(x, y)
            if idx is None:
                return {'focus': False, 'inside': False, 'dist': math.inf, 'dx': 0.0, 'dy': 0.0, 'center': (0.0, 0.0), 'ratio': ratio, 'index': None}
            focus_trigger = dist <= radius * focus_radius_scale or ratio >= focus_ratio
            inside = dist <= radius
            return {
                'focus': focus_trigger,
                'inside': inside,
                'dist': dist,
                'dx': dx,
                'dy': dy,
                'center': (cx, cy),
                'ratio': ratio,
                'index': idx,
            }

        def _clip_to_foot(info: dict[str, float | tuple[float, float] | int | None]) -> tuple[float, float] | None:
            center = info.get('center')
            if center is None:
                return None
            dist = float(info.get('dist', math.inf))
            if dist <= 1e-6:
                return None
            cx, cy = center
            dx = float(info.get('dx', 0.0))
            dy = float(info.get('dy', 0.0))
            scale = radius / dist
            clip_x = (cx + dx * scale) % box_width
            clip_y = (cy + dy * scale) % 1.0
            return clip_x, clip_y

        def _normalized_field(x: float, y: float) -> tuple[float, float]:
            vx, vy = vector_field(x, y)
            mag = math.hypot(vx, vy)
            if mag < 1e-6:
                return (0.0, 0.0)
            inv = 1.0 / (mag + 1e-12)
            return (vx * inv, vy * inv)

        def _rk4_step(x: float, y: float, h: float, direction: int) -> tuple[float, float] | None:
            k1x, k1y = _normalized_field(x, y)
            if direction < 0:
                k1x = -k1x
                k1y = -k1y
            k2x, k2y = _normalized_field(x + 0.5 * h * k1x, y + 0.5 * h * k1y)
            if direction < 0:
                k2x = -k2x
                k2y = -k2y
            k3x, k3y = _normalized_field(x + 0.5 * h * k2x, y + 0.5 * h * k2y)
            if direction < 0:
                k3x = -k3x
                k3y = -k3y
            k4x, k4y = _normalized_field(x + h * k3x, y + h * k3y)
            if direction < 0:
                k4x = -k4x
                k4y = -k4y
            deltax = (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            deltay = (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            if math.isnan(deltax) or math.isnan(deltay):
                return None
            return (x + deltax, y + deltay)

        def _trace_streamline(seed: tuple[float, float], direction: int, config: dict[str, float | int | bool]) -> list[tuple[float, float]]:
            pts: list[tuple[float, float]] = []
            x, y = seed
            prev_dir: tuple[float, float] | None = None
            h0_base = max(0.02, float(self.params.get('field_line_spread', 0.18)) * 0.25)
            h_min_base = max(0.003, h0_base * 0.3)
            h_max_base = max(0.04, h0_base * 2.5)
            max_steps = int(config.get('max_steps', 600))
            for _ in range(max_steps):
                focus_info = _focus_state(x, y)
                if focus_info['inside']:
                    break
                if focus_info['focus'] and not config.get('allow_focus', False):
                    break
                dir_vec = _normalized_field(x, y)
                if dir_vec == (0.0, 0.0):
                    break
                actual_dir = (dir_vec[0] * direction, dir_vec[1] * direction)
                h0 = h0_base
                h_min = h_min_base
                h_max = h_max_base
                if focus_info['focus']:
                    focus_scale = float(config.get('focus_step_scale', 0.6))
                    h0 *= focus_scale
                    h_min *= focus_scale
                    h_max *= focus_scale * 1.15
                h = h0
                if prev_dir is not None:
                    dot = max(-1.0, min(1.0, prev_dir[0] * actual_dir[0] + prev_dir[1] * actual_dir[1]))
                    ang = math.acos(dot)
                    h = h0 / (1.0 + 6.0 * ang)
                h = max(h_min, min(h_max, h))
                result = _rk4_step(x, y, h, direction)
                if result is None:
                    break
                nx, ny = result
                next_focus = _focus_state(nx, ny)
                if next_focus['inside']:
                    clip_target = _clip_to_foot(next_focus)
                    if clip_target is not None:
                        pts.append(clip_target)
                    break
                next_pt = (nx % box_width, ny % 1.0)
                spacing_needed = config.get('focus_spacing', spacing_px) if next_focus['focus'] else config.get('spacing', spacing_px)
                if _is_too_close(next_pt, float(spacing_needed)):
                    break
                pts.append(next_pt)
                prev_dir = actual_dir
                x, y = next_pt
            return pts

        def _chaikin(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
            if len(points) < 2 or iterations <= 0:
                return points
            pts = points
            for _ in range(iterations):
                new_pts = [pts[0]]
                for i in range(len(pts) - 1):
                    p = pts[i]
                    q = pts[i + 1]
                    new_pts.append(((3 * p[0] + q[0]) / 4.0, (3 * p[1] + q[1]) / 4.0))
                    new_pts.append(((p[0] + 3 * q[0]) / 4.0, (p[1] + 3 * q[1]) / 4.0))
                new_pts.append(pts[-1])
                pts = new_pts
            return pts

        def _to_screen(point: tuple[float, float]) -> tuple[float, float]:
            px = self.sim_left_px + point[0] * x_scale
            py = rect.bottom - point[1] * y_scale
            return px, py

        grid_cell_size = max(1.0, spacing_px, focus_spacing_px)
        grid: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

        def _grid_key(px: float, py: float) -> tuple[int, int]:
            return int(px // grid_cell_size), int(py // grid_cell_size)

        def _is_too_close(point: tuple[float, float], required_spacing: float) -> bool:
            px, py = _to_screen(point)
            key = _grid_key(px, py)
            threshold_sq = required_spacing * required_spacing
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = grid.get((key[0] + dx, key[1] + dy), [])
                    for bx, by, spacing in bucket:
                        required = max(required_spacing, spacing)
                        if (px - bx) ** 2 + (py - by) ** 2 < required * required:
                            return True
            return False

        def _stamp_line(points: list[tuple[float, float]], spacing: float) -> None:
            for point in points:
                px, py = _to_screen(point)
                key = _grid_key(px, py)
                grid.setdefault(key, []).append((px, py, spacing))

        background_seeds: list[tuple[float, float]] = []
        edge_seeds = max(3, min(10, int(max(1, self.params.get('field_line_count', 9)))))
        for i in range(edge_seeds):
            y = (i + 0.5) / edge_seeds
            background_seeds.append((0.0, y))
            background_seeds.append((box_width - 1e-6, y))

        detail_seeds: list[tuple[float, float]] = []
        if centers:
            desired_lines = max(6, int(max(1, self.params.get('field_line_count', 9))))
            angle_step = max(20, 360 // desired_lines)
            offsets = (radius * 1.1, radius * 1.45)
            seen: set[tuple[float, float]] = set()

            def _add_detail_seed(x: float, y: float) -> None:
                key = (round(x, 5), round(y, 5))
                if key in seen:
                    return
                if self._point_in_foot(x, y, centers, radius, box_width):
                    return
                seen.add(key)
                detail_seeds.append((x % box_width, y % 1.0))

            for offset in offsets:
                for cx, cy in centers:
                    for angle in range(0, 360, angle_step):
                        rad = math.radians(angle)
                        x = cx + offset * math.cos(rad)
                        y = cy + offset * math.sin(rad)
                        _add_detail_seed(x, y)

        pass_configs = [
            {
                'name': 'background',
                'seeds': background_seeds,
                'allow_focus': False,
                'spacing': spacing_px,
                'focus_spacing': spacing_px,
                'stamp_spacing': spacing_px,
                'max_steps': 450,
                'focus_step_scale': 0.75,
            },
            {
                'name': 'detail',
                'seeds': detail_seeds,
                'allow_focus': True,
                'spacing': spacing_px,
                'focus_spacing': focus_spacing_px,
                'stamp_spacing': focus_spacing_px,
                'max_steps': 900,
                'focus_step_scale': 0.45,
            },
        ]

        lines: list[list[tuple[float, float]]] = []
        for config in pass_configs:
            for seed in config['seeds']:
                forward = _trace_streamline(seed, 1, config)
                backward = _trace_streamline(seed, -1, config)
                if not forward and not backward:
                    continue
                line = backward[::-1] + [seed] + forward
                smoothed = _chaikin(line, iterations=2)
                if len(smoothed) >= 2:
                    _stamp_line(smoothed[::2], float(config.get('stamp_spacing', spacing_px)))
                    lines.append(smoothed)
        return lines

    @staticmethod
    def _point_in_foot(x: float, y: float, centers: list[tuple[float, float]], radius: float, box_width: float) -> bool:
        rad2 = radius * radius
        for cx, cy in centers:
            dx = x - cx
            dy = y - cy
            dx -= round(dx / box_width) * box_width
            dy -= round(dy / 1.0) * 1.0
            if dx * dx + dy * dy <= rad2:
                return True
        return False
    def _draw_background_B_arrow(self, rect: pygame.Rect) -> None:
        """Draw a horizontal background-field arrow at the top of the simulation box."""
        arrow_color = (70, 110, 220)
        length = max(60, int(rect.width * 0.2))
        margin_top = max(14, int(rect.height * 0.06))
        start = (rect.left + rect.width // 2 - length // 2, rect.top + margin_top)
        end = (start[0] + length, start[1])
        self._draw_vector_arrow(start, end, arrow_color)
        label_font = get_font(16, bold=True)
        label_surface = label_font.render('B', True, arrow_color)
        label_rect = label_surface.get_rect()
        label_rect.midbottom = (start[0] - max(6, length // 10), start[1] - 2)
        self.screen.blit(label_surface, label_rect)


    def _draw_footpoints(
        self,
        rect: pygame.Rect,
        centers: list[tuple[float, float]],
        surface: pygame.Surface | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> None:
        """Draw footpoint disks with polarity labels."""
        radius_unit = max(0.0, float(self.params.get('foot_radius', 0.12)))
        box_w = max(self.simulation.get_box_width(), 1e-6)
        x_scale = self.sim_width_px / box_w
        y_scale = self.height
        radius_px = max(8, int(round(radius_unit * min(x_scale, y_scale))))
        labels = ['+', '-']
        amps = [float(self.params.get('B_left', 1.0)), float(self.params.get('B_right', -1.0))]
        target_surface = surface if surface is not None else self.screen
        for idx, (cx, cy) in enumerate(centers):
            px = self.sim_left_px + cx * x_scale
            py = rect.bottom - cy * y_scale
            draw_x = px - offset[0]
            draw_y = py - offset[1]
            polarity = labels[idx] if idx < len(labels) else '+'
            polarity = '+' if amps[idx] >= 0 else '-'
            color = (220, 80, 80) if polarity == '+' else (70, 110, 220)
            pygame.draw.circle(target_surface, (255, 255, 255), (int(draw_x), int(draw_y)), radius_px + 2)
            pygame.draw.circle(target_surface, color, (int(draw_x), int(draw_y)), radius_px, width=3)
            font = get_font(max(16, radius_px), bold=True)
            label_surface = font.render(polarity, True, color)
            label_rect = label_surface.get_rect()
            label_rect.center = (int(draw_x), int(draw_y))
            target_surface.blit(label_surface, label_rect)

    def _select_random_focus_particle(self, particle_count: Optional[int] = None) -> None:
        """Choose a random particle index to highlight."""
        count = particle_count if particle_count is not None else self.simulation.get_particle_count()
        if count <= 0:
            self.focus_particle_idx = None
            return
        self.focus_particle_idx = int(np.random.randint(0, count))

    def _ensure_focus_particle(self, available_count: int) -> None:
        """Keep the focused particle index valid as the simulation changes."""
        if not self.focus_active:
            self.focus_particle_idx = None
            return
        if available_count <= 0:
            self.focus_particle_idx = None
            return
        tracked_idx = self.tracked_particle_id
        if tracked_idx is not None and 0 <= tracked_idx < available_count:
            self.focus_particle_idx = int(tracked_idx)
            return
        if self.focus_particle_idx is None or self.focus_particle_idx >= available_count:
            self._select_random_focus_particle(available_count)

    def _focus_dim_color(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Blend the provided color toward the dim palette for secondary particles."""
        return tuple(
            int(color[i] * 0.45 + self.dim_color[i] * 0.55)
            for i in range(3)
        )

    def _update_wall_hit_counters(self) -> None:
        """Increment counters when tagged particles touch left/right walls."""
        if not self.tagged_indices:
            return
        tagged_set = set(self.tagged_indices)
        hits = self.simulation.get_last_wall_hits()
        for idx in hits.get('left', []):
            if int(idx) in tagged_set:
                self.wall_hits['left'] += 1
        for idx in hits.get('right', []):
            if int(idx) in tagged_set:
                self.wall_hits['right'] += 1

    def _normalize_temperatures(self, temperatures: np.ndarray) -> np.ndarray:
        """Return colour-normalised intensities for the supplied temperatures."""
        temps = np.asarray(temperatures, dtype=float)
        if temps.size == 0:
            return np.zeros_like(temps, dtype=float)
        low = float(self._color_scale_min)
        high = float(self._color_scale_max)
        span = max(high - low, 1e-9)
        temps = np.nan_to_num(temps, nan=low, neginf=low, posinf=high)
        normalized = np.clip((temps - low) / span, 0.0, 1.0)
        gamma = float(self.color_gamma)
        if gamma not in (1.0, 0.0):
            normalized = np.power(normalized, gamma)
        return normalized

    def _set_color_scale_bounds(self, bounds: Optional[tuple[float, float]]) -> None:
        """Configure the fixed temperature range used for colour gradients."""
        default_low, default_high = 100.0, 2000.0
        low, high = default_low, default_high
        if bounds and len(bounds) == 2:
            try:
                low = float(bounds[0])
                high = float(bounds[1])
            except (TypeError, ValueError):
                low, high = default_low, default_high
        if not math.isfinite(low):
            low = default_low
        if not math.isfinite(high):
            high = default_high
        if high <= low:
            high = low + 1.0
        self.wall_temp_bounds = (low, high)
        self._color_scale_min = max(0.0, low)
        self._color_scale_max = max(self._color_scale_min + 1e-6, high)

    def set_wall_color_range(self, bounds: Optional[tuple[float, float]]) -> None:
        """
        Public helper to update the temperature bounds driving the colour gradient.
        """
        self._set_color_scale_bounds(bounds or self.wall_temp_bounds)

    def _record_trail_point(self, pixel_positions: np.ndarray) -> None:
        """Append the current screen-space position of the tracked particle."""
        if not self.trail_enabled:
            return
        self._ensure_tracked_particle()
        if self.tracked_particle_id is None:
            return
        idx = int(self.tracked_particle_id)
        if idx >= pixel_positions.shape[1]:
            return
        point = (int(pixel_positions[0, idx]), int(pixel_positions[1, idx]))
        if self.trail_points and self.trail_points[-1] == point:
            return
        self.trail_points.append(point)
        if len(self.trail_points) > self.max_trail_points:
            excess = len(self.trail_points) - self.max_trail_points
            del self.trail_points[:excess]

    def _record_tail_positions(self, pixel_positions: np.ndarray) -> None:
        """Persist short tails for every particle when enabled."""
        if not self.params.get('tails_enabled', False):
            self.tail_history = []
            return
        count = pixel_positions.shape[1] if pixel_positions is not None else 0
        if count <= 0:
            self.tail_history = []
            return
        if len(self.tail_history) != count:
            from collections import deque

            self.tail_history = [deque(maxlen=self.max_tail_points) for _ in range(count)]
        for idx in range(count):
            point = (int(pixel_positions[0, idx]), int(pixel_positions[1, idx]))
            self.tail_history[idx].append(point)

    def _draw_tails(self, colors: np.ndarray, rect: pygame.Rect, radius_px: int) -> None:
        """Render per-particle tails behind the main dots."""
        if not self.params.get('tails_enabled', False):
            return
        if not self.tail_history:
            return
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        for idx, trail in enumerate(self.tail_history):
            if len(trail) < 2:
                continue
            raw_color = colors[:, idx] if colors.shape[1] > idx else np.array([180, 180, 180])
            base_color = tuple(int(max(0, min(255, c))) for c in raw_color)
            local_points = [(p[0] - rect.left, p[1] - rect.top) for p in trail]
            pygame.draw.lines(
                overlay,
                (*base_color, 130),
                False,
                local_points,
                max(1, int(radius_px * 0.6)),
            )
        self.screen.blit(overlay, rect.topleft)

    def get_closest_particle_index(self, point: tuple[int, int], *, max_distance: float | None = None) -> Optional[int]:
        """Return the index of the particle nearest to the supplied screen point."""
        positions = self._last_particle_screen_positions
        if positions is None:
            r = np.array(self.simulation.r, copy=True)
            if r.shape[1] == 0:
                return None
            self._project_particle_positions(r)
            positions = self._last_particle_screen_positions
        if positions is None or positions.shape[1] == 0:
            return None
        dx = positions[0] - point[0]
        dy = positions[1] - point[1]
        distances_sq = dx * dx + dy * dy
        idx = int(np.argmin(distances_sq))
        if max_distance is None:
            return idx
        if distances_sq[idx] <= max_distance * max_distance:
            return idx
        return None

    def select_particle_for_tracking(self, point: tuple[int, int], *, max_distance: float | None = None) -> bool:
        """
        Choose the particle closest to `point` for tracking, optionally requiring proximity.
        """
        idx = self.get_closest_particle_index(point, max_distance=max_distance)
        if idx is None:
            return False
        if self.tracked_particle_id == idx:
            return True
        self.tracked_particle_id = idx
        if self.focus_active:
            self.focus_particle_idx = idx
        if self.trail_points:
            self.trail_points.clear()
        return True

    def _store_flux_sample(self, timestamp: float, flux_raw: float, flux_avg: float, cumulative: float) -> None:
        """Cache the latest heat-flux values for future visualisation."""
        if not math.isfinite(timestamp):
            return
        if not math.isfinite(flux_raw) or not math.isfinite(flux_avg) or not math.isfinite(cumulative):
            return
        if self.midplane_flux_samples and abs(self.midplane_flux_samples[-1][0] - timestamp) < 1e-9:
            self.midplane_flux_samples[-1] = (timestamp, flux_raw, flux_avg, cumulative)
        else:
            self.midplane_flux_samples.append((timestamp, flux_raw, flux_avg, cumulative))
        if len(self.midplane_flux_samples) > self.max_flux_samples:
            self.midplane_flux_samples.pop(0)

        window_span = getattr(self, 'flux_time_window', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_span = float(window_span)
        except (TypeError, ValueError):
            window_span = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_span) or window_span <= 0.0:
            window_span = FLUX_GRAPH_TIME_WINDOW

        sim_window = None
        if self.simulation is not None:
            try:
                sim_window = float(self.simulation.get_heat_cumulative_span())
            except (TypeError, ValueError):
                sim_window = None
        if sim_window is not None and math.isfinite(sim_window) and sim_window > 0.0:
            window_span = max(window_span, sim_window)

        cutoff = timestamp - window_span
        while self.midplane_flux_samples and self.midplane_flux_samples[0][0] < cutoff:
            self.midplane_flux_samples.pop(0)

    def _draw_midplane_wall(self) -> None:
        """Render a semi-transparent dashed divider at the domain centre."""
        color = MIDPLANE_WALL_COLOR
        ratio = MIDPLANE_WALL_THICKNESS_RATIO
        rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        box_width_units = max(self.simulation.get_box_width(), 1e-6)
        plane_coord = self.simulation.get_midplane_position()
        if self.midplane_axis == 'y':
            wall_height = max(1, int(round(rect.height * ratio)))
            dash_length = max(10, int(round(self.sim_width_px * 0.06)))
            gap_length = max(6, int(round(dash_length * 0.6)))
            wall_surface = pygame.Surface((self.sim_width_px, wall_height), pygame.SRCALPHA)
            center_y = wall_height // 2
            x = 0
            while x < self.sim_width_px:
                end_x = min(self.sim_width_px, x + dash_length)
                pygame.draw.line(wall_surface, color, (x, center_y), (end_x, center_y), wall_height)
                x = end_x + gap_length
            norm = float(np.clip(plane_coord, self.simulation.R, 1.0 - self.simulation.R))
            wall_y = rect.top + rect.height - int(round(rect.height * norm)) - wall_height // 2
            wall_y = max(rect.top, min(rect.bottom - wall_height, wall_y))
            self.screen.blit(wall_surface, (self.sim_left_px, wall_y))
        else:
            wall_width = max(1, int(round(rect.width * ratio)))
            dash_length = max(10, int(round(rect.height * 0.06)))
            gap_length = max(6, int(round(dash_length * 0.6)))
            wall_surface = pygame.Surface((wall_width, rect.height), pygame.SRCALPHA)
            center_x = wall_width // 2
            y = 0
            while y < rect.height:
                end_y = min(rect.height, y + dash_length)
                pygame.draw.line(wall_surface, color, (center_x, y), (center_x, end_y), wall_width)
                y = end_y + gap_length
            norm = float(np.clip(plane_coord / box_width_units, 0.0, 1.0))
            wall_x = self.sim_left_px + int(round(self.sim_width_px * norm)) - wall_width // 2
            wall_x = max(self.sim_left_px, min(self.sim_left_px + self.sim_width_px - wall_width, wall_x))
            self.screen.blit(wall_surface, (wall_x, rect.top))

    def set_params(self, params, par):
        # Dispatch updated simulation parameters based on the changed
        # parameter name.  Legacy parameters such as ``gamma`` and ``k``
        # are no longer processed, because they have no effect in the
        # current model.  Updates to particle size are handled in the
        # DemoScreen via ``update_radius_scale``.
        if par == 'T':
            self.simulation.set_params(T=params['T'])
        elif par == 'r':
            self.simulation.set_params(particles_cnt=params['r'])
            self._sync_tagged_indices(self.simulation._n_particles)
        elif par == 'speed':
            self.update_speed_factor(params.get('speed', self.speed_factor), force=True)
        elif par == 'wind_speed':
            self.simulation.set_wind_speed(params.get('wind_speed', self.wind_speed))
            self.wind_speed = params.get('wind_speed', self.wind_speed)
        elif par == 'B_background':
            val = params.get('B_background', self.params.get('B_background', 0.8))
            try:
                val = float(val)
            except Exception:
                val = self.params.get('B_background', 0.8)
            self.params['B_background'] = val
            self.simulation.set_field_range(val, val)
            self.b_min = val
            self.b_max = val
        elif par in (
            'foot_radius',
            'foot_distance',
            'foot_offset',
            'foot_asymmetry',
            'B_left',
            'B_right',
            'B0',
            'B_grad_x',
            'B_grad_y',
            'transition_width',
            'kperp_scale',
            'kperp_contrast',
            'dmm_min',
            'dmm_max',
            'dmm_dependence',
            'T_left',
            'T_right',
            'T_background',
            'field_lines_enabled',
            'field_line_count',
            'field_line_thickness',
            'field_line_spread',
            'heatmap_mode',
            'tails_enabled',
            'lambda_param',
        ):
            self.params[par] = params.get(par, self.params.get(par))
            if par == 'lambda_param':
                self.simulation.set_lambda_param(self.params[par])
            if par in ('foot_radius', 'foot_distance', 'foot_offset', 'foot_asymmetry', 'B_left', 'B_right', 'B0'):
                self._apply_absorber_to_sim()
            self._invalidate_heatmap()
        # ignore any other parameters (gamma, k, R, etc.)

    @staticmethod
    def _values_differ(old_val, new_val) -> bool:
        try:
            return abs(float(new_val) - float(old_val)) > 1e-6
        except Exception:
            return new_val != old_val

    def _cleanup_pending_foot_params(self) -> None:
        for key in list(self._pending_foot_params.keys()):
            if not self._values_differ(self.params.get(key), self._pending_foot_params.get(key)):
                self._pending_foot_params.pop(key, None)

    def has_pending_foot_parameters(self) -> bool:
        self._cleanup_pending_foot_params()
        return bool(self._pending_foot_params)

    def apply_pending_foot_parameters(self) -> bool:
        if not self._pending_foot_params:
            return False
        updates = dict(self._pending_foot_params)
        self._pending_foot_params.clear()
        applied = False
        for key, value in updates.items():
            if not self._values_differ(self.params.get(key), value):
                continue
            self.set_params({key: value}, key)
            self.params[key] = value
            applied = True
        if applied:
            self._invalidate_field_lines()
        return applied

    def _apply_slider_params(self, slider_params: dict) -> bool:
        """Apply any slider-driven parameter changes immediately."""
        changed = False
        pending_keys = FIELD_LINE_PARAM_KEYS
        field_lines_active = bool(self.params.get('field_lines_enabled', False))
        for key, new_val in slider_params.items():
            if key in ('particles_to_add', 'field_lines_enabled'):
                continue
            old_val = self.params.get(key)
            if not self._values_differ(old_val, new_val):
                if key in pending_keys:
                    self._pending_foot_params.pop(key, None)
                continue
            if field_lines_active and key in pending_keys:
                self._pending_foot_params[key] = new_val
                continue
            if key == 'size_scale':
                try:
                    self.update_radius_scale(float(new_val))
                except Exception:
                    pass
                changed = True
                self.params[key] = new_val
            else:
                self.set_params({key: new_val}, key)
                self.params[key] = new_val
                changed = True
        if changed:
            self._invalidate_field_lines()
        return changed

    def draw_check(self, params):
        # Draw background with field gradient and wind arrows
        self._draw_field_background()

        slider_params = params.get('params', {})
        applied_changes = self._apply_slider_params(slider_params)
        params['is_changed'] = applied_changes

        # Advance simulation and record energies
        loader = config.ConfigLoader()
        speed_raw = params['params'].get('speed', self.speed_factor)
        self.update_speed_factor(speed_raw)
        params['params']['speed'] = self.speed_factor
        steps = self._speed_steps

        new_args = None
        for i in range(steps):
            new_args = next(self.simulation)
            self._update_wall_hit_counters()
            if i < len(params['kinetic']):
                params['kinetic'][i] = self.simulation.calc_kinetic_energy()
                params['potential'][i] = self.simulation.calc_potential_energy()
                params['mean_kinetic'][i] = self.simulation.mean_kinetic_energy(loader['sim_avg_frames_c'])
                params['mean_potential'][i] = self.simulation.mean_potential_energy(loader['sim_avg_frames_c'])
        for i in range(steps, len(params['kinetic'])):
            params['kinetic'][i] = -1
            params['potential'][i] = -1
            params['mean_kinetic'][i] = -1
            params['mean_potential'][i] = -1

        if new_args is None:
            new_args = (
                self.simulation.r,
                self.simulation.r_spring,
                self.simulation.v,
                self.simulation.v_spring,
                0.0,
            )

        # Unpack positions; r_spring is empty
        r = np.array(new_args[0], copy=True)
        self._ensure_focus_particle(r.shape[1])
        focus_index = self.focus_particle_idx if self.focus_active else None

        # Particle colours come from their species definitions
        colors = self.simulation.get_particle_colors()

        # Convert positions and radius to screen-space once so they can be reused.
        r, r_radius, x_scale, y_scale = self._project_particle_positions(r)

        # Collect decay events and convert to screen bursts
        self._collect_decay_events(x_scale, y_scale)

        # Store the current position of the tracked particle for the trail feature
        self._record_trail_point(r)
        self._record_tail_positions(r)
        # Precompute sets for tagged particles and highlight handling
        tagged_set = set(self.tagged_indices)

        if self.params.get('tails_enabled', False):
            self._draw_tails(colors, self.main, r_radius)

        # Draw gas particles with dimming/highlighting options
        for idx in range(r.shape[1]):
            point = (int(r[0, idx]), int(r[1, idx]))
            if idx in tagged_set:
                color = self.tagged_color
            elif self.dim_untracked:
                color = self.dim_color
            else:
                raw_color = colors[:, idx] if colors.shape[1] > idx else np.array([200, 200, 200])
                color = tuple(int(max(0, min(255, c))) for c in raw_color)
            if focus_index is not None and idx != focus_index:
                color = self._focus_dim_color(color)
            pygame.draw.circle(self.screen, color, point, r_radius)

        # Render decay bursts on top
        self._render_decay_bursts()

        # Draw trajectory trail over the particles so it remains visible
        if self.trail_enabled and len(self.trail_points) >= 2:
            pygame.draw.lines(self.screen, self.tagged_color, False, self.trail_points, 2)

        if self.trail_enabled:
            self._ensure_tracked_particle()
        if self.tracked_particle_id is not None and self.tracked_particle_id < r.shape[1]:
            tracked_point = (int(r[0, self.tracked_particle_id]), int(r[1, self.tracked_particle_id]))
            pygame.draw.circle(self.screen, (255, 255, 255), tracked_point, r_radius + 3, 2)
        focus_point: tuple[int, int] | None = None
        if focus_index is not None and focus_index < r.shape[1]:
            focus_point = (int(r[0, focus_index]), int(r[1, focus_index]))
            pygame.draw.circle(self.screen, self.focus_field_color, focus_point, r_radius + 3, 2)
            self._draw_focus_vectors(focus_point)

        # Draw border
        inner_border = 3
        mask_border = 50
        pygame.draw.rect(
            self.screen,
            self.bg_screen_color,
            (
                self.position[0] - mask_border,
                self.position[1] - mask_border,
                self.width + mask_border * 2,
                self.height + mask_border * 2,
            ),
            mask_border,
        )
        pygame.draw.rect(
            self.screen,
            self.bd_color,
            (
                self.position[0] - inner_border,
                self.position[1] - inner_border,
                self.width + inner_border * 2,
                self.height + inner_border * 2,
            ),
            inner_border,
        )

    def _draw_focus_vectors(self, focus_point: tuple[int, int]) -> None:
        """Draw field and wind influence vectors anchored at the focus particle."""
        wind_speed = max(0.0, self.simulation.get_wind_speed())
        field_strength = self.simulation.get_field_strength()
        focus_index = self.focus_particle_idx
        if focus_index is None:
            return

        grad_x = 0.0
        grad_y = 0.0
        try:
            b_local, grad_x_arr, grad_y_arr = self.simulation.get_local_field_profile()
            if 0 <= focus_index < b_local.size:
                field_strength = float(b_local[focus_index])
                grad_x = float(grad_x_arr[focus_index])
                grad_y = float(grad_y_arr[focus_index])
        except Exception:
            pass

        field_strength = max(0.0, field_strength)

        # Determine charge sign for the focused particle to flip the magnetic force
        charge_sign = 0.0
        try:
            types = self.simulation.get_particle_types()
            if focus_index is not None and focus_index < len(types):
                type_id = int(types[focus_index])
                defs = self.simulation.get_type_definitions()
                if 0 <= type_id < len(defs):
                    charge_sign = float(defs[type_id].charge_sign)
        except Exception:
            charge_sign = 0.0

        diag = max(1, min(self.sim_width_px, self.height))
        base_length = diag * 0.07
        base_angle = self.simulation.get_field_direction()
        base_vec_x = math.cos(base_angle) * field_strength
        base_vec_y = math.sin(base_angle) * field_strength
        grad_radius = max(self.simulation.get_footpoint_field_radius(), 0.05)
        grad_vec_x = -grad_x * grad_radius
        grad_vec_y = -grad_y * grad_radius
        combined_x = base_vec_x + grad_vec_x
        combined_y = base_vec_y + grad_vec_y
        arrow_mag = math.hypot(combined_x, combined_y)
        if arrow_mag <= 1e-8:
            field_angle = base_angle
            arrow_mag = field_strength
        else:
            field_angle = math.atan2(combined_y, combined_x)
        if charge_sign < 0:
            field_angle += math.pi
        effective_field = max(field_strength, arrow_mag)
        if effective_field <= 0.0 and wind_speed <= 0.0:
            return

        field_length = max(18, min(140, int(base_length + effective_field * 60)))
        wind_length = max(16, min(120, int(base_length + wind_speed * 70)))
        wind_angle = self.simulation.get_wind_direction()

        field_end = self._vector_endpoint(focus_point, field_angle, field_length)
        wind_end = self._vector_endpoint(focus_point, wind_angle, wind_length)
        self._draw_vector_arrow(focus_point, field_end, self.focus_field_color)
        self._draw_vector_arrow(focus_point, wind_end, self.focus_wind_color)

    def _vector_endpoint(self, start: tuple[int, int], angle: float, length: float) -> tuple[float, float]:
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length
        return (start[0] + dx, start[1] - dy)

    def _draw_vector_arrow(self, start: tuple[int, int], end: tuple[float, float], color: tuple[int, int, int]) -> None:
        pygame.draw.line(self.screen, color, start, end, 3)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 6:
            return
        ux = dx / distance
        uy = dy / distance
        perp_x = -uy
        perp_y = ux
        head_length = min(14, max(6, int(distance * 0.25)))
        head_width = max(3, int(head_length * 0.35))
        left = (
            end[0] - ux * head_length + perp_x * head_width,
            end[1] - uy * head_length + perp_y * head_width,
        )
        right = (
            end[0] - ux * head_length - perp_x * head_width,
            end[1] - uy * head_length - perp_y * head_width,
        )
        arrow_points = [end, left, right]
        pygame.draw.polygon(
            self.screen,
            color,
            [(int(round(x)), int(round(y))) for x, y in arrow_points],
        )

    def _configure_flux_graph(self, loader: config.ConfigLoader) -> None:
        """Initialise fixed axis limits and grid styling for the flux chart."""
        config_dict = getattr(loader, '_loader', {})
        graph_cfg = {}
        if isinstance(config_dict, dict):
            graph_cfg = config_dict.get('flux_graph', {}) or {}

        scale_value = graph_cfg.get('display_scale', FLUX_GRAPH_DISPLAY_SCALE)
        try:
            display_scale = float(scale_value)
        except (TypeError, ValueError):
            display_scale = FLUX_GRAPH_DISPLAY_SCALE
        if not math.isfinite(display_scale) or display_scale <= 0.0:
            display_scale = 1.0
        self.flux_display_scale = display_scale

        axis_limits = graph_cfg.get('axis_limits')
        min_limit = max_limit = None
        if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
            try:
                min_limit = float(axis_limits[0])
                max_limit = float(axis_limits[1])
            except (TypeError, ValueError):
                min_limit = max_limit = None
        axis_limits_specified = 'axis_limits' in graph_cfg or 'axis_limit' in graph_cfg

        needs_default = (
            min_limit is None
            or max_limit is None
            or not math.isfinite(min_limit)
            or not math.isfinite(max_limit)
            or max_limit <= min_limit
        )
        if needs_default:
            axis_limit_value = graph_cfg.get('axis_limit', FLUX_GRAPH_DEFAULT_HALF_RANGE)
            try:
                half_range = abs(float(axis_limit_value))
            except (TypeError, ValueError):
                half_range = FLUX_GRAPH_DEFAULT_HALF_RANGE
            if not math.isfinite(half_range) or half_range <= 0.0:
                half_range = FLUX_GRAPH_DEFAULT_HALF_RANGE
            min_limit = -half_range
            max_limit = half_range

        self.flux_axis_min: float = float(min_limit)
        self.flux_axis_max: float = float(max_limit)
        if self.flux_axis_max <= self.flux_axis_min:
            half_range = max(abs(self.flux_axis_min), abs(self.flux_axis_max), FLUX_GRAPH_DEFAULT_HALF_RANGE)
            self.flux_axis_min = -half_range
            self.flux_axis_max = half_range
        self.flux_axis_span: float = self.flux_axis_max - self.flux_axis_min
        if self.flux_axis_span <= 0.0:
            self.flux_axis_min = -FLUX_GRAPH_DEFAULT_HALF_RANGE
            self.flux_axis_max = FLUX_GRAPH_DEFAULT_HALF_RANGE
            self.flux_axis_span = self.flux_axis_max - self.flux_axis_min
        self.flux_axis_default_half_range: float = max(
            abs(self.flux_axis_min),
            abs(self.flux_axis_max),
            FLUX_GRAPH_DEFAULT_HALF_RANGE,
        )
        if axis_limits_specified:
            self.flux_axis_fixed_limits = (self.flux_axis_min, self.flux_axis_max)
        else:
            self.flux_axis_fixed_limits = None

        grid_lines_value = graph_cfg.get('grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES)
        try:
            grid_lines = int(grid_lines_value)
        except (TypeError, ValueError):
            grid_lines = FLUX_GRAPH_DEFAULT_GRID_LINES
        self.flux_grid_lines: int = max(2, grid_lines)

        grid_color_value = graph_cfg.get('grid_color')
        color = FLUX_GRID_COLOR
        if isinstance(grid_color_value, (list, tuple)) and len(grid_color_value) == 3:
            try:
                rgb = tuple(int(max(0, min(255, component))) for component in grid_color_value)
                if len(rgb) == 3:
                    color = rgb
            except (TypeError, ValueError):
                color = FLUX_GRID_COLOR
        self.flux_grid_color: tuple[int, int, int] = color

        follow_window_value = graph_cfg.get('follow_window_samples', FLUX_GRAPH_FOLLOW_WINDOW)
        try:
            follow_window = int(follow_window_value)
        except (TypeError, ValueError):
            follow_window = FLUX_GRAPH_FOLLOW_WINDOW
        self.flux_follow_window_samples: int = max(8, follow_window)

        padding_value = graph_cfg.get('follow_padding_fraction', FLUX_GRAPH_PADDING_FRACTION)
        try:
            padding_fraction = float(padding_value)
        except (TypeError, ValueError):
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION
        if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION
        self.flux_follow_padding: float = padding_fraction

        extra_padding_value = graph_cfg.get('extra_padding_fraction', FLUX_GRAPH_EXTRA_PADDING)
        try:
            extra_padding_fraction = float(extra_padding_value)
        except (TypeError, ValueError):
            extra_padding_fraction = FLUX_GRAPH_EXTRA_PADDING
        if not math.isfinite(extra_padding_fraction) or extra_padding_fraction < 0.0:
            extra_padding_fraction = FLUX_GRAPH_EXTRA_PADDING
        self.flux_axis_extra_padding: float = extra_padding_fraction

        tick_step_value = None
        tick_step_in_config = 'tick_step' in graph_cfg
        if tick_step_in_config:
            tick_step_value = graph_cfg.get('tick_step')
            try:
                tick_step_value = float(tick_step_value)
            except (TypeError, ValueError):
                tick_step_value = None
            if tick_step_value is not None and (not math.isfinite(tick_step_value) or tick_step_value <= 0.0):
                tick_step_value = None
        # Only store a fixed tick-step if the user asked for it; otherwise pick it dynamically per frame.
        self.flux_tick_step: float | None = tick_step_value
        window_value = graph_cfg.get('time_window_seconds', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_seconds = float(window_value)
        except (TypeError, ValueError):
            window_seconds = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_seconds) or window_seconds <= 0.0:
            window_seconds = FLUX_GRAPH_TIME_WINDOW
        self.flux_time_window: float = window_seconds

    def _flux_zoom_factor(self) -> float:
        """Return a multiplier (<1 zooms in) based on current particle count."""
        count = None
        if getattr(self, 'params', None):
            count = self.params.get('r')
        if count is None and getattr(self, 'simulation', None):
            count = getattr(self.simulation, '_n_particles', None)
        try:
            count = int(count)
        except (TypeError, ValueError):
            return 1.0
        if count <= 0:
            return 1.0
        ref = FLUX_GRAPH_PARTICLE_REF
        min_zoom = FLUX_GRAPH_MIN_ZOOM
        zoom = math.sqrt(count / ref) if ref > 0 else 1.0
        return min(1.0, max(min_zoom, zoom))

    def _compute_dynamic_flux_limits(self, values: list[float]) -> tuple[float, float]:
        """Return adaptive axis limits that track the recent flux signal."""
        zoom = self._flux_zoom_factor()
        base_half_range = getattr(self, 'flux_axis_default_half_range', FLUX_GRAPH_DEFAULT_HALF_RANGE) * zoom
        try:
            follow_window = int(getattr(self, 'flux_follow_window_samples', FLUX_GRAPH_FOLLOW_WINDOW))
        except (TypeError, ValueError):
            follow_window = FLUX_GRAPH_FOLLOW_WINDOW
        follow_window = max(8, follow_window)
        subset = values[-follow_window:] if len(values) > follow_window else list(values)
        if not subset:
            return -base_half_range, base_half_range

        local_min = min(subset)
        local_max = max(subset)
        if not (math.isfinite(local_min) and math.isfinite(local_max)):
            return -base_half_range, base_half_range
        if local_max < local_min:
            local_min, local_max = local_max, local_min

        grid_lines = max(2, getattr(self, 'flux_grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES))
        configured_step = getattr(self, 'flux_tick_step', None)
        if configured_step is not None and math.isfinite(configured_step) and configured_step > 0.0:
            step = configured_step
        else:
            span_from_data = local_max - local_min
            step = span_from_data / grid_lines if grid_lines else span_from_data
            if not math.isfinite(step) or step <= 0.0:
                step = base_half_range / max(1, grid_lines)

        padding_fraction = getattr(self, 'flux_follow_padding', FLUX_GRAPH_PADDING_FRACTION)
        if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION

        span = max(local_max - local_min, 0.0)
        padding = span * padding_fraction
        if step > 0.0:
            padding = max(padding, step * 0.5)
        elif padding <= 0.0:
            padding = base_half_range * 0.1

        desired_min = local_min - padding
        desired_max = local_max + padding
        if not math.isfinite(desired_min) or not math.isfinite(desired_max):
            return -base_half_range, base_half_range
        if desired_max <= desired_min:
            epsilon = step if step > 0.0 else base_half_range
            desired_min -= epsilon
            desired_max += epsilon

        def align_down(value: float) -> float:
            if step <= 0.0:
                return value
            return math.floor(value / step) * step

        def align_up(value: float) -> float:
            if step <= 0.0:
                return value
            return math.ceil(value / step) * step

        axis_min = align_down(desired_min)
        axis_max = align_up(desired_max)
        try:
            negative_fraction = float(FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION)
        except (TypeError, ValueError):
            negative_fraction = FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION
        if not math.isfinite(negative_fraction) or negative_fraction < 0.0:
            negative_fraction = FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION
        if axis_max > 0.0:
            allowed_negative = -axis_max * negative_fraction
            if local_min >= allowed_negative:
                axis_min = max(axis_min, allowed_negative)

        extra_fraction = getattr(self, 'flux_axis_extra_padding', FLUX_GRAPH_EXTRA_PADDING)
        try:
            extra_fraction = float(extra_fraction)
        except (TypeError, ValueError):
            extra_fraction = FLUX_GRAPH_EXTRA_PADDING
        if not math.isfinite(extra_fraction) or extra_fraction < 0.0:
            extra_fraction = FLUX_GRAPH_EXTRA_PADDING
        axis_span = axis_max - axis_min
        extra_margin = 0.0
        if math.isfinite(axis_span) and axis_span > 0.0:
            extra_margin = axis_span * extra_fraction
            if step > 0.0:
                extra_margin = max(extra_margin, step * 0.5)
        if extra_margin > 0.0 and math.isfinite(extra_margin):
            axis_min -= extra_margin
            axis_max += extra_margin

        # Ensure the zero level is always visible on the vertical axis.
        if axis_max < 0.0:
            axis_max = 0.0
        if axis_min > 0.0:
            axis_min = 0.0
        if axis_max <= axis_min:
            epsilon = step if step > 0.0 else base_half_range
            axis_min -= epsilon
            axis_max += epsilon

        if not math.isfinite(axis_min) or not math.isfinite(axis_max) or axis_max <= axis_min:
            half_range = base_half_range
            axis_min = -half_range
            axis_max = half_range
        return axis_min, axis_max

    def draw_midplane_flux_graph(
        self,
        target_surface: pygame.Surface,
        rect: pygame.Rect,
        samples: Optional[list[tuple[float, ...]]] = None,
        line_color: tuple[int, int, int] = (90, 180, 255),
        baseline_color: tuple[int, int, int] = (180, 180, 180),
        background: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0),
        series: str = 'avg',
        highlight_bounds: bool = True,
        highlight_color: tuple[int, int, int] = (192, 80, 80),
    ) -> Optional[tuple[float, float]]:
        """
        Render a step-style graph of the recorded midplane heat quantities.

        The function draws onto ``target_surface`` but does not blit the
        result to the on-screen display.  This lets callers integrate the
        graph into their own layout later.

        Returns
        -------
        Optional[Tuple[float, float]]
            The ``(min, max)`` flux values plotted on the vertical axis.
        """
        self.last_flux_limits = None
        data = self.midplane_flux_samples if samples is None else samples
        rect = pygame.Rect(rect)
        if len(data) < 2 or rect.width <= 1 or rect.height <= 1:
            return None

        original_series = series
        series_key = series.lower().strip()
        idx_map = {
            'raw': 1,
            'raw_flux': 1,
            'avg': 2,
            'average': 2,
            'flux': 2,
            'cumulative': 3,
            'heat': 3,
            'total': 3,
            'heat_total': 3,
            'cumulative_heat': 3,
        }
        idx = idx_map.get(series_key, idx_map.get(original_series, 2))
        scale = getattr(self, 'flux_display_scale', 1.0)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        times: list[float] = []
        values: list[float] = []
        for sample in data:
            if not sample:
                continue
            t = float(sample[0])
            value_idx = min(idx, len(sample) - 1)
            if value_idx <= 0 or value_idx >= len(sample):
                continue
            val = float(sample[value_idx]) * scale
            times.append(t)
            values.append(val)

        if len(times) < 2:
            return None

        window_span = getattr(self, 'flux_time_window', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_span = float(window_span)
        except (TypeError, ValueError):
            window_span = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_span) or window_span <= 0.0:
            window_span = FLUX_GRAPH_TIME_WINDOW

        latest_time = times[-1]
        window_start = latest_time - window_span
        filtered_times: list[float] = []
        filtered_values: list[float] = []
        prev_time = None
        prev_value = None
        for t, v in zip(times, values):
            if t < window_start:
                prev_time = t
                prev_value = v
                continue
            if not filtered_times and prev_time is not None and prev_time < window_start and t > window_start:
                filtered_times.append(window_start)
                filtered_values.append(prev_value)
            filtered_times.append(t)
            filtered_values.append(v)

        if len(filtered_times) < 2:
            return None

        if filtered_times and filtered_times[0] > window_start:
            window_start = filtered_times[0]

        times = [t - window_start for t in filtered_times]
        values = filtered_values

        t_min = 0.0
        t_max = times[-1]
        if not math.isfinite(t_max) or t_max <= t_min:
            return None

        actual_min = min(values)
        actual_max = max(values)
        if not (math.isfinite(actual_min) and math.isfinite(actual_max)):
            return None

        time_span = t_max - t_min
        if time_span <= 0.0 or not math.isfinite(time_span):
            return None

        fixed_limits = getattr(self, 'flux_axis_fixed_limits', None)
        axis_min = axis_max = None
        if isinstance(fixed_limits, (list, tuple)) and len(fixed_limits) == 2:
            try:
                axis_min = float(fixed_limits[0])
                axis_max = float(fixed_limits[1])
            except (TypeError, ValueError):
                axis_min = axis_max = None
        if (
            axis_min is None
            or axis_max is None
            or not (math.isfinite(axis_min) and math.isfinite(axis_max))
            or axis_max <= axis_min
        ):
            axis_min, axis_max = self._compute_dynamic_flux_limits(values)

        if axis_max < 0.0:
            axis_max = 0.0
        if axis_min > 0.0:
            axis_min = 0.0

        # Keep the recent extrema visible even if fixed limits were set.
        span_for_padding = max(1.0, abs(axis_max - axis_min))
        pad = span_for_padding * 0.02
        axis_min = min(axis_min, actual_min - pad)
        axis_max = max(axis_max, actual_max + pad)

        if axis_max <= axis_min:
            spread = max(1.0, abs(actual_min), abs(actual_max))
            axis_min = -spread
            axis_max = spread

        axis_span = axis_max - axis_min
        if not math.isfinite(axis_span) or axis_span <= 0.0:
            half_range = getattr(self, 'flux_axis_default_half_range', FLUX_GRAPH_DEFAULT_HALF_RANGE)
            axis_min = -half_range
            axis_max = half_range
            axis_span = axis_max - axis_min

        self.last_flux_limits = (axis_min, axis_max)

        graph_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        if background is not None:
            graph_surface.fill(background)

        grid_lines = max(2, getattr(self, 'flux_grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES))
        grid_color = getattr(self, 'flux_grid_color', FLUX_GRID_COLOR)
        for idx in range(grid_lines + 1):
            frac = idx / grid_lines
            value = axis_min + axis_span * frac
            y_rel = (value - axis_min) / axis_span
            y = rect.height - y_rel * rect.height
            pygame.draw.line(
                graph_surface,
                grid_color,
                (0, int(round(y))),
                (rect.width, int(round(y))),
                1,
            )

        def value_to_y(value: float) -> int:
            y_rel = (value - axis_min) / axis_span
            return int(round(rect.height - y_rel * rect.height))

        window_min = min(values)
        window_max = max(values)
        if highlight_bounds:
            min_y = value_to_y(window_min)
            max_y = value_to_y(window_max)
            pygame.draw.line(graph_surface, highlight_color, (0, min_y), (rect.width, min_y), 2)
            if window_max != window_min:
                pygame.draw.line(graph_surface, highlight_color, (0, max_y), (rect.width, max_y), 2)

        if axis_min <= 0.0 <= axis_max:
            zero_y = value_to_y(0.0)
            zero_color = (0, 0, 0)
            pygame.draw.line(
                graph_surface,
                zero_color,
                (0, zero_y),
                (rect.width, zero_y),
                3,
            )

        def clamp_value(value: float) -> float:
            if not math.isfinite(value):
                return max(axis_min, min(axis_max, 0.0))
            return max(axis_min, min(axis_max, value))

        def to_point(time_value: float, flux_value: float) -> tuple[int, int]:
            x_rel = (time_value - t_min) / time_span
            y_rel = (flux_value - axis_min) / axis_span
            x = x_rel * rect.width
            y = rect.height - y_rel * rect.height
            return int(round(max(0.0, min(rect.width, x)))), int(round(max(0.0, min(rect.height, y))))

        points: list[tuple[int, int]] = []
        prev_time = times[0]
        prev_val = clamp_value(values[0])
        points.append(to_point(prev_time, prev_val))
        for time_value, flux_value in zip(times[1:], values[1:]):
            clamped = clamp_value(flux_value)
            points.append(to_point(time_value, prev_val))
            points.append(to_point(time_value, clamped))
            prev_time = time_value
            prev_val = clamped

        if len(points) >= 2:
            pygame.draw.lines(graph_surface, line_color, False, points, 2)

        target_surface.blit(graph_surface, rect.topleft)
        return self.last_flux_limits

    # -----------------------------------------------------------------
    def add_tagged_particles(self, count: int) -> int:
        """
        Add a specified number of tagged (coloured) particles at the
        centre of the box.

        The new particles are placed at the centre of the simulation
        domain (``x = 0.5``, ``y = 0.5``) with random jitter to avoid
        immediate overlap.  They start stationary so that their motion
        is driven only by the fields once introduced.  The indices of
        these particles are recorded in ``self.tagged_indices`` so that
        they can be drawn in a different colour.

        Parameters
        ----------
        count : int
            Number of particles to add.

        Returns
        -------
        int
            Actual number of tagged particles appended to the simulation.
        """
        count = max(0, int(count))
        if count <= 0:
            return 0
        # Determine the starting index for new particles
        n_old = int(self.simulation._n_particles)
        self.reset_wall_hit_counters()
        # Build positions at the box centre with small random jitter
        jitter = 0.001  # small displacement to avoid stacking
        # Uniformly distribute jitter within a tiny square around centre
        box_width = max(self.simulation.get_box_width(), 1e-9)
        center_x = box_width / 2.0
        r_new = np.tile(np.array([[center_x], [0.5]]), (1, count))
        if count > 0:
            r_new = r_new + (np.random.uniform(low=-jitter, high=jitter, size=(2, count)))
        # Ensure the new particles respect the walls (stay within [R, 1-R])
        # Clip in case jitter pushes them outside
        R_phys = self.simulation.R
        box_width = max(self.simulation.get_box_width(), 1e-9)
        r_new[0] = np.clip(r_new[0], R_phys, box_width - R_phys)
        r_new[1] = np.clip(r_new[1], R_phys, 1.0 - R_phys)
        masses = self.simulation.m
        if masses.size > 0:
            m_typ = np.median(masses)
        else:
            m_typ = 1.0
        v_new = np.zeros((2, count), dtype=float)
        # Masses for new particles
        m_new = np.full((count,), m_typ)
        # Add to simulation
        self.simulation.add_particles(r_new, v_new, m_new)
        n_new = int(self.simulation._n_particles)
        actual_added = max(0, n_new - n_old)
        if actual_added <= 0:
            return 0
        # Record tagged indices
        new_indices = list(range(n_old, n_old + actual_added))
        self.tagged_indices.extend(new_indices)
        self._ensure_tracked_particle()
        if self.trail_enabled:
            self.trail_points.clear()
        # Update parameter dictionary to reflect increased number of particles
        self.params['r'] = n_new
        return actual_added
