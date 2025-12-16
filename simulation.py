"""
Simulation of particles on a 2D periodic torus.

This module defines a Simulation class that models the motion of many
identical spherical particles in a unit square with periodic boundaries
along both axes.  The particles move ballistically until they collide
with one another; leaving one side of the domain wraps them to the
opposite side with no reflections or wall interactions.

Unlike the original version of this project, there are no "spring"
particles linked by a harmonic potential.  All particles are free
entities.  Functions and state pertaining to the spring particles and
their potential energy have been removed.  To update wall temperatures
at run time (legacy parameter, currently unused), call
``Simulation.set_params`` with the appropriate keywords.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

################################################################################
# Thermal wall utility functions (legacy; temperatures disabled in the cosmic
# ray model but kept for API compatibility and energy calculations).
################################################################################

# Boltzmann constant (scaled to simulation units).  Kept for compatibility with
# the previous temperature-based visualisation; current dynamics ignore wall
# temperatures.
kB: float = 3.0e-6

# Base integration step in simulation time units.  Larger values speed up
# motion on screen but may slightly increase numerical error.
TIME_STEP: float = 6.0e-4

# Duration (seconds) of the rolling window used for cumulative heat plots.
HEAT_CUMULATIVE_WINDOW: float = 5.0
# Inject this many electrons at startup to highlight the negative charge contribution.
INITIAL_EXTRA_ELECTRONS: int = 2

# Create a default random number generator.  If NumPy is unavailable
# ``rng`` will remain ``None`` and fallbacks will be used instead.
try:
    rng: np.random.Generator = np.random.default_rng()
except Exception:
    rng = None

_APP_CONFIG_CACHE: Optional[dict] = None


def _load_app_config_dict() -> Optional[dict]:
    """Return the parsed application config or ``None`` if unavailable."""
    global _APP_CONFIG_CACHE
    if _APP_CONFIG_CACHE is not None:
        return _APP_CONFIG_CACHE

    cfg_path = Path(__file__).resolve().parent / 'config.json'
    if not cfg_path.exists():
        alt = Path.cwd() / 'config.json'
        if alt.exists():
            cfg_path = alt
        else:
            return None
    try:
        with cfg_path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, dict):
        _APP_CONFIG_CACHE = data
        return data
    return None


def _load_particle_type_config() -> dict:
    """Load per-particle-type parameters from ``particle_types.json``.

    Prefers a user-editable copy (same folder as config.json) and falls back to
    the bundled default.  This helps ensure Windows/macOS/Linux builds pick up
    the file even if the working directory differs.
    """
    from paths import resource_file, user_config_path

    user_path = user_config_path().parent / 'particle_types.json'
    default_path = resource_file('particle_types.json')
    cwd_path = Path.cwd() / 'particle_types.json'
    candidates = [user_path, default_path, cwd_path]

    data: dict | None = None
    loaded_from: Optional[Path] = None
    for path in candidates:
        try:
            if path.exists():
                with path.open('r', encoding='utf-8') as fh:
                    maybe = json.load(fh)
                if isinstance(maybe, dict):
                    data = maybe
                    loaded_from = path
                    break
        except Exception:
            continue

    if data is None:
        return {}

    # If we loaded from the bundled default, persist a user copy for easy edits
    if loaded_from == default_path and not user_path.exists():
        try:
            user_path.parent.mkdir(parents=True, exist_ok=True)
            user_path.write_text(default_path.read_text(encoding='utf-8'), encoding='utf-8')
        except Exception:
            pass

    return data


def _parse_hex_color(color: str, fallback: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """Convert a #RRGGBB string into an RGB tuple with clamping."""
    if not isinstance(color, str):
        return fallback
    value = color.strip().lstrip('#')
    if len(value) != 6:
        return fallback
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (
            int(max(0, min(255, r))),
            int(max(0, min(255, g))),
            int(max(0, min(255, b))),
        )
    except Exception:
        return fallback


def _load_magnetic_field_config() -> dict:
    """Load background magnetic field settings and solar wind."""
    data = _load_app_config_dict() or {}
    if not isinstance(data, dict):
        return {}
    return data.get('magnetic_field', {}) or {}


def _load_solar_wind_config() -> dict:
    """Load solar wind parameters from config."""
    data = _load_app_config_dict() or {}
    if not isinstance(data, dict):
        return {}
    return data.get('solar_wind', {}) or {}


def _parse_species_preset(raw: dict) -> list[ParticleType]:
    """Parse a preset file that lists species with weights and metadata."""
    if not isinstance(raw, dict):
        return []
    species_list = raw.get('species')
    if not isinstance(species_list, list):
        return []
    types: list[ParticleType] = []
    for entry in species_list:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name', '') or '')
        if not name:
            continue
        charge = float(entry.get('charge', 0.0))
        mass = float(entry.get('mass', 1.0))
        weight = float(entry.get('weight', 0.0))
        color_hex = entry.get('color', "#cccccc")
        color = _parse_hex_color(color_hex, fallback=(200, 200, 200))
        source = str(entry.get('source', '') or '')
        types.append(
            ParticleType(
                name=name,
                charge_sign=charge,
                diffusion_x=0.0,
                diffusion_y=0.0,
                drift_coeff=0.0,
                lifetime=None,
                color=color,
                mass=mass,
                weight=weight,
                source=source,
            )
        )
    return types


def _load_species_preset() -> list[ParticleType]:
    """Load weighted SEP composition from ``species_presets.json`` when present."""
    from paths import resource_file, user_config_path

    user_path = user_config_path().parent / 'species_presets.json'
    default_path = resource_file('species_presets.json')
    cwd_path = Path.cwd() / 'species_presets.json'
    for path in (user_path, default_path, cwd_path):
        try:
            if path.exists():
                with path.open('r', encoding='utf-8') as fh:
                    data = json.load(fh)
                types = _parse_species_preset(data)
                if types:
                    if path == default_path and not user_path.exists():
                        try:
                            user_path.parent.mkdir(parents=True, exist_ok=True)
                            user_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
                        except Exception:
                            pass
                    return types
        except Exception:
            continue
    return []


def _thermal_speed(T: float, masses: np.ndarray) -> np.ndarray:
    """Return deterministic speeds matching the kinetic energy ``k_B T``."""
    T_value = max(float(T or 0.0), 0.0)
    speeds_sq = np.maximum(2.0 * kB * T_value / masses, 0.0)
    return np.sqrt(speeds_sq)


def _isotropic_vectors_from_speeds(speeds: np.ndarray) -> np.ndarray:
    """Create 2D velocity vectors with random directions and fixed speeds."""
    count = speeds.shape[0]
    if count == 0:
        return np.zeros((2, 0))
    if rng is not None:
        angles = rng.uniform(0.0, 2.0 * math.pi, size=count)
    else:
        angles = np.random.uniform(0.0, 2.0 * math.pi, size=count)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    return np.vstack((speeds * cos_a, speeds * sin_a))

def reflect_from_wall(
    vx: np.ndarray,
    vy: np.ndarray,
    side: str,
    T_wall: float,
    masses: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Apply thermal reflection rules when particles collide with a wall.

    Only the elements selected by ``mask`` are modified.  For the top and
    bottom walls a specular reflection is performed.  For the left and
    right walls, particles keep the incidence/reflection angle while the
    overall kinetic energy is re-initialised to match the wall
    temperature.  Instead of drawing new speeds from a distribution, the
    post-collision speed magnitude is set deterministically from ``k_B T``.

    Parameters
    ----------
    vx, vy: ndarray
        Arrays of x‑ and y‑velocity components for all particles.
    side: str
        One of ``'left'``, ``'right'``, ``'top'`` or ``'bottom'``.
    T_wall: float or None
        Temperature of the wall for diffusive reflection; ignored for
        top and bottom walls.
    masses: ndarray
        Masses of the particles.
    mask: ndarray of bool
        Boolean mask selecting which particles are touching the wall and
        moving toward it.  Only these particles are updated.
    """
    if not np.any(mask):
        return

    if side == 'top' or side == 'bottom':
        # Purely specular: flip the sign of the y‑component
        vy[mask] = -vy[mask]
        return

    # For left and right walls the local outward normal is +x on the left
    # wall and -x on the right wall.
    if side in ('left', 'right'):
        vx_in = vx[mask]
        vy_in = vy[mask]
        if vx_in.size == 0:
            return

        out_sign = 1.0 if side == 'left' else -1.0
        v_spec_normal = np.abs(vx_in)

        if T_wall is None or T_wall <= 0.0:
            vx[mask] = out_sign * v_spec_normal
            vy[mask] = vy_in
            return

        masses_sel = masses[mask]
        # Direction follows the specular reflection law (mirror relative to the normal)
        spec_vec = np.vstack((out_sign * v_spec_normal, vy_in))
        norms = np.linalg.norm(spec_vec, axis=0)
        zero_mask = norms < 1e-12
        if np.any(zero_mask):
            spec_vec[0, zero_mask] = out_sign
            spec_vec[1, zero_mask] = 0.0
            norms[zero_mask] = 1.0
        unit_dir = spec_vec / norms

        speed_samples = _thermal_speed(float(T_wall), masses_sel)

        vx[mask] = unit_dir[0] * speed_samples
        vy[mask] = unit_dir[1] * speed_samples
        return


@dataclass
class ThermalWallConfig:
    """Dataclass storing thermal wall parameters.

    Attributes
    ----------
    T_left, T_right: float
        Temperatures (K) of the left and right walls.  A higher temperature
        results in particles leaving the wall with higher average speed.
    """
    T_left: float = 600.0
    T_right: float = 300.0


@dataclass
class ParticleType:
    """Parameter bundle for a single particle species."""

    name: str
    charge_sign: float
    diffusion_x: float
    diffusion_y: float
    drift_coeff: float
    lifetime: Optional[float]
    color: Tuple[int, int, int]
    mass: Optional[float] = None
    weight: Optional[float] = None
    source: str = ""


def _load_thermal_wall_config() -> ThermalWallConfig:
    """Read default wall temperatures from ``config.json`` if available."""
    data = _load_app_config_dict()
    defaults = ThermalWallConfig()
    if not isinstance(data, dict):
        return defaults

    section = data.get('wall_temperatures')
    if not isinstance(section, dict):
        return defaults

    def _extract(entry, fallback):
        if isinstance(entry, dict):
            value = entry.get('initial', entry.get('value', fallback))
        else:
            value = entry
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    left_entry = section.get('left', section.get('T_left'))
    right_entry = section.get('right', section.get('T_right'))
    return ThermalWallConfig(
        T_left=_extract(left_entry, defaults.T_left),
        T_right=_extract(right_entry, defaults.T_right),
    )


def _parse_particle_types(raw: dict) -> list[ParticleType]:
    """Convert JSON-like mapping into ParticleType records."""
    types: list[ParticleType] = []
    if not isinstance(raw, dict):
        return types
    for name, params in raw.items():
        if not isinstance(params, dict):
            continue
        charge_sign = float(params.get('charge_sign', 0.0))
        diffusion_x = float(params.get('diffusion_x', 0.0))
        diffusion_y = float(params.get('diffusion_y', diffusion_x))
        drift_coeff = float(params.get('drift_coeff', 0.0))
        lifetime_val = params.get('lifetime')
        lifetime = float(lifetime_val) if isinstance(lifetime_val, (int, float)) else None
        color_val = params.get('color', [255, 255, 255])
        if isinstance(color_val, (list, tuple)) and len(color_val) == 3:
            color_tuple = tuple(int(max(0, min(255, c))) for c in color_val)
        else:
            color_tuple = (255, 255, 255)
        mass_val = params.get('mass')
        mass = float(mass_val) if isinstance(mass_val, (int, float)) else None
        weight_val = params.get('weight')
        weight = float(weight_val) if isinstance(weight_val, (int, float)) else None
        source = str(params.get('source', "") or "")
        types.append(
            ParticleType(
                name=str(name),
                charge_sign=charge_sign,
                diffusion_x=diffusion_x,
                diffusion_y=diffusion_y,
                drift_coeff=drift_coeff,
                lifetime=lifetime,
                color=color_tuple,
                mass=mass,
                weight=weight,
                source=source,
            )
        )
    return types


thermal_cfg = _load_thermal_wall_config()


def _load_particle_types_defaults() -> list[ParticleType]:
    """Load particle species definitions from the dedicated config file."""
    preset = _load_species_preset()
    if preset:
        return preset
    raw = _load_particle_type_config()
    if not isinstance(raw, dict):
        return []
    types: list[ParticleType] = _parse_particle_types(raw)
    return types


@dataclass
class HeatFluxConfig:
    """Configuration for midplane heat flux calculations."""

    axis: str = 'x'
    position: float = 0.5
    area: Optional[float] = None
    area_provided: bool = False
    average_window: float = 0.0
    csv_path: Optional[str] = None
    csv_interval: float = 0.0


def _load_heat_flux_config() -> HeatFluxConfig:
    """Load optional heat flux settings from ``config.json`` if available."""
    defaults = HeatFluxConfig()
    data = _load_app_config_dict()
    if not isinstance(data, dict):
        return defaults

    section = data.get('midplane') or data.get('heat_flux') or {}
    if not isinstance(section, dict):
        section = {}

    axis = str(section.get('axis', defaults.axis)).lower()
    if axis not in ('x', 'y'):
        axis = defaults.axis

    position = section.get('position', defaults.position)
    area_provided = False
    area = None
    if 'area' in section:
        area = section.get('area')
        area_provided = True
    elif 'thickness' in section:
        area = section.get('thickness')
        area_provided = True
    average_window = section.get('average_window', section.get('window', defaults.average_window))
    csv_path = section.get('csv_path', defaults.csv_path)
    csv_interval = section.get('csv_interval', defaults.csv_interval)

    try:
        position = float(position)
    except (TypeError, ValueError):
        position = defaults.position
    if area is not None:
        try:
            area = max(float(area), 1e-12)
        except (TypeError, ValueError):
            area = None
    try:
        average_window = max(float(average_window), 0.0)
    except (TypeError, ValueError):
        average_window = defaults.average_window
    try:
        csv_interval = max(float(csv_interval), 0.0)
    except (TypeError, ValueError):
        csv_interval = defaults.csv_interval

    if csv_path is not None:
        csv_path = str(csv_path).strip()
        if not csv_path:
            csv_path = None

    return HeatFluxConfig(
        axis=axis,
        position=position,
        area=area,
        area_provided=area_provided,
        average_window=average_window,
        csv_path=csv_path,
        csv_interval=csv_interval,
    )


heat_flux_cfg = _load_heat_flux_config()

################################################################################
# Simulation class
################################################################################

class Simulation:
    """Evolve a gas of particles on a 2D torus with periodic boundaries.

    The simulation uses a simple elastic collision model between
    particles.  Particle positions and velocities are stored in
    continuous arrays for efficiency.  At each time step particles may
    collide with one another, and when they move beyond one edge of the
    unit square they re-enter from the opposite side.  No external
    potentials or springs are present in this variant.
    """

    def __init__(
        self,
        gamma: float,
        k: float,
        l_0: float,
        R: float,
        particles_cnt: int,
        T: float,
        m: ndarray,
        particle_types: Optional[list[ParticleType]] = None,
        magnetic_field: Optional[dict] = None,
        solar_wind: Optional[dict] = None,
    ):
        """Create a simulation with the given parameters.

        Parameters
        ----------
        gamma, k, l_0: float
            Legacy parameters from the original model (spring potential),
            unused in this version but accepted for API compatibility.
        R: float
            Radius of each particle in box units.  The simulation assumes
            the box extends from 0 to 1 in both x and y directions, so
            ``R`` must satisfy ``0 < R < 0.5``.
        particles_cnt: int
            Number of gas particles to simulate.
        T: float
            Legacy parameter (ignored in the cosmic-ray model).
        m: ndarray
            Masses of the particles (length ``particles_cnt``).  If a
            scalar is provided, it will be broadcast to the required
            shape.  Masses should be provided in SI units consistent
            with ``kB``.
        particle_types: list[ParticleType], optional
            Species definitions; if omitted, defaults are loaded from
            ``particle_types.json``.
        magnetic_field: dict, optional
            Configuration for the background field (keys: ``base``,
            ``gradient``).
        solar_wind: dict, optional
            Parameters of the solar wind (keys: ``speed``, ``direction_deg``).
        """
        # Store constants and parameters
        self._k_boltz: float = kB
        self._gamma: float = gamma
        self._k: float = k
        self._l_0: float = l_0
        self._R: float = R
        self._box_width: float = 1.0  # horizontal extent of the domain in box units
        self._lambda_param: float = 0.0

        # Number of gas particles
        self._n_particles: int = int(particles_cnt)
        self._n_spring: int = 0  # no spring particles

        # Ensure masses have correct shape
        masses = np.asarray(m, dtype=float)
        if masses.ndim == 0:
            masses = np.full((self._n_particles,), float(masses))
        elif masses.ndim == 1 and masses.shape[0] != self._n_particles:
            raise ValueError("Length of m must equal particles_cnt")
        self._m = masses

        # Load particle type definitions
        self._type_definitions: list[ParticleType] = particle_types or _load_particle_types_defaults()
        if not self._type_definitions:
            # Fallback single neutral species
            self._type_definitions = [
                ParticleType(
                    name='neutral',
                    charge_sign=0.0,
                    diffusion_x=0.0,
                    diffusion_y=0.0,
                    drift_coeff=0.0,
                    lifetime=None,
                    color=(200, 200, 200),
                    mass=None,
                    weight=1.0,
                    source="",
                )
            ]
        type_count = len(self._type_definitions)
        self._type_weights = np.array(
            [float(t.weight) if isinstance(t.weight, (int, float)) else 1.0 for t in self._type_definitions],
            dtype=float,
        )
        # Assign species using provided weights (uniform fallback)
        assignments = self._choose_types_by_weight(self._n_particles, self._type_weights)
        self._particle_type_indices = assignments
        # Per-particle parameters derived from types
        self._type_charge = np.array([t.charge_sign for t in self._type_definitions], dtype=float)
        self._type_diffusion_x = np.array([max(float(t.diffusion_x), 0.0) for t in self._type_definitions], dtype=float)
        self._type_diffusion_y = np.array([max(float(t.diffusion_y), 0.0) for t in self._type_definitions], dtype=float)
        self._type_drift_coeff = np.array([float(t.drift_coeff) for t in self._type_definitions], dtype=float)
        self._type_lifetimes = np.array(
            [float(t.lifetime) if isinstance(t.lifetime, (int, float)) and t.lifetime > 0 else np.inf for t in self._type_definitions],
            dtype=float,
        )
        self._type_colors = np.array([t.color for t in self._type_definitions], dtype=float).T  # shape (3, type_count)
        # Override masses per type when provided
        for idx, t in enumerate(self._type_definitions):
            if t.mass is not None:
                self._m[self._particle_type_indices == idx] = float(t.mass)

        # Initialize positions uniformly in the full periodic domain
        x_positions = np.random.uniform(low=0.0, high=self._box_width, size=self._n_particles)
        y_positions = np.random.uniform(low=0.0, high=1.0, size=self._n_particles)
        self._r = np.vstack((x_positions, y_positions))

        # Velocities start at zero; motion is driven by diffusion and fields.
        self._v = np.zeros((2, self._n_particles), dtype=float)
        # Add a small supplemental electron cloud to the initial gas.
        self._add_initial_electrons(INITIAL_EXTRA_ELECTRONS)

        # Save initial target temperature and energy (legacy bookkeeping)
        self._potential_energy = []
        self._kinetic_energy = []
        self._E_full: float = self.calc_full_energy()
        self._T_tar: float = max(float(T), 1.0)

        # Thermal wall parameters (unused with periodic boundaries)
        self.T_left: float = thermal_cfg.T_left
        self.T_right: float = thermal_cfg.T_right

        # Magnetic field configuration
        field_cfg = magnetic_field if isinstance(magnetic_field, dict) else _load_magnetic_field_config()
        try:
            self._B0 = float(field_cfg.get('base', 0.0))
        except Exception:
            self._B0 = 0.0
        # Keep the background field uniform and horizontal (left to right).
        self._B_background: float = self._B0
        self._grad_B: float = 0.0
        self._B_angle: float = 0.0
        # Tunable strength multipliers to balance wind vs. magnetic effects.
        try:
            self._field_push_scale = float(field_cfg.get('push_scale', 8.0))
        except Exception:
            self._field_push_scale = 8.0
        try:
            self._drift_scale = float(field_cfg.get('drift_scale', 10.0))
        except Exception:
            self._drift_scale = 10.0
        try:
            self._dipole_moment = float(field_cfg.get('dipole_moment', 0.35))
        except Exception:
            self._dipole_moment = 0.35
        try:
            self._dipole_scale = float(field_cfg.get('dipole_scale', 1.0))
        except Exception:
            self._dipole_scale = 1.0
        try:
            self._electric_scale = float(field_cfg.get('electric_scale', 0.6))
        except Exception:
            self._electric_scale = 0.6

        wind_cfg = solar_wind if isinstance(solar_wind, dict) else _load_solar_wind_config()
        try:
            self._wind_speed = float(wind_cfg.get('speed', 0.0))
        except Exception:
            self._wind_speed = 0.0
        try:
            self._wind_angle = math.radians(float(wind_cfg.get('direction_deg', 0.0)))
        except Exception:
            self._wind_angle = 0.0

        # Lifetime timers per particle for decay-enabled species (muons)
        self._time_to_live = self._sample_time_to_live(self._particle_type_indices)
        # Store decay events for visual effects
        self._decay_events: list[tuple[np.ndarray, int]] = []
        # Absorption zones (set by UI)
        self._absorber_centers: np.ndarray = np.zeros((2, 0), dtype=float)
        self._absorber_radius: float = 0.0
        # Footpoint field parameters (centres set via UI)
        self._foot_centers_field: np.ndarray = np.zeros((2, 0), dtype=float)
        self._foot_radius_field: float = 0.0
        self._foot_strengths: np.ndarray = np.zeros((0,), dtype=float)

        # Prepare collision pairs for particle collisions
        self._init_ids_pairs()

        # Frame counter for energy fixes (unused, kept for API compatibility)
        self._frame_no: int = 1
        # Base integration time step and current scaling factor (slow-mo support)
        self._base_dt: float = TIME_STEP
        self._time_scale: float = 1.0
        self._dt: float = self._base_dt * self._time_scale

        # Track particle indices that touched each wall during the last step
        self._last_wall_hits: Dict[str, np.ndarray] = {
            'left': np.empty(0, dtype=int),
            'right': np.empty(0, dtype=int),
            'top': np.empty(0, dtype=int),
            'bottom': np.empty(0, dtype=int),
        }
        self._midplane_axis: str = heat_flux_cfg.axis
        self._midplane_axis_index: int = 0 if self._midplane_axis == 'x' else 1
        axis_range_max = self._box_width if self._midplane_axis == 'x' else 1.0
        self._midplane_position: float = float(np.clip(heat_flux_cfg.position, 0.0 + R, axis_range_max - R))
        default_area = self._box_width if self._midplane_axis == 'y' else 1.0
        area_value = heat_flux_cfg.area if heat_flux_cfg.area_provided and heat_flux_cfg.area is not None else default_area
        self._midplane_area: float = float(max(area_value, 1e-12))
        self._midplane_area_locked: bool = heat_flux_cfg.area_provided
        self._flux_average_window: float = float(max(heat_flux_cfg.average_window, 0.0))
        self._last_midplane_flux_raw: float = 0.0
        self._last_midplane_flux: float = 0.0
        self._last_midplane_heat_transfer: float = 0.0
        self._last_midplane_crossings: Dict[str, np.ndarray] = {
            'positive': np.empty(0, dtype=int),
            'negative': np.empty(0, dtype=int),
        }
        self._last_midplane_counts: Dict[str, int] = {
            'positive': 0,
            'negative': 0,
        }
        self._last_midplane_energy: Dict[str, float] = {
            'positive': 0.0,
            'negative': 0.0,
        }
        self._flux_history: list[Tuple[float, float, float]] = []
        self._max_flux_history: int = 5000
        self._flux_window: Deque[Tuple[float, float, float]] = deque()
        self._flux_window_energy: float = 0.0
        self._flux_window_duration: float = 0.0
        self._heat_cumulative_window: Deque[Tuple[float, float]] = deque()
        self._heat_cumulative_span: float = float(max(HEAT_CUMULATIVE_WINDOW, 0.0))
        self._cumulative_heat: float = 0.0
        self._flux_csv_path: Optional[str] = heat_flux_cfg.csv_path
        self._flux_csv_interval: float = float(max(heat_flux_cfg.csv_interval, 0.0))
        self._last_flux_csv_time: float = -math.inf
        if self._flux_csv_path:
            csv_file = Path(self._flux_csv_path).expanduser()
            try:
                self._flux_csv_header_written = csv_file.exists() and csv_file.stat().st_size > 0
            except OSError:
                self._flux_csv_header_written = False
        else:
            self._flux_csv_header_written = False
        self._elapsed_time: float = 0.0

    # -------------------------------------------------------------------------
    # Properties to expose slices of the state
    @property
    def r(self) -> ndarray:
        """Return positions of gas particles as a 2×N array."""
        return self._r

    @property
    def r_spring(self) -> ndarray:
        """Return positions of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def v(self) -> ndarray:
        """Return velocities of gas particles as a 2×N array."""
        return self._v

    @property
    def v_spring(self) -> ndarray:
        """Return velocities of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def m(self) -> ndarray:
        """Return masses of gas particles."""
        return self._m

    @property
    def m_spring(self) -> ndarray:
        """Return masses of spring particles (empty array)."""
        return np.zeros((0,), dtype=float)

    @property
    def R(self) -> float:
        """Radius of gas particles."""
        return self._R

    @property
    def R_spring(self) -> float:
        """Radius of spring particles (zero since none exist)."""
        return 0.0

    # -------------------------------------------------------------------------
    # Time scaling helpers ----------------------------------------------------
    def set_time_scale(self, scale: float) -> None:
        """Adjust integration step multiplier used for slow-motion or speed-up."""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not math.isfinite(scale):
            scale = 1.0
        # Clamp to a sensible range to keep the integrator stable
        scale = float(np.clip(scale, 0.05, 5.0))
        self._time_scale = scale
        self._dt = self._base_dt * self._time_scale

    def get_time_scale(self) -> float:
        """Return the current integration step multiplier."""
        return self._time_scale

    def get_last_wall_hits(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that touched each wall during the last step."""
        return {side: hits.copy() for side, hits in self._last_wall_hits.items()}

    def get_last_midplane_flux(self) -> float:
        """Return the most recent (optionally averaged) heat flux through the midplane."""
        return self._last_midplane_flux

    def get_last_midplane_flux_raw(self) -> float:
        """Return the most recent instantaneous heat flux sample."""
        return self._last_midplane_flux_raw

    def get_last_midplane_counts(self) -> Dict[str, int]:
        """Return counts of crossings through the midplane for the last step (positive/negative)."""
        return self._last_midplane_counts.copy()

    def get_last_midplane_energy(self) -> Dict[str, float]:
        """Return signed heat transfer contributions (per unit area) for the last step."""
        return self._last_midplane_energy.copy()

    def get_midplane_crossings(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that crossed the midplane in the last step."""
        return {side: indices.copy() for side, indices in self._last_midplane_crossings.items()}

    def get_midplane_flux_history(self, raw: bool = False) -> list[Tuple[float, float]]:
        """Return recorded heat flux history as (time, flux) pairs."""
        if raw:
            return [(t, raw_flux) for t, raw_flux, _ in self._flux_history]
        return [(t, avg_flux) for t, _, avg_flux in self._flux_history]

    def get_cumulative_midplane_heat(self) -> float:
        """Return heat transferred during roughly the last five seconds (per unit area)."""
        return self._cumulative_heat

    def get_heat_cumulative_span(self) -> float:
        """Return the configured duration (seconds) used for the cumulative heat window."""
        return self._heat_cumulative_span

    def get_last_midplane_heat_transfer(self) -> float:
        """Return heat transferred during the most recent step (per unit area)."""
        return self._last_midplane_heat_transfer

    def get_midplane_axis(self) -> str:
        """Return the axis normal to the tracked midplane ('x' or 'y')."""
        return self._midplane_axis

    def get_midplane_position(self) -> float:
        """Return the midplane coordinate along its axis."""
        return self._midplane_position

    def get_midplane_area(self) -> float:
        """Return the effective cross-sectional area used for flux calculations."""
        return self._midplane_area

    def get_elapsed_time(self) -> float:
        """Return the total elapsed simulation time."""
        return self._elapsed_time

    def get_field_direction(self) -> float:
        """Return background magnetic field direction in radians."""
        return float(self._B_angle)

    def get_wind_direction(self) -> float:
        """Return solar wind direction in radians."""
        return float(self._wind_angle)

    def get_wind_speed(self) -> float:
        """Return solar wind speed (simulation units)."""
        return float(self._wind_speed)

    def get_field_min_max(self) -> tuple[float, float]:
        """Return current minimum and maximum B across the domain (background + peaks)."""
        base = float(self._B_background)
        if self._foot_strengths.size:
            max_extra = float(np.max(self._foot_strengths))
            min_extra = float(np.min(self._foot_strengths))
        else:
            max_extra = 0.0
            min_extra = 0.0
        return base + min(0.0, min_extra), base + max(0.0, max_extra)

    def get_local_field_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-particle field magnitude along with the in-plane gradients."""
        return self._local_field_profile()

    def get_footpoint_field_radius(self) -> float:
        """Return the radius used for the magnetic footpoint profile."""
        return float(self._foot_radius_field)

    def _local_field_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local field magnitude and its 2D gradient at particle positions."""
        width = max(self._box_width, 1e-9)
        x = self._r[0]
        y = self._r[1]
        b_local = np.full_like(x, self._B_background)
        grad_x = np.zeros_like(b_local)
        grad_y = np.zeros_like(b_local)

        if self._grad_B != 0.0:
            b_local += self._grad_B * (x - 0.5 * width)
            grad_x += self._grad_B

        if self._foot_centers_field.size:
            radius = max(self._foot_radius_field, 1e-6)
            centers = self._foot_centers_field.T
            pos = np.stack((x, y), axis=1)  # (N, 2)
            for idx, center in enumerate(centers):
                amp = float(self._foot_strengths[idx]) if idx < len(self._foot_strengths) else 0.0
                dx = pos[:, 0] - center[0]
                dy = pos[:, 1] - center[1]
                # Account for periodic domain: wrap to nearest image
                dx -= np.round(dx / width) * width
                dy -= np.round(dy / 1.0) * 1.0
                r2 = dx * dx + dy * dy
                safe_r = np.sqrt(np.maximum(r2, 1e-12))
                profile = np.exp(-((safe_r / radius) ** 4))
                b_local += amp * profile
                prefac = (-4.0 * (safe_r ** 3) / (radius ** 4)) * amp * profile
                inv_r = 1.0 / (safe_r + 1e-12)
                grad_x += prefac * dx * inv_r
                grad_y += prefac * dy * inv_r
        return b_local, grad_x, grad_y

    def set_field_direction(self, angle_rad: float) -> None:
        """Update displayed magnetic field direction (for UI)."""
        try:
            self._B_angle = float(angle_rad)
        except (TypeError, ValueError):
            return

    def set_field_range(self, b_min: float, b_max: float) -> None:
        """Set a uniform background field magnitude using provided range."""
        b_min = float(b_min)
        b_max = float(b_max)
        if b_max < b_min:
            b_max = b_min
        self._B_background = 0.5 * (b_min + b_max)
        self._B0 = self._B_background
        # Keep the background uniform; gradient is intentionally zeroed.
        self._grad_B = 0.0

    def set_wind_direction(self, angle_rad: float) -> None:
        """Update solar wind direction (used for acceleration)."""
        try:
            self._wind_angle = float(angle_rad)
        except (TypeError, ValueError):
            return

    def set_wind_speed(self, speed: float) -> None:
        """Update solar wind speed magnitude."""
        try:
            self._wind_speed = float(speed)
        except (TypeError, ValueError):
            return

    def get_particle_types(self) -> np.ndarray:
        """Return integer type indices for each particle."""
        return self._particle_type_indices.copy()

    def get_particle_colors(self) -> np.ndarray:
        """Return RGB colors for each particle (shape 3×N)."""
        colors = self._type_colors[:, self._particle_type_indices]
        return colors.copy()

    def get_lambda_param(self) -> float:
        """Return the current λ parameter for SDE interpretation."""
        return self._lambda_param

    def set_lambda_param(self, value: float) -> None:
        """Set λ parameter (clamped to [0, 1])."""
        try:
            self._lambda_param = float(np.clip(value, 0.0, 1.0))
        except Exception:
            self._lambda_param = 0.0

    def get_type_definitions(self) -> list[ParticleType]:
        """Return copies of the particle type definitions."""
        return [ParticleType(**dataclasses.asdict(t)) for t in self._type_definitions]

    def get_box_width(self) -> float:
        """Return the horizontal extent of the simulation domain."""
        return self._box_width

    def get_particle_count(self) -> int:
        """Return the number of gas particles being simulated."""
        return int(self._n_particles)

    def get_field_strength(self) -> float:
        """Return the magnitude of the background magnetic field."""
        base = float(getattr(self, "_B_background", getattr(self, "_B0", 0.0)))
        if getattr(self, "_foot_strengths", None) is not None and self._foot_strengths.size:
            return base + float(np.max(np.abs(self._foot_strengths)))
        return base

    # -------------------------------------------------------------------------
    # Thermodynamic properties
    @property
    def T(self) -> float:
        """Compute instantaneous temperature from kinetic energies (K)."""
        # The factor 2 accounts for 2 degrees of freedom per particle
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / (2 * self._k_boltz)

    @T.setter
    def T(self, val: float) -> None:
        if val <= 0:
            raise ValueError("Temperature must be positive")
        delta = val / self._T_tar
        # Scale velocities to achieve new temperature
        self._v *= np.sqrt(delta)
        self._E_full = self.calc_full_energy()
        self._T_tar = val

    # -------------------------------------------------------------------------
    def _init_ids_pairs(self) -> None:
        """Compute index pairs for potential collisions between gas particles."""
        particles_ids = np.arange(self._n_particles)
        self._particles_ids_pairs = np.asarray(list(itertools.combinations(particles_ids, 2)), dtype=int)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_deltad2_pairs(r: np.ndarray, ids_pairs: np.ndarray) -> np.ndarray:
        """Compute squared distances between all pairs of points given by indices."""
        dx = np.diff(np.stack([r[0][ids_pairs[:, 0]], r[0][ids_pairs[:, 1]]]).T).squeeze()
        dy = np.diff(np.stack([r[1][ids_pairs[:, 0]], r[1][ids_pairs[:, 1]]]).T).squeeze()
        return dx ** 2 + dy ** 2

    @staticmethod
    def compute_new_v(
        v1: np.ndarray, v2: np.ndarray, dr: np.ndarray, m1: np.ndarray, m2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute post‑collision velocities for an elastic collision of two particles."""
        m_s = m1 + m2
        dr_norm_sq = np.linalg.norm(dr, axis=0) ** 2
        dot1 = np.sum((2 * m2 / m_s) * (v1 - v2) * dr, axis=0)
        dot2 = np.sum((2 * m1 / m_s) * (v2 - v1) * dr, axis=0)
        v1new = v1 - (dot1 * dr) / dr_norm_sq
        v2new = v2 - (dot2 * dr) / dr_norm_sq
        return v1new, v2new

    # -------------------------------------------------------------------------
    def _apply_decay(self, dt: float) -> None:
        """Decay is disabled; particles no longer vanish automatically."""
        return

    def _sample_time_to_live(self, type_indices: np.ndarray) -> np.ndarray:
        """Sample decay timers for the given particle type indices."""
        if type_indices.size == 0:
            return np.empty((0,), dtype=float)
        lifetimes = self._type_lifetimes[type_indices]
        timers = np.full_like(lifetimes, np.inf, dtype=float)
        finite_mask = np.isfinite(lifetimes)
        if np.any(finite_mask):
            scales = lifetimes[finite_mask]
            if rng is not None:
                timers[finite_mask] = rng.exponential(scales)
            else:
                timers[finite_mask] = np.random.exponential(scales)
        return timers

    def _find_electron_type_index(self) -> int:
        """Return the type index that appears to represent electrons."""
        electron_aliases = ('electron', 'электро', 'e-')
        for idx, t in enumerate(self._type_definitions):
            name = str(t.name or '').lower()
            if any(alias in name for alias in electron_aliases):
                return idx
        return -1

    def _add_initial_electrons(self, count: int) -> None:
        """Inject a small, fixed number of electrons near the initial particles."""
        electron_idx = self._find_electron_type_index()
        if electron_idx < 0:
            return
        total_count = int(max(0, count))
        if total_count <= 0:
            return
        base_count = self._n_particles
        if base_count > 0:
            if rng is not None:
                sample_indices = rng.integers(0, base_count, size=total_count)
            else:
                sample_indices = np.random.randint(0, base_count, size=total_count)
            electron_positions = self._r[:, sample_indices].copy()
        else:
            electron_positions = np.vstack(
                (
                    np.random.uniform(0.0, self._box_width, size=total_count),
                    np.random.uniform(0.0, 1.0, size=total_count),
                )
            )
        jitter_scale = max(min(self._R * 0.25, 0.15), 0.01)
        if rng is not None:
            offsets = rng.normal(loc=0.0, scale=jitter_scale, size=electron_positions.shape)
        else:
            offsets = np.random.normal(loc=0.0, scale=jitter_scale, size=electron_positions.shape)
        electron_positions += offsets
        electron_positions[0] %= self._box_width
        electron_positions[1] %= 1.0
        electron_velocities = np.zeros((2, total_count), dtype=float)
        electron_mass_val = self._type_definitions[electron_idx].mass
        electron_mass = float(electron_mass_val) if isinstance(electron_mass_val, (int, float)) else 1.0
        additional_masses = np.full((total_count,), electron_mass, dtype=float)
        additional_indices = np.full((total_count,), electron_idx, dtype=int)
        self._r = np.hstack((self._r, electron_positions))
        self._v = np.hstack((self._v, electron_velocities))
        self._m = np.concatenate((self._m, additional_masses))
        self._particle_type_indices = np.concatenate((self._particle_type_indices, additional_indices))
        self._n_particles = base_count + total_count

    def _refresh_type_arrays(self) -> None:
        """Recompute per-type and per-particle arrays after edits."""
        self._type_charge = np.array([t.charge_sign for t in self._type_definitions], dtype=float)
        self._type_diffusion_x = np.array([max(float(t.diffusion_x), 0.0) for t in self._type_definitions], dtype=float)
        self._type_diffusion_y = np.array([max(float(t.diffusion_y), 0.0) for t in self._type_definitions], dtype=float)
        self._type_drift_coeff = np.array([float(t.drift_coeff) for t in self._type_definitions], dtype=float)
        self._type_lifetimes = np.array(
            [float(t.lifetime) if isinstance(t.lifetime, (int, float)) and t.lifetime > 0 else np.inf for t in self._type_definitions],
            dtype=float,
        )
        self._type_colors = np.array([t.color for t in self._type_definitions], dtype=float).T
        self._type_weights = np.array(
            [float(t.weight) if isinstance(t.weight, (int, float)) else 1.0 for t in self._type_definitions],
            dtype=float,
        )
        for idx, t in enumerate(self._type_definitions):
            if t.mass is not None:
                self._m[self._particle_type_indices == idx] = float(t.mass)
        self._time_to_live = self._sample_time_to_live(self._particle_type_indices)
        self._reassign_types_by_weight()

    def update_particle_type_value(self, type_index: int, field: str, value: float) -> None:
        """Update a single field of a particle type and refresh derived arrays."""
        if type_index < 0 or type_index >= len(self._type_definitions):
            return
        t = self._type_definitions[type_index]
        if not hasattr(t, field):
            return
        updated = dataclasses.replace(t, **{field: value})
        self._type_definitions[type_index] = updated
        self._refresh_type_arrays()

    def _choose_types_by_weight(self, count: int, weights: np.ndarray) -> np.ndarray:
        """Sample particle types respecting provided weights."""
        type_count = len(weights)
        if count <= 0 or type_count == 0:
            return np.zeros((max(count, 0),), dtype=int)
        safe_weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if not np.any(safe_weights):
            safe_weights = np.ones_like(safe_weights)
        probs = safe_weights / np.sum(safe_weights)
        if rng is not None:
            return rng.choice(type_count, size=count, p=probs)
        return np.random.choice(type_count, size=count, p=probs)

    def _reassign_types_by_weight(self) -> None:
        """Re-sample particle species when the weight table changes."""
        if getattr(self, "_type_weights", None) is None or self._n_particles <= 0:
            return
        new_indices = self._choose_types_by_weight(self._n_particles, self._type_weights)
        self._particle_type_indices = new_indices
        for idx, t in enumerate(self._type_definitions):
            if t.mass is not None:
                self._m[self._particle_type_indices == idx] = float(t.mass)
        self._time_to_live = self._sample_time_to_live(self._particle_type_indices)

    def _apply_absorption(self) -> None:
        """Remove particles whose centers enter the absorber disks."""
        centers = getattr(self, "_absorber_centers", np.zeros((2, 0), dtype=float))
        radius = getattr(self, "_absorber_radius", 0.0)
        if centers.size == 0 or radius <= 0.0 or self._r.shape[1] == 0:
            return
        pos = self._r.T  # shape (N, 2)
        width = max(self._box_width, 1e-6)
        dx = pos[:, None, 0] - centers[0][None, :]
        dy = pos[:, None, 1] - centers[1][None, :]
        dx -= np.round(dx / width) * width
        dy -= np.round(dy / 1.0) * 1.0
        dist2 = dx ** 2 + dy ** 2
        rad2 = float(radius) ** 2
        mask = np.any(dist2 <= rad2, axis=1)
        if not np.any(mask):
            return
        keep = ~mask
        self._r = self._r[:, keep]
        self._v = self._v[:, keep]
        self._m = self._m[keep]
        self._particle_type_indices = self._particle_type_indices[keep]
        self._time_to_live = self._time_to_live[keep] if hasattr(self, "_time_to_live") else np.empty((0,), dtype=float)
        self._n_particles = self._r.shape[1]
        self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def consume_decay_events(self) -> list[tuple[np.ndarray, int]]:
        """Return and clear recorded decay events."""
        events = getattr(self, "_decay_events", [])
        self._decay_events = []
        return events

    def zero_velocities(self) -> None:
        """Reset all particle velocities to zero."""
        self._v[:] = 0.0

    def _apply_magnetic_push(self, dt: float) -> None:
        """Accelerate charged particles along the in-plane field lines."""
        if dt <= 0.0:
            return
        charges = self._type_charge[self._particle_type_indices]
        if not np.any(charges):
            return
        inv_mass = 1.0 / np.maximum(self._m, 1e-12)
        cos_dir = math.cos(self._B_angle)
        sin_dir = math.sin(self._B_angle)
        b_local, _, _ = self._local_field_profile()

        # Field-aligned push keeps motion visible even from rest; no trajectory twisting.
        accel_mag = charges * b_local * self._field_push_scale * inv_mass
        self._v[0] += accel_mag * cos_dir * dt
        self._v[1] += accel_mag * sin_dir * dt

    def _apply_solar_wind(self, dt: float) -> None:
        """Accelerate particles in the direction of the solar wind."""
        if dt <= 0.0 or self._wind_speed == 0.0:
            return
        accel_mag = self._wind_speed * 5.0  # boost for visible effect
        ax = accel_mag * math.cos(self._wind_angle)
        ay = accel_mag * math.sin(self._wind_angle)
        inv_mass = 1.0 / np.maximum(self._m, 1e-12)
        self._v[0] += ax * inv_mass * dt
        self._v[1] += ay * inv_mass * dt

    def _apply_footpoint_magnetic_field(self, dt: float) -> None:
        """Apply the radial magnetic-gradient force produced by each footpoint."""
        if dt <= 0.0:
            return
        count = self._foot_centers_field.shape[1]
        if count == 0:
            return
        charges = self._type_charge[self._particle_type_indices]
        if not np.any(charges):
            return
        centers = self._foot_centers_field
        strengths = self._foot_strengths
        radius = max(self._foot_radius_field, 1e-6)
        inv_mass = 1.0 / np.maximum(self._m, 1e-12)
        safe_radius4 = max(radius ** 4, 1e-18)

        grad_x = np.zeros_like(self._r[0])
        grad_y = np.zeros_like(self._r[1])
        for idx in range(count):
            strength = float(strengths[idx]) if idx < strengths.shape[0] else 0.0
            if abs(strength) < 1e-12:
                continue
            center_x = centers[0, idx]
            center_y = centers[1, idx]
            dx = self._r[0] - center_x
            dy = self._r[1] - center_y
            dx -= np.round(dx / self._box_width) * self._box_width
            dy -= np.round(dy / 1.0) * 1.0
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2)
            safe_r = np.maximum(r, 1e-9)
            exp_term = np.exp(-((r / radius) ** 4))
            prefactor = -4.0 * strength * exp_term * (safe_r ** 3) / safe_radius4
            grad_x += prefactor * dx / safe_r
            grad_y += prefactor * dy / safe_r

        if not np.any(grad_x) and not np.any(grad_y):
            return
        accel_x = -charges * grad_x * self._dipole_scale * inv_mass
        accel_y = -charges * grad_y * self._dipole_scale * inv_mass
        self._v[0] += accel_x * dt
        self._v[1] += accel_y * dt

    def _apply_footpoint_electric_field(self, dt: float) -> None:
        """Apply a dipole-like electric field from each footpoint."""
        if dt <= 0.0:
            return
        count = self._foot_centers_field.shape[1]
        if count == 0:
            return
        charges = self._type_charge[self._particle_type_indices]
        if not np.any(charges):
            return
        centers = self._foot_centers_field
        strengths = self._foot_strengths
        box_width = max(self._box_width, 1e-6)
        field_x = np.zeros_like(self._r[0])
        field_y = np.zeros_like(self._r[1])
        for idx in range(count):
            strength = float(strengths[idx]) if idx < strengths.shape[0] else 0.0
            if abs(strength) < 1e-12:
                continue
            dx = self._r[0] - centers[0, idx]
            dy = self._r[1] - centers[1, idx]
            dx -= np.round(dx / box_width) * box_width
            dy -= np.round(dy / 1.0) * 1.0
            r2 = dx * dx + dy * dy
            safe_r2 = np.maximum(r2, 1e-6)
            inv_r = 1.0 / np.sqrt(safe_r2)
            contribution = strength / safe_r2
            field_x += contribution * dx * inv_r
            field_y += contribution * dy * inv_r
        if not np.any(field_x) and not np.any(field_y):
            return
        inv_mass = 1.0 / np.maximum(self._m, 1e-12)
        accel_x = charges * field_x * self._electric_scale * inv_mass
        accel_y = charges * field_y * self._electric_scale * inv_mass
        self._v[0] += accel_x * dt
        self._v[1] += accel_y * dt

    def _apply_diffusion_and_drift(self, dt: float) -> None:
        """Add stochastic diffusion and deterministic drift from grad|B|."""
        if dt <= 0.0:
            return
        types = self._particle_type_indices
        diff_x = self._type_diffusion_x[types]
        diff_y = self._type_diffusion_y[types]
        drift_coeff = self._type_drift_coeff[types]
        charges = self._type_charge[types]
        _, grad_x, grad_y = self._local_field_profile()
        drift_x = charges * drift_coeff * grad_x * self._drift_scale
        drift_y = charges * drift_coeff * grad_y * self._drift_scale
        if rng is not None:
            noise_x = rng.normal(loc=0.0, scale=np.sqrt(2.0 * diff_x * dt))
            noise_y = rng.normal(loc=0.0, scale=np.sqrt(2.0 * diff_y * dt))
        else:
            noise_x = np.random.normal(loc=0.0, scale=np.sqrt(2.0 * diff_x * dt))
            noise_y = np.random.normal(loc=0.0, scale=np.sqrt(2.0 * diff_y * dt))

        # Update velocities to reflect drift for this step
        self._v[0] += drift_x * dt
        self._v[1] += drift_y * dt
        # Incorporate stochastic velocity component for diagnostics/flux
        inv_dt = 1.0 / dt if dt > 0.0 else 0.0
        self._v[0] += noise_x * inv_dt
        self._v[1] += noise_y * inv_dt
        # Displacement from stochastic increments; add to positions directly
        self._r[0] += noise_x
        self._r[1] += noise_y

    # -------------------------------------------------------------------------
    def motion(self, dt: float) -> float:
        """Advance the system by one time step of length ``dt``.

        Returns
        -------
        float
            Always returns ``0.0`` in this version.  In the original code
            this value was the work done by the spring force.
        """
        # Apply drift and stochastic kicks before collision handling so new
        # displacements participate in collision checks.
        self._apply_diffusion_and_drift(dt)
        # Rotate velocities due to the background magnetic field
        self._apply_magnetic_push(dt)
        # Accelerate due to solar wind
        self._apply_solar_wind(dt)
        # Magnetic gradient from the footpoint poles
        self._apply_footpoint_magnetic_field(dt)
        self._apply_footpoint_electric_field(dt)
        # ------------------------------------------------------------------
        # Handle particle–particle collisions
        box_w = self._box_width
        box_h = 1.0
        if self._particles_ids_pairs.size:
            idx_i = self._particles_ids_pairs[:, 0]
            idx_j = self._particles_ids_pairs[:, 1]

            dx = self._r[0, idx_i] - self._r[0, idx_j]
            dy = self._r[1, idx_i] - self._r[1, idx_j]
            # Apply minimal image for periodic boundaries
            dx -= np.round(dx / box_w) * box_w
            dy -= np.round(dy / box_h) * box_h

            d2 = dx ** 2 + dy ** 2
            colliding_mask = d2 < (2 * self._R) ** 2
            ic_particles = self._particles_ids_pairs[colliding_mask]
            dx_collide = dx[colliding_mask]
            dy_collide = dy[colliding_mask]
        else:
            ic_particles = np.zeros((0, 2), dtype=int)
            dx_collide = np.zeros((0,), dtype=float)
            dy_collide = np.zeros((0,), dtype=float)

        if ic_particles.size:
            # Resolve collisions by updating velocities
            v1 = self._v[:, ic_particles[:, 0]]
            v2 = self._v[:, ic_particles[:, 1]]
            dr = np.vstack((dx_collide, dy_collide))
            m1 = self._m[ic_particles[:, 0]]
            m2 = self._m[ic_particles[:, 1]]
            v1new, v2new = self.compute_new_v(v1, v2, dr, m1, m2)
            self._v[:, ic_particles[:, 0]] = v1new
            self._v[:, ic_particles[:, 1]] = v2new

            # ------------------------------------------------------------------
            # Positional correction: separate overlapping particles
            # After updating velocities, particles may still overlap because they
            # penetrated each other within one time step.  To prevent "sticking"
            # and repeated collisions, move them apart along the line of centers.
            idx_i = ic_particles[:, 0]
            idx_j = ic_particles[:, 1]
            # Vector from j to i for each colliding pair
            dr = np.vstack((dx_collide, dy_collide))  # shape (2, K)
            # Euclidean distance between centers
            dist = np.linalg.norm(dr, axis=0)
            # Compute how much they overlap: (2R - dist).  Negative values mean no overlap
            overlap = (2.0 * self._R) - dist
            # Mask of truly overlapping pairs (distance < 2R)
            overlap_mask = overlap > 0.0
            if np.any(overlap_mask):
                # Normalize the direction vector for overlapping pairs
                # To avoid division by zero, clip very small distances
                safe_dist = np.copy(dist[overlap_mask])
                safe_dist[safe_dist < 1e-12] = 1e-12
                n = dr[:, overlap_mask] / safe_dist  # shape (2, M)
                # Each particle moves half the overlap distance in opposite directions
                shift = 0.5 * overlap[overlap_mask]
                # Broadcast shift to both components
                self._r[:, idx_i[overlap_mask]] += n * shift
                self._r[:, idx_j[overlap_mask]] -= n * shift

        # ------------------------------------------------------------------
        # Periodic boundaries: clear wall hit trackers
        self._last_wall_hits['left'] = np.empty(0, dtype=int)
        self._last_wall_hits['right'] = np.empty(0, dtype=int)
        self._last_wall_hits['bottom'] = np.empty(0, dtype=int)
        self._last_wall_hits['top'] = np.empty(0, dtype=int)

        # ------------------------------------------------------------------
        # Integrate positions
        prev_axis_coord = self._r[self._midplane_axis_index].copy()
        self._r += self._v * dt
        # Wrap positions onto the periodic domain (torus)
        self._r[0] = np.mod(self._r[0], self._box_width)
        self._r[1] = np.mod(self._r[1], 1.0)
        self._update_midplane_flux(prev_axis_coord, dt)
        self._apply_absorption()

        # ------------------------------------------------------------------
        return 0.0

    # -------------------------------------------------------------------------
    def __iter__(self) -> 'Simulation':
        return self

    def __next__(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
        """Advance the simulation and return state arrays.

        The tuple contains positions and velocities of gas particles and
        (empty) spring particles, plus the value returned by ``motion``.
        """
        f = self.motion(dt=self._dt)
        self._frame_no = (self._frame_no + 1) % 5

        self._potential_energy.append(self.calc_potential_energy())
        self._kinetic_energy.append(self.calc_kinetic_energy())

        if self._frame_no == 0:
            self._fix_energy()

        return self.r, self.r_spring, self.v, self.v_spring, f

    # -------------------------------------------------------------------------
    def _update_midplane_flux(self, prev_axis: np.ndarray, dt: float) -> None:
        """Update heat flux across the midplane based on the last step."""
        plane = self._midplane_position if self._midplane_axis == 'y' else self._midplane_position
        if self._midplane_axis == 'x':
            plane = self._midplane_position
        axis_idx = self._midplane_axis_index
        new_axis = self._r[axis_idx]
        prev_axis_arr = np.asarray(prev_axis, dtype=float)
        if prev_axis_arr.shape != new_axis.shape:
            prev_axis_arr = np.reshape(prev_axis_arr, new_axis.shape)

        positive_mask = (prev_axis_arr < plane) & (new_axis >= plane)
        negative_mask = (prev_axis_arr > plane) & (new_axis <= plane)
        positive_idx = np.nonzero(positive_mask)[0]
        negative_idx = np.nonzero(negative_mask)[0]

        self._last_midplane_crossings['positive'] = positive_idx
        self._last_midplane_crossings['negative'] = negative_idx

        pos_count = int(positive_idx.size)
        neg_count = int(negative_idx.size)
        self._last_midplane_counts['positive'] = pos_count
        self._last_midplane_counts['negative'] = neg_count

        velocities_sq = np.sum(self._v * self._v, axis=0)
        energies = 0.5 * self._m * velocities_sq
        pos_energy = float(np.sum(energies[positive_idx])) if pos_count else 0.0
        neg_energy = float(np.sum(energies[negative_idx])) if neg_count else 0.0
        net_energy = pos_energy - neg_energy

        if dt > 0.0:
            self._elapsed_time += dt
        timestamp = self._elapsed_time

        heat_transfer = net_energy / self._midplane_area
        self._last_midplane_heat_transfer = heat_transfer
        self._update_cumulative_heat_window(timestamp, heat_transfer)
        self._last_midplane_energy['positive'] = pos_energy / self._midplane_area
        self._last_midplane_energy['negative'] = -neg_energy / self._midplane_area

        flux_raw = heat_transfer / dt if dt > 0.0 else 0.0
        self._last_midplane_flux_raw = flux_raw

        if self._flux_average_window > 0.0 and dt > 0.0:
            avg_flux = self._update_flux_window(net_energy, dt)
        else:
            avg_flux = flux_raw
            if self._flux_average_window <= 0.0:
                self._flux_window.clear()
                self._flux_window_energy = 0.0
                self._flux_window_duration = 0.0
        self._last_midplane_flux = avg_flux

        self._flux_history.append((timestamp, flux_raw, avg_flux))
        if len(self._flux_history) > self._max_flux_history:
            self._flux_history.pop(0)

        self._maybe_write_flux_csv(timestamp)

    def _update_flux_window(self, net_energy: float, dt: float) -> float:
        """Update the rolling window used to time-average the flux."""
        window = self._flux_average_window
        if window <= 0.0 or dt <= 0.0:
            return self._last_midplane_flux_raw

        timestamp = self._elapsed_time
        self._flux_window.append((timestamp, net_energy, dt))
        self._flux_window_energy += net_energy
        self._flux_window_duration += dt

        # Trim to the configured window length, allowing partial removal.
        while self._flux_window and self._flux_window_duration - self._flux_window[0][2] > window:
            _, energy_old, dt_old = self._flux_window.popleft()
            self._flux_window_energy -= energy_old
            self._flux_window_duration -= dt_old

        if self._flux_window and self._flux_window_duration > window:
            excess = self._flux_window_duration - window
            ts_old, energy_old, dt_old = self._flux_window[0]
            if dt_old > 0.0:
                fraction = min(1.0, excess / dt_old)
                energy_remove = energy_old * fraction
                dt_remove = dt_old * fraction
                remaining_energy = max(0.0, energy_old - energy_remove)
                remaining_dt = max(0.0, dt_old - dt_remove)
                self._flux_window[0] = (ts_old, remaining_energy, remaining_dt)
                self._flux_window_energy -= energy_remove
                self._flux_window_duration -= dt_remove

        duration = self._flux_window_duration if self._flux_window_duration > 0.0 else dt
        if duration <= 0.0:
            return self._last_midplane_flux_raw
        return (self._flux_window_energy / self._midplane_area) / duration

    def _update_cumulative_heat_window(self, timestamp: float, heat_transfer: float) -> None:
        """Keep only the last few seconds of heat contributions in the cumulative total."""
        if not math.isfinite(heat_transfer):
            return
        window = self._heat_cumulative_span
        if window <= 0.0:
            self._cumulative_heat += heat_transfer
            return

        if not math.isfinite(timestamp):
            timestamp = self._elapsed_time

        self._heat_cumulative_window.append((timestamp, heat_transfer))
        self._cumulative_heat += heat_transfer

        cutoff = timestamp - window
        while self._heat_cumulative_window and self._heat_cumulative_window[0][0] < cutoff:
            _, old_value = self._heat_cumulative_window.popleft()
            self._cumulative_heat -= old_value

        # Avoid accumulating tiny residuals over long runs
        if abs(self._cumulative_heat) < 1e-18:
            self._cumulative_heat = 0.0

    def _maybe_write_flux_csv(self, timestamp: float) -> None:
        """Optionally append the latest flux sample to a CSV log."""
        path = self._flux_csv_path
        if not path:
            return
        interval = self._flux_csv_interval
        if interval > 0.0 and timestamp - self._last_flux_csv_time < interval:
            return

        file_path = Path(path).expanduser()
        try:
            if file_path.parent and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            need_header = not file_path.exists() or not self._flux_csv_header_written
            with file_path.open('a', encoding='utf-8') as fh:
                if need_header:
                    fh.write('time,heat_flux,cumulative_heat\n')
                    self._flux_csv_header_written = True
                fh.write(f"{timestamp:.9f},{self._last_midplane_flux:.9g},{self._cumulative_heat:.9g}\n")
        except OSError:
            # Ignore logging errors to keep the simulation running.
            return

        self._last_flux_csv_time = timestamp

    # -------------------------------------------------------------------------
    def reset_midplane_statistics(self) -> None:
        """Clear accumulated flux statistics and heat counters."""
        self._cumulative_heat = 0.0
        self._last_midplane_heat_transfer = 0.0
        self._last_midplane_flux = 0.0
        self._last_midplane_flux_raw = 0.0
        self._flux_history.clear()
        self._flux_window.clear()
        self._flux_window_energy = 0.0
        self._flux_window_duration = 0.0
        self._heat_cumulative_window.clear()
        self._last_midplane_counts = {'positive': 0, 'negative': 0}
        self._last_midplane_crossings = {
            'positive': np.empty(0, dtype=int),
            'negative': np.empty(0, dtype=int),
        }
        self._last_midplane_energy = {'positive': 0.0, 'negative': 0.0}
        # Keep elapsed time so timestamps remain monotonic; callers can
        # decide whether to interpret time relative to the reset moment.

    # -------------------------------------------------------------------------
    def add_particles(self, r: ndarray, v: ndarray, m: ndarray) -> None:
        """Add new gas particles to the simulation.

        New particles are appended at the supplied positions with zeroed
        velocities so that they begin at rest, and the provided masses are
        used for thermal calculations.

        Parameters
        ----------
        r: ndarray
            Positions of the new particles, shape (2, N_new).
        v: ndarray
            Velocities of the new particles, shape (2, N_new).
        m: ndarray
            Masses of the new particles, shape (N_new,).
        """
        if r.shape != v.shape or r.shape[0] != 2 or r.shape[1] != m.shape[0]:
            raise ValueError("Shapes of r, v and m are inconsistent")
        v_zeroed = np.zeros_like(v)
        self._r = np.hstack([self._r, r])
        self._v = np.hstack([self._v, v_zeroed])
        self._m = np.hstack([self._m, m])
        old_count = self._n_particles
        new_count = r.shape[1]
        type_count = len(self._type_definitions)
        if type_count > 0 and new_count > 0:
            new_types = self._choose_types_by_weight(new_count, self._type_weights)
        else:
            new_types = np.zeros((new_count,), dtype=int)
        self._particle_type_indices = np.concatenate([self._particle_type_indices, new_types])
        new_timers = self._sample_time_to_live(new_types)
        existing_timers = getattr(self, "_time_to_live", np.empty((0,), dtype=float))
        self._time_to_live = np.concatenate([existing_timers, new_timers])
        self._n_particles += new_count
        # Recompute collision pairs
        self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def _set_particles_cnt(self, particles_cnt: int) -> None:
        """Reset the number of gas particles to the given count.

        If the new count is smaller than the current count, particles are
        removed from the end.  If it is larger, new particles are added
        with random positions, zero velocities and median mass.
        """
        if particles_cnt < 0:
            raise ValueError("particles_cnt must be >= 0")
        if particles_cnt < self._n_particles:
            idx = slice(particles_cnt)
            self._r = self._r[:, idx]
            self._v = self._v[:, idx]
            self._m = self._m[idx]
            self._particle_type_indices = self._particle_type_indices[:particles_cnt]
            self._time_to_live = self._time_to_live[:particles_cnt]
        if particles_cnt > self._n_particles:
            new_cnt = particles_cnt - self._n_particles
            # Positions uniformly distributed away from walls
            x_new = np.random.uniform(low=self._R, high=self._box_width - self._R, size=new_cnt)
            y_new = np.random.uniform(low=self._R, high=1.0 - self._R, size=new_cnt)
            new_r = np.vstack((x_new, y_new))
            # Use median mass and start the new particles stationary
            new_m = np.full((new_cnt,), np.median(self._m) if self._m.size > 0 else 1.0)
            new_v = np.zeros((2, new_cnt), dtype=float)
            self.add_particles(new_r, new_v, new_m)
        if particles_cnt != self._n_particles:
            self._n_particles = particles_cnt
            self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def set_absorber_zones(self, centers: list[tuple[float, float]], radius: float) -> None:
        """Configure absorbing disks that remove particles entering the footpoints."""
        try:
            radius = float(radius)
        except Exception:
            radius = 0.0
        self._absorber_radius = max(0.0, radius)
        valid = []
        for c in centers:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                continue
            try:
                valid.append((float(c[0]), float(c[1])))
            except Exception:
                continue
        if valid:
            self._absorber_centers = np.array(valid, dtype=float).T
        else:
            self._absorber_centers = np.zeros((2, 0), dtype=float)

    def spawn_particles_uniform(self, count: int) -> None:
        """Add uniformly distributed particles (positions + types) like the initial distribution.

        Velocities are zeroed so the newcomers begin at rest.
        """
        if count <= 0:
            return
        new_cnt = int(count)
        x_new = np.random.uniform(low=self._R, high=self._box_width - self._R, size=new_cnt)
        y_new = np.random.uniform(low=self._R, high=1.0 - self._R, size=new_cnt)
        new_r = np.vstack((x_new, y_new))
        new_m = np.full((new_cnt,), np.median(self._m) if self._m.size > 0 else 1.0)
        new_v = np.zeros((2, new_cnt), dtype=float)
        self.add_particles(new_r, new_v, new_m)

    def set_footpoint_field(self, centers: list[tuple[float, float]], radius: float, strengths: list[float] | None = None) -> None:
        """Update magnetic footpoint profile used for gradient calculations."""
        try:
            radius = float(radius)
        except Exception:
            radius = 0.0
        self._foot_radius_field = max(0.0, radius)
        valid_centers = []
        for c in centers:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                continue
            try:
                valid_centers.append((float(c[0]), float(c[1])))
            except Exception:
                continue
        if valid_centers:
            self._foot_centers_field = np.array(valid_centers, dtype=float).T
        else:
            self._foot_centers_field = np.zeros((2, 0), dtype=float)
        self._update_strength_array(strengths, target_attr='_foot_strengths')

    def _update_strength_array(self, strengths, target_attr: str) -> None:
        """Helper to align strength arrays with current footpoint count."""
        target_size = self._foot_centers_field.shape[1]
        if strengths is None:
            setattr(self, target_attr, np.zeros((target_size,), dtype=float))
            return
        try:
            arr = np.array(strengths, dtype=float)
        except Exception:
            arr = np.zeros((target_size,), dtype=float)
        if arr.ndim == 0:
            arr = np.full((target_size,), float(arr))
        if arr.shape[0] < target_size:
            pad = target_size - arr.shape[0]
            arr = np.pad(arr, (0, pad))
        elif arr.shape[0] > target_size:
            arr = arr[:target_size]
        setattr(self, target_attr, arr)

    # -------------------------------------------------------------------------
    def set_params(
        self,
        gamma: float = None,
        k: float = None,
        l_0: float = None,
        R: float = None,
        T: float = None,
        m: float = None,
        particles_cnt: int = None,
        T_left: float = None,
        T_right: float = None,
        magnetic_gradient: float = None,
        lambda_param: float = None,
    ) -> None:
        """Update simulation parameters on the fly.

        Parameters correspond to those accepted by the constructor.  Any
        parameter passed as ``None`` will be left unchanged.
        """
        if gamma is not None:
            self._gamma = float(gamma)
        if k is not None:
            self._k = float(k)
        if l_0 is not None:
            self._l_0 = float(l_0)
        if R is not None:
            self._R = float(R)
            min_width = max(2.0 * self._R + 1e-6, 1e-3)
            if self._box_width < min_width:
                self.set_box_width(min_width)
        if T is not None:
            self.T = float(T)
        if m is not None:
            if m <= 0:
                raise ValueError("m must be > 0")
            self._m[:] = float(m)
        if particles_cnt is not None:
            self._set_particles_cnt(int(particles_cnt))
        if T_left is not None:
            self.T_left = float(T_left)
        if T_right is not None:
            self.T_right = float(T_right)
        if magnetic_gradient is not None:
            # Keep the background field uniform despite any incoming gradient updates.
            try:
                _ = float(magnetic_gradient)
            except Exception:
                pass
            self._grad_B = 0.0
        if lambda_param is not None:
            self.set_lambda_param(lambda_param)
        # Recompute full energy after parameter changes
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def set_box_width(self, width: float) -> None:
        """Adjust the horizontal size of the simulation domain."""
        width = float(max(width, 2.0 * self._R + 1e-6))
        if width <= 0:
            width = 1.0
        if abs(width - self._box_width) < 1e-9:
            return
        old_width = self._box_width
        scale = width / old_width if old_width > 0 else 1.0
        self._box_width = width
        self._r[0] *= scale
        self._r[0] = np.clip(self._r[0], self._R, self._box_width - self._R)
        if self._midplane_axis == 'x':
            self._midplane_position = float(np.clip(self._midplane_position * scale, self._R, self._box_width - self._R))
        else:
            self._midplane_position = float(np.clip(self._midplane_position, self._R, 1.0 - self._R))
        if self._midplane_axis == 'y' and not self._midplane_area_locked:
            self._midplane_area = float(max(self._box_width, 1e-12))
        self._init_ids_pairs()

    # -------------------------------------------------------------------------
    def expected_potential_energy(self) -> float:
        """Return zero since no external potential exists."""
        return 0.0

    def expected_kinetic_energy(self) -> float:
        """Return the expected kinetic energy per particle (k_B T)."""
        return float(self._k_boltz * self._T_tar)

    def calc_kinetic_energy(self) -> float:
        """Calculate mean kinetic energy of gas particles."""
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def calc_full_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of gas particles."""
        return np.sum((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def _fix_energy(self) -> None:
        """Gently counteract numerical drift without blocking heat exchange.

        The total energy should change only due to wall interactions or
        deliberate parameter updates.  We therefore track a slowly varying
        target energy and only rescale velocities when the instantaneous
        energy deviates slightly from that target (typical of integration
        error).  Substantial changes driven by the walls are preserved.
        """
        current_E = self.calc_full_energy()
        if current_E <= 0.0:
            return

        if self._E_full <= 0.0:
            self._E_full = current_E
            return

        relax = 0.1
        self._E_full = (1.0 - relax) * self._E_full + relax * current_E
        scale = self._E_full / current_E
        if scale <= 0.0:
            return
        scale = math.sqrt(scale)
        if abs(scale - 1.0) < 0.05:
            self._v *= scale

    def calc_full_energy(self) -> float:
        """Return the total energy (purely kinetic)."""
        return self.calc_full_kinetic_energy()

    def calc_potential_energy(self) -> float:
        """Return zero since there is no potential energy."""
        return 0.0

    def mean_potential_energy(self, frames_c: Union[int, None] = None) -> float:
        """Always return zero in absence of potential energy."""
        return 0.0

    def mean_kinetic_energy(self, frames_c: Union[int, None] = None) -> float:
        """Return the mean of the stored kinetic energy history."""
        if frames_c is None:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy))
        else:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy[-frames_c:]))
