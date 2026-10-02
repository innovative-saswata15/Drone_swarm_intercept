"""Load the YAML configuration file, add derived values and check feasibility."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
REFERENCE_R_OVER_V = 3.0      # R/v of the reference setup (1.2 m / 0.4 m/s), used to scale run times
ARENA_MARGIN = 0.3            # the formation circle must stay this far inside the arena edge [m]


def load_config(path=None):
    """Return the configuration as a nested dict. Uses configs/default.yaml if no path is given.

    Adds these derived values:
      formation.k_spacing, formation.v_min, formation.v_max   (absolute, from the *_rel entries)
      sim.time_scale   how much slower/faster this setup is than the reference one
                       (scripts multiply their run times by it)
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    v = cfg["drone"]["v"]
    fm = cfg["formation"]
    fm["k_spacing"] = fm["k_spacing_rel"] * v
    fm["v_min"] = fm["v_min_rel"] * v
    fm["v_max"] = fm["v_max_rel"] * v
    cfg["sim"]["time_scale"] = (fm["radius"] / v) / REFERENCE_R_OVER_V
    check_config(cfg)
    return cfg


def check_config(cfg):
    """Raise ValueError with a plain explanation if the numbers cannot work together."""
    v, w_max = cfg["drone"]["v"], cfg["drone"]["omega_max"]
    fm, half = cfg["formation"], cfg["arena"]["half_size"]
    R, hold = fm["radius"], fm["hold_radius"]
    problems = []
    if R + ARENA_MARGIN > half:
        problems.append(f"formation.radius = {R} m does not fit in the arena: need radius + {ARENA_MARGIN} <= "
                        f"arena.half_size = {half}. Reduce the radius or set half_size >= {R + ARENA_MARGIN:.1f}.")
    if hold > R - ARENA_MARGIN:
        problems.append(f"formation.hold_radius = {hold} m must be at least {ARENA_MARGIN} m smaller than "
                        f"formation.radius = {R} m.")
    if not (0 < fm["v_min_rel"] < 1 < fm["v_max_rel"]):
        problems.append("need 0 < v_min_rel < 1 < v_max_rel so the drones can both slow down and speed up.")
    tightest = min(R, hold)
    if fm["v_max"] / tightest > w_max:
        problems.append(f"the drone cannot turn tightly enough: v_max / radius = {fm['v_max'] / tightest:.2f} rad/s "
                        f"exceeds drone.omega_max = {w_max}. Lower the speed, enlarge the circle or raise omega_max.")
    if problems:
        raise ValueError("configs problem:\n  - " + "\n  - ".join(problems))
