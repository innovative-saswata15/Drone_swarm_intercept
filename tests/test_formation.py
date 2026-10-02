"""Tests for Step 1.2. Run with:  pytest -q"""
from pathlib import Path
import numpy as np
from swarm_intercept.config import load_config
from swarm_intercept.control.formation import ring_gaps, spacing_speeds
from swarm_intercept.sim_multi import simulate_formation

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers, independent of configs/default.yaml
f = cfg["formation"]
V0 = cfg["drone"]["v"]
STARTS = np.array([[1.9, 0.0, 2.0], [1.5, 1.0, -1.0], [0.3, -0.2, 0.5]])


def test_gaps_sum_to_full_circle():
    phis = np.array([0.1, 2.0, -2.5])
    ahead, behind = ring_gaps(phis, [True, True, True])
    assert np.isclose(ahead.sum(), 2 * np.pi) and np.isclose(behind.sum(), 2 * np.pi)


def test_gap_values():
    phis = np.radians([0.0, 90.0, 180.0])
    ahead, behind = ring_gaps(phis, [True, True, True], direction=1)
    assert np.allclose(np.degrees(ahead), [90, 90, 180])
    assert np.allclose(np.degrees(behind), [180, 90, 90])


def test_inactive_drone_is_ignored():
    phis = np.radians([0.0, 90.0, 180.0])
    ahead, _ = ring_gaps(phis, [True, False, True])
    assert np.isnan(ahead[1]) and np.allclose(np.degrees(ahead[[0, 2]]), [180, 180])


def test_equal_spacing_gives_cruise_speed():
    phis = np.radians([0.0, 120.0, 240.0])
    v, _ = spacing_speeds(phis, [True] * 3, V0, f["k_spacing"], f["v_min"], f["v_max"])
    assert np.allclose(v, V0)


def test_speed_up_when_gap_ahead_is_larger():
    phis = np.radians([0.0, 60.0, 120.0])           # drone 3 has a 240 deg gap ahead
    v, _ = spacing_speeds(phis, [True] * 3, V0, f["k_spacing"], f["v_min"], f["v_max"])
    assert v[2] > V0 and v[0] < V0


def test_three_drones_reach_120_deg():
    lg = simulate_formation(STARTS, cfg, T=60.0)
    assert np.allclose(np.degrees(lg["gap"][-1]), 120.0, atol=1.0)
    assert np.all(np.abs(lg["e"][-1]) < 0.02)


def test_two_drones_reach_180_deg_then_back_to_120():
    lg = simulate_formation(STARTS, cfg, leave=(2, 50.0, 100.0), T=150.0)
    t = lg["t"]
    i = np.searchsorted(t, 100.0) - 1
    assert np.allclose(np.degrees(lg["gap"][i, :2]), 180.0, atol=1.0)
    assert np.allclose(np.degrees(lg["gap"][-1]), 120.0, atol=1.0)


def test_speed_limits_respected():
    lg = simulate_formation(STARTS, cfg, leave=(2, 50.0, 100.0), T=150.0)
    assert lg["v"].min() >= f["v_min"] - 1e-12 and lg["v"].max() <= f["v_max"] + 1e-12


def test_bad_config_is_rejected():
    import copy, pytest
    from swarm_intercept.config import check_config
    bad = copy.deepcopy(cfg)
    bad["formation"]["radius"] = 2.0            # circle would touch the arena edge
    with pytest.raises(ValueError):
        check_config(bad)


def test_formation_works_at_other_speed_and_radius():
    import copy
    from swarm_intercept.config import load_config as _lc
    other = copy.deepcopy(cfg)
    other["drone"]["v"] = 0.1
    other["formation"]["radius"] = 1.6
    fm = other["formation"]
    fm["k_spacing"], fm["v_min"], fm["v_max"] = (fm[k] * 0.1 for k in ("k_spacing_rel", "v_min_rel", "v_max_rel"))
    scale = (1.6 / 0.1) / 3.0
    lg = simulate_formation(STARTS, other, T=60.0 * scale)
    assert np.allclose(np.degrees(lg["gap"][-1]), 120.0, atol=1.0)
