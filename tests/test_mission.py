"""Tests for Step 1.4. Run with:  pytest -q"""
import copy
from pathlib import Path
import numpy as np
import pytest
from swarm_intercept.config import load_config, check_config
from swarm_intercept.control.guidance import intercept_time, pursuit_velocity, orbit_velocity
from swarm_intercept.mission.allocator import intercept_costs, choose_interceptor
from swarm_intercept.mission.fsm import SURVEIL, INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.core.sensors import area_measurement
from swarm_intercept.core.estimator import TargetKF
from swarm_intercept.core.target import PATTERNS
from swarm_intercept.sim_mission import simulate_mission

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers


def test_intercept_time_stationary_target():
    assert np.isclose(intercept_time([0, 0], [3, 4], [0, 0], 0.5), 10.0)      # 5 m at 0.5 m/s


def test_intercept_time_moving_target():
    # target 2 m ahead running away at 0.1 m/s, drone at 0.5 m/s: closing speed 0.4 -> 5 s
    assert np.isclose(intercept_time([0, 0], [2, 0], [0.1, 0], 0.5), 5.0)
    # target coming toward the drone: closing speed 0.6
    assert np.isclose(intercept_time([0, 0], [3, 0], [-0.1, 0], 0.5), 5.0)


def test_faster_target_running_away_cannot_be_caught():
    assert intercept_time([0, 0], [2, 0], [0.8, 0], 0.5) == np.inf


def test_pursuit_aims_ahead_of_the_target():
    u, aim, t_go = pursuit_velocity([0, 0], [2, 0], [0, 0.2], 0.5)
    assert aim[1] > 0 and u[1] > 0                     # leads the target, which is moving in +y
    assert np.isclose(np.linalg.norm(u), 0.5)
    assert np.isclose(np.linalg.norm(aim), 0.5 * t_go) # the drone arrives exactly when the target does


def test_orbit_velocity_adds_target_velocity():
    u, e = orbit_velocity([0.4, 0.0], [0.0, 0.0], [0.1, 0.0], 0.4, 1.5, 0.3, direction=1)
    assert abs(e) < 1e-12
    assert np.allclose(u, [0.1, 0.3])                  # tangent (0, 0.3) + target velocity (0.1, 0)


def test_allocator_picks_the_drone_that_gets_there_first():
    states = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, np.pi], [0.0, 5.0, 0.0]])
    costs = intercept_costs(states, [1.0, 0.0], [0.0, 0.0], 0.5, 1.5)
    assert choose_interceptor(costs) == 0
    costs = intercept_costs(states, [1.0, 0.0], [0.0, 0.0], 0.5, 1.5, eligible=[False, True, True])
    assert choose_interceptor(costs) == 1 and costs[0] == np.inf


def test_allocator_counts_the_time_to_turn_round():
    # both drones are 1 m from the target; drone 0 faces it, drone 1 faces away
    states = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    costs = intercept_costs(states, [0.0, 0.0], [0.0, 0.0], 0.5, 1.5)
    assert costs[0] < costs[1]


def test_area_sensor_only_inside_the_zone():
    rng = np.random.default_rng(0)
    assert area_measurement([1.0, -1.2], 1.5, 0.1, 0.0, rng) is not None
    assert area_measurement([1.7, 0.0], 1.5, 0.1, 0.0, rng) is None


def test_kf_trusts_an_accurate_measurement_more():
    a, b = TargetKF(0.03, 0.2), TargetKF(0.03, 0.2)
    for kf in (a, b):
        kf.update([0.0, 0.0]); kf.predict(0.5)
    a.update([1.0, 0.0], sigma=0.03)
    b.update([1.0, 0.0], sigma=0.30)
    assert a.position[0] > b.position[0] > 0.0


def test_too_fast_target_is_rejected():
    bad = copy.deepcopy(cfg)
    bad["target"]["speed"] = 0.6
    with pytest.raises(ValueError):
        check_config(bad)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_full_mission(pattern):
    lg, fsm = simulate_mission(cfg, pattern, T=90.0)
    texts = [e for _, e in fsm.events]
    assert len(texts) == 4
    assert "SURVEIL -> INTERCEPT" in texts[0] and "INTERCEPT -> INVESTIGATE" in texts[1]
    assert "INVESTIGATE -> RETURN" in texts[2] and "RETURN -> SURVEIL" in texts[3]
    mode = lg["mode"]
    assert (mode != SURVEIL).sum(axis=1).max() == 1                    # never more than one drone away
    inv = (mode == INVESTIGATE).any(axis=1)
    assert abs(np.nanmean(lg["dist_true"][inv]) - cfg["mission"]["orbit_radius"]) < 0.15
    i_ret = np.where((mode == RETURN).any(axis=1))[0][0]               # just before the drone returns
    two = np.degrees(lg["gap"][i_ret - 1]); two = two[~np.isnan(two)]
    assert len(two) == 2 and np.allclose(two, 180.0, atol=5.0)         # the other two had re-spaced
    assert np.allclose(np.degrees(lg["gap"][-1]), 120.0, atol=2.0)     # and everyone is back at 120 deg
    assert lg["min_sep"][lg["present"]].min() >= 0.45                  # 3-D separation kept during the mission


def test_nothing_happens_before_the_target_enters():
    lg, fsm = simulate_mission(cfg, "straight", T=29.0)
    assert len(fsm.events) == 0 and not lg["tracked"].any()
