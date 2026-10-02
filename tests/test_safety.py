"""Tests for Step 1.6. Run with:  pytest -q"""
import copy
from pathlib import Path
import numpy as np
import pytest
from swarm_intercept.config import load_config
from swarm_intercept.core.dynamics import unicycle_step
from swarm_intercept.control.safety import solve_qp_2d, safety_filter, lookahead_point
from swarm_intercept.sim_encounter import simulate_encounter, ENCOUNTERS
from swarm_intercept.sim_mission import simulate_mission

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers
D_SAFE = cfg["safety"]["d_safe"]


def test_qp_returns_nominal_when_it_is_allowed():
    x, ok = solve_qp_2d([1.0, 1.0], np.array([[1.0, 0.0]]), np.array([0.0]))       # constraint x0 >= 0
    assert ok and np.allclose(x, [1.0, 1.0])


def test_qp_projects_onto_a_violated_constraint():
    x, ok = solve_qp_2d([-1.0, 1.0], np.array([[1.0, 0.0]]), np.array([0.0]))
    assert ok and np.allclose(x, [0.0, 1.0])                                        # nearest allowed point


def test_qp_two_constraints_meet_in_a_corner():
    A, b = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0])
    x, ok = solve_qp_2d([-1.0, -2.0], A, b)
    assert ok and np.allclose(x, [0.0, 0.0])


def test_qp_reports_when_nothing_is_allowed():
    A, b = np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([1.0, 1.0])                # x0 >= 1 and x0 <= -1
    _, ok = solve_qp_2d([0.0, 0.0], A, b)
    assert not ok


def test_lookahead_point():
    assert np.allclose(lookahead_point([1.0, 2.0, np.pi / 2], 0.1), [1.0, 2.1])


def test_filter_does_nothing_when_drones_are_far_apart():
    states = np.array([[-1.0, 0.0, 0.0], [1.0, 0.5, np.pi], [0.0, -1.2, 1.0]])
    last = np.tile([0.4, 0.0], (3, 1))
    v, w, changed = safety_filter(0, states, last, 0.4, 0.3, cfg, 0.55)
    assert not changed and v == 0.4 and w == 0.3


def test_filter_acts_when_drones_are_on_a_collision_course():
    states = np.array([[-0.45, 0.0, 0.0], [0.45, 0.0, np.pi]])
    last = np.tile([0.4, 0.0], (2, 1))
    v, w, changed = safety_filter(1, states, last, 0.4, 0.0, cfg, 0.55)
    assert changed


@pytest.mark.parametrize("name", list(ENCOUNTERS))
def test_encounters_stay_separated(name):
    starts, goals = ENCOUNTERS[name]
    unsafe = simulate_encounter(starts, goals, cfg, use_filter=False, T=12.0)
    safe = simulate_encounter(starts, goals, cfg, use_filter=True, T=12.0)
    assert unsafe["min_sep"].min() < 0.1              # without the filter they (nearly) collide
    assert safe["min_sep"].min() >= D_SAFE            # with it they never get closer than d_safe


def test_the_drone_with_right_of_way_is_not_delayed():
    starts, goals = ENCOUNTERS["crossing"]
    free = simulate_encounter(starts, goals, cfg, use_filter=False, T=20.0)
    safe = simulate_encounter(starts, goals, cfg, use_filter=True, T=20.0)
    assert safe["arrived"].all()
    assert abs(safe["t_arrive"][0] - free["t_arrive"][0]) < 0.5      # drone 1 has right of way
    assert safe["t_arrive"][1] > free["t_arrive"][1] + 1.0           # drone 2 gave way


def test_wall_barrier_turns_a_drone_back():
    states = np.array([[0.0, 1.0, np.pi / 2]])        # one drone, told to fly straight ahead at full speed
    last = np.array([[0.4, 0.0]])
    far = 0.0
    for _ in range(2000):                             # 40 s: it ends up following the walls round the arena
        v, w, _ = safety_filter(0, states, last, 0.55, 0.0, cfg, 0.55)
        states[0] = unicycle_step(states[0], v, w, cfg["sim"]["dt"], cfg["drone"]["omega_max"])
        far = max(far, np.abs(states[0, :2]).max())
    assert far <= cfg["arena"]["half_size"]


def test_mission_is_separated_with_the_filter_and_not_without():
    on, fsm = simulate_mission(cfg, "straight", T=80.0)
    off_cfg = copy.deepcopy(cfg)
    off_cfg["safety"]["enabled"] = False
    off, _ = simulate_mission(off_cfg, "straight", T=80.0)
    assert on["min_sep_xy"].min() >= D_SAFE
    assert off["min_sep_xy"].min() < D_SAFE
    assert len(fsm.events) == 4                                       # the mission still completes
    assert np.allclose(np.degrees(on["gap"][-1]), 120.0, atol=2.0)
    assert 0.0 < on["filtered"].any(axis=1).mean() < 0.3              # the filter is active only part of the time
