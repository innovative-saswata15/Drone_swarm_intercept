"""Tests for Step 1.5. Run with:  pytest -q"""
from pathlib import Path
import numpy as np
import pytest
from swarm_intercept.config import load_config
from swarm_intercept.core.dynamics import unicycle_step
from swarm_intercept.control.guidance import (LAWS, pure_pursuit_velocity, los_rate_and_closing_speed,
                                               guidance_command)
from swarm_intercept.sim_intercept import simulate_intercept, random_case

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers
DT = cfg["sim"]["dt"]


def fly(law, target0, v_t, speed=0.55, heading=0.0, capture=0.5, t_max=60.0):
    """Chase a constant-velocity target with PERFECT information. Returns the capture time."""
    state, p_t, v_t = np.array([0.0, 0.0, heading]), np.array(target0, dtype=float), np.array(v_t, dtype=float)
    for k in range(int(t_max / DT)):
        if np.linalg.norm(state[:2] - p_t) <= capture:
            return k * DT
        v, om = guidance_command(law, state, speed, p_t, v_t, DT, 3.0, 3.0)
        state = unicycle_step(state, v, om, DT, 1.5)
        p_t = p_t + v_t * DT
    return np.nan


def test_pure_pursuit_points_at_the_target():
    u = pure_pursuit_velocity([1.0, 1.0], [1.0, 3.0], 0.5)
    assert np.allclose(u, [0.0, 0.5])


def test_los_geometry():
    # target straight ahead and stationary, drone flying at it: LOS does not rotate, range closes at the drone speed
    lam_dot, closing = los_rate_and_closing_speed([0, 0], [0.5, 0], [4, 0], [0, 0])
    assert np.isclose(lam_dot, 0.0) and np.isclose(closing, 0.5)
    # target crossing to the left (+y): the LOS rotates counter-clockwise
    lam_dot, _ = los_rate_and_closing_speed([0, 0], [0.5, 0], [4, 0], [0, 0.2])
    assert lam_dot > 0


def test_pn_commands_no_turn_on_a_collision_course():
    # drone velocity chosen so that it meets the target: no LOS rotation -> no turn
    v_t, speed = np.array([0.0, 0.3]), 0.5
    theta = np.arcsin(0.3 / 0.5)
    v, om = guidance_command("pn", [0.0, 0.0, theta], speed, [4.0, 0.0], v_t, DT, 3.0, 3.0)
    assert abs(om) < 1e-9


def test_unknown_law_is_rejected():
    with pytest.raises(ValueError):
        guidance_command("magic", [0, 0, 0], 0.5, [1, 0], [0, 0], DT, 3.0)


@pytest.mark.parametrize("law", LAWS)
def test_every_law_catches_a_crossing_target(law):
    assert np.isfinite(fly(law, [4.0, 0.0], [0.0, 0.3]))


@pytest.mark.parametrize("law", LAWS)
def test_every_law_turns_round_when_facing_away(law):
    assert np.isfinite(fly(law, [4.0, 0.0], [0.0, 0.2], heading=np.pi))


def test_lead_beats_pure_on_a_straight_crossing_target():
    t_pure, t_lead = fly("pure", [4.0, 0.0], [0.0, 0.3]), fly("lead", [4.0, 0.0], [0.0, 0.3])
    assert t_lead < t_pure - 0.3
    # and it is close to the theoretical minimum for a straight-line intercept
    assert abs(t_lead - 3.5 / np.sqrt(0.55 ** 2 - 0.3 ** 2)) < 0.3


def test_benchmark_gives_every_law_the_same_case():
    a, b = simulate_intercept(cfg, "pure", "large", 5), simulate_intercept(cfg, "pn", "large", 5)
    n = min(len(a["target_path"]), len(b["target_path"]))
    assert np.array_equal(a["target_path"][:n], b["target_path"][:n])       # identical target path
    assert np.array_equal(a["path"][0], b["path"][0])                        # identical drone start
    c = random_case("room", 5)
    assert 1.0 <= np.linalg.norm(c["target"] - c["drone"][:2]) <= 3.0


@pytest.mark.parametrize("law", LAWS)
def test_benchmark_success_rate(law):
    caught = [simulate_intercept(cfg, law, "room", s)["captured"] for s in range(20)]
    assert all(caught)
