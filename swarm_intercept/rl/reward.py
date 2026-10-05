"""Reward for the residual agent.

The reward may use the TRUE target position: it is only used for learning in simulation, the
policy itself never sees it (see observation.py). Phases the agent cannot speed up (the fixed
investigation time) carry no time penalty.
"""
import numpy as np
from swarm_intercept.mission.fsm import INTERCEPT, INVESTIGATE, RETURN


def step_reward(sim, j, mode_pre, d_pre, e_pre, rw, completed):
    """Reward for ONE control step of drone j that was in mode_pre before the step.

    d_pre, e_pre  true range to the target and signed formation-circle error BEFORE the step.
    Called after sim.apply(), so sim.states / sim.applied / sim.filtered describe the result.
    """
    cfg = sim.cfg
    dt = sim.dt
    p = sim.states[j, :2]
    d_post = float(np.linalg.norm(p - sim.target.pos))
    r = 0.0
    if mode_pre == INTERCEPT:
        r += rw["progress"] * (d_pre - d_post) - rw["time"] * dt
    elif mode_pre == INVESTIGATE:
        r += rw["view"] * dt * float(d_post <= cfg["sensor"]["fov_radius"])
        r -= rw["orbit"] * dt * abs(d_post - cfg["mission"]["orbit_radius"])
    elif mode_pre == RETURN:
        e_post = float(np.linalg.norm(p - sim.c) - cfg["formation"]["radius"])
        r += rw["return_progress"] * (abs(e_pre) - abs(e_post)) - rw["time"] * dt
    r -= rw["effort"] * float(sim.applied[j, 1]) ** 2 * dt
    r -= rw["filter"] * dt * float(sim.filtered[j])
    margin = cfg["safety"]["d_safe"] + rw["sep_soft"]
    r -= rw["separation"] * dt * max(0.0, margin - sim.min_sep_xy)
    if completed:
        r += rw["done"]
    return float(r)
