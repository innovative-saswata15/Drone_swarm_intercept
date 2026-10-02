"""Interceptor selection: which drone should leave the formation?

Cost of drone i = time to turn toward the intercept point + time to fly there.
The drone with the smallest cost is chosen. The decision is made once and then
kept (no switching mid-flight).
"""
import numpy as np
from swarm_intercept.utils.angles import wrap
from swarm_intercept.control.guidance import pursuit_velocity


def intercept_costs(states, p_t, v_t, speed, omega_max, eligible=None):
    """Estimated time [s] for each drone to reach the target. np.inf if it cannot, or is not eligible.

    states    array (N, 3) of [x, y, theta]
    p_t, v_t  estimated target position and velocity
    """
    N = len(states)
    eligible = np.ones(N, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    costs = np.full(N, np.inf)
    for i in range(N):
        if not eligible[i]:
            continue
        u, aim, t_go = pursuit_velocity(states[i, :2], p_t, v_t, speed)
        if np.linalg.norm(u) < 1e-9:
            continue
        turn = abs(wrap(np.arctan2(u[1], u[0]) - states[i, 2])) / omega_max     # time to point at the aim point
        costs[i] = turn + t_go
    return costs


def choose_interceptor(costs):
    """Index of the cheapest drone, or None if nobody can reach the target."""
    i = int(np.argmin(costs))
    return i if np.isfinite(costs[i]) else None
