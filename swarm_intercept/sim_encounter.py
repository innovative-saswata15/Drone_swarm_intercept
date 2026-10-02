"""Small test scenarios for the safety filter: drones flying straight at fixed goals."""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step
from swarm_intercept.control.guidance import pure_pursuit_velocity, follow_velocity
from swarm_intercept.control.safety import safety_filter


def simulate_encounter(starts, goals, cfg, use_filter=True, T=20.0):
    """Each drone flies to its own goal point at cruise speed.

    starts  array (N, 3) of [x, y, theta];  goals  array (N, 2)
    Returns a dict with the paths, the minimum distance and whether each drone arrived.
    """
    dt, v0 = cfg["sim"]["dt"], cfg["drone"]["v"]
    omega_max, k_gamma = cfg["drone"]["omega_max"], cfg["formation"]["k_heading"]
    states = np.array(starts, dtype=float)
    goals = np.array(goals, dtype=float)
    N = len(states)
    n = int(round(T / dt))
    last_cmd = np.column_stack([np.full(N, v0), np.zeros(N)])
    path = np.zeros((n, N, 2))
    min_sep = np.zeros(n)
    arrived = np.zeros(N, dtype=bool)
    t_arrive = np.full(N, np.nan)
    iu = np.triu_indices(N, 1)

    for i in range(n):
        path[i] = states[:, :2]
        min_sep[i] = np.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=2)[iu].min()
        applied = np.zeros((N, 2))
        for j in range(N):
            if not arrived[j] and np.linalg.norm(goals[j] - states[j, :2]) < 0.15:
                arrived[j], t_arrive[j] = True, i * dt
            u = pure_pursuit_velocity(states[j, :2], goals[j], v0)
            p_next = states[j, :2] + v0 * np.array([np.cos(states[j, 2]), np.sin(states[j, 2])]) * dt
            v, omega, _ = follow_velocity(states[j, 2], u, pure_pursuit_velocity(p_next, goals[j], v0), dt, k_gamma, v0, v0)
            omega = float(np.clip(omega, -omega_max, omega_max))
            if use_filter:
                v, omega, _ = safety_filter(j, states, last_cmd, v, omega, cfg, v0)
            applied[j] = v, omega
        for j in range(N):
            if not arrived[j]:
                states[j] = unicycle_step(states[j], applied[j, 0], applied[j, 1], dt, omega_max)
            else:
                applied[j] = 0.0, 0.0
        last_cmd = applied
    return {"path": path, "min_sep": min_sep, "arrived": arrived, "t_arrive": t_arrive, "t": np.arange(n) * dt}


ENCOUNTERS = {
    "head-on":  (np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, np.pi]]), np.array([[1.5, 0.0], [-1.5, 0.0]])),
    "crossing": (np.array([[-1.5, 0.0, 0.0], [0.0, -1.5, np.pi / 2]]), np.array([[1.5, 0.0], [0.0, 1.5]])),
    "three-way": (np.array([[-1.5, 0.0, 0.0], [0.75, 1.3, -2.094], [0.75, -1.3, 2.094]]),
                  np.array([[1.5, 0.0], [-0.75, -1.3], [-0.75, 1.3]])),
}
