"""Closed-loop simulation of several drones holding formation on the circle."""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step, velocity
from swarm_intercept.control.gvf import desired_heading
from swarm_intercept.control.heading import heading_control
from swarm_intercept.control.formation import phase_angles, spacing_speeds


def simulate_formation(states0, cfg, leave=None, T=None):
    """Simulate N drones.

    states0  array (N, 3): initial [x, y, theta] of every drone
    leave    optional (drone_index, t_leave, t_return): that drone leaves the
             formation at t_leave (it flies a small inner 'holding' circle) and
             rejoins at t_return. Use None to keep everyone in formation.
    Returns a dict of logged arrays; per-drone logs have shape (steps, N).
    """
    dt = cfg["sim"]["dt"]
    T = cfg["sim"]["T"] if T is None else T
    v0 = cfg["drone"]["v"]
    omega_max = cfg["drone"]["omega_max"]
    c = np.array(cfg["arena"]["center"], dtype=float)
    f = cfg["formation"]
    R, k, k_gamma, direction = f["radius"], f["k_gvf"], f["k_heading"], f["direction"]

    states = np.array(states0, dtype=float)
    N = len(states)
    n = int(round(T / dt))
    log = {key: np.zeros((n, N)) for key in ("x", "y", "theta", "e", "omega", "v", "gap")}
    log["t"] = np.arange(n) * dt
    log["active"] = np.zeros((n, N), dtype=bool)
    log["min_sep"] = np.zeros(n)

    for i in range(n):
        t = i * dt
        active = np.ones(N, dtype=bool)
        if leave is not None and leave[1] <= t < leave[2]:
            active[leave[0]] = False

        phis = phase_angles(states[:, :2], c)
        speeds, gap_ahead = spacing_speeds(phis, active, v0, f["k_spacing"], f["v_min"], f["v_max"], direction)

        for j in range(N):
            radius = R if active[j] else f["hold_radius"]
            theta_d, omega_d, e = desired_heading(states[j, :2], velocity(states[j], speeds[j]),
                                                  c, radius, k, direction, dt)
            omega_cmd, _ = heading_control(states[j, 2], theta_d, omega_d, k_gamma)
            omega = float(np.clip(omega_cmd, -omega_max, omega_max))
            log["e"][i, j], log["omega"][i, j] = e, omega
            log["x"][i, j], log["y"][i, j], log["theta"][i, j] = states[j]
            states[j] = unicycle_step(states[j], speeds[j], omega, dt, omega_max)

        log["v"][i], log["gap"][i], log["active"][i] = speeds, gap_ahead, active
        d = np.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=2)
        log["min_sep"][i] = d[np.triu_indices(N, 1)].min() if N > 1 else np.inf

    return log
