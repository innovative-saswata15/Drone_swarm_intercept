"""Closed-loop simulation of several drones holding formation on the circle."""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step, velocity
from swarm_intercept.control.gvf import desired_heading
from swarm_intercept.control.heading import heading_control
from swarm_intercept.control.formation import phase_angles, spacing_speeds


def formation_tick(states, active, cfg):
    """Advance all drones by one time step of formation flight.

    states  array (N, 3), modified in place
    active  boolean mask (N,): True = in formation, False = on the inner holding circle
    Returns (speeds, gap_ahead, e, omega), each of shape (N,).
    """
    dt = cfg["sim"]["dt"]
    v0, omega_max = cfg["drone"]["v"], cfg["drone"]["omega_max"]
    c = np.array(cfg["arena"]["center"], dtype=float)
    f = cfg["formation"]
    R, k, k_gamma, direction = f["radius"], f["k_gvf"], f["k_heading"], f["direction"]
    N = len(states)
    e, omega = np.zeros(N), np.zeros(N)

    phis = phase_angles(states[:, :2], c)
    speeds, gap_ahead = spacing_speeds(phis, active, v0, f["k_spacing"], f["v_min"], f["v_max"], direction)
    for j in range(N):
        radius = R if active[j] else f["hold_radius"]
        theta_d, omega_d, e[j] = desired_heading(states[j, :2], velocity(states[j], speeds[j]),
                                                 c, radius, k, direction, dt)
        omega_cmd, _ = heading_control(states[j, 2], theta_d, omega_d, k_gamma)
        omega[j] = float(np.clip(omega_cmd, -omega_max, omega_max))
    for j in range(N):
        states[j] = unicycle_step(states[j], speeds[j], omega[j], dt, omega_max)
    return speeds, gap_ahead, e, omega


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

        log["x"][i], log["y"][i], log["theta"][i] = states[:, 0], states[:, 1], states[:, 2]
        speeds, gap_ahead, e, omega = formation_tick(states, active, cfg)
        log["e"][i], log["omega"][i] = e, omega
        log["v"][i], log["gap"][i], log["active"][i] = speeds, gap_ahead, active
        d = np.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=2)
        log["min_sep"][i] = d[np.triu_indices(N, 1)].min() if N > 1 else np.inf

    return log
