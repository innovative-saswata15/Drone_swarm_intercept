"""Closed-loop simulation of ONE drone following the circle GVF."""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step, velocity
from swarm_intercept.control.gvf import desired_heading
from swarm_intercept.control.heading import heading_control


def simulate_single(state0, cfg, k_gvf=None, use_feedforward=True, T=None):
    """Run one drone from state0 = [x, y, theta]. Returns a dict of logged arrays."""
    dt = cfg["sim"]["dt"]
    T = cfg["sim"]["T"] if T is None else T
    v = cfg["drone"]["v"]
    omega_max = cfg["drone"]["omega_max"]
    c = np.array(cfg["arena"]["center"], dtype=float)
    R = cfg["formation"]["radius"]
    k = cfg["formation"]["k_gvf"] if k_gvf is None else k_gvf
    k_gamma = cfg["formation"]["k_heading"]
    direction = cfg["formation"]["direction"]

    n = int(round(T / dt))
    state = np.array(state0, dtype=float)
    log = {key: np.zeros(n) for key in ("t", "x", "y", "theta", "e", "gamma", "omega")}

    for i in range(n):
        theta_d, omega_d, e = desired_heading(state[:2], velocity(state, v), c, R, k, direction, dt)
        if not use_feedforward:
            omega_d = 0.0
        omega_cmd, gamma = heading_control(state[2], theta_d, omega_d, k_gamma)
        omega = float(np.clip(omega_cmd, -omega_max, omega_max))

        log["t"][i] = i * dt
        log["x"][i], log["y"][i], log["theta"][i] = state
        log["e"][i], log["gamma"][i], log["omega"][i] = e, gamma, omega

        state = unicycle_step(state, v, omega, dt, omega_max)

    log["V"] = 0.5 * log["e"] ** 2
    return log
