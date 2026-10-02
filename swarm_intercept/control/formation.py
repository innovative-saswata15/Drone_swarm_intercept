"""Formation keeping on a circle: equal angular spacing by speed modulation.

Every active drone follows the same circle (GVF, see gvf.py). Spacing is
controlled only through SPEED:

    v_i = v0 + k_c * (gap_ahead_i - gap_behind_i)

A drone speeds up when the gap to the drone in front is larger than the gap
to the drone behind, and slows down in the opposite case. The law never uses
the number of drones, so when one drone leaves, the others spread out
automatically (120 deg -> 180 deg), and close up again when it returns.

Lyapunov sketch: each gap obeys g_i_dot = (k_c/R) * (g_{i+1} - 2 g_i + g_{i-1}),
a diffusion equation on a ring. With V = 0.5 * sum((g_i - 2*pi/N)^2),
V_dot = -(k_c/R) * sum((g_{i+1} - g_i)^2) <= 0, so all gaps become equal.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def phase_angles(positions, c):
    """Polar angle of each drone about the centre c. positions has shape (N, 2)."""
    r = np.asarray(positions, dtype=float) - np.asarray(c, dtype=float)
    return np.arctan2(r[:, 1], r[:, 0])


def ring_gaps(phis, active, direction=1):
    """Gap ahead of and behind each active drone, measured along the travel direction.

    phis    polar angles of ALL drones, shape (N,)
    active  boolean mask, shape (N,): which drones are in the formation
    Returns (gap_ahead, gap_behind), each shape (N,), in radians, NaN for inactive drones.
    The gaps ahead of the active drones always sum to 2*pi.
    """
    phis = np.asarray(phis, dtype=float)
    active = np.asarray(active, dtype=bool)
    n = len(phis)
    gap_ahead = np.full(n, np.nan)
    gap_behind = np.full(n, np.nan)
    idx = np.where(active)[0]
    m = len(idx)
    if m == 0:
        return gap_ahead, gap_behind
    if m == 1:
        gap_ahead[idx[0]] = gap_behind[idx[0]] = TWO_PI
        return gap_ahead, gap_behind
    psi = np.mod(direction * phis[idx], TWO_PI)      # angle measured along the travel direction
    order = np.argsort(psi)                          # drones sorted around the ring
    psi_sorted = psi[order]
    ahead_sorted = np.mod(np.roll(psi_sorted, -1) - psi_sorted, TWO_PI)
    behind_sorted = np.roll(ahead_sorted, 1)
    gap_ahead[idx[order]] = ahead_sorted
    gap_behind[idx[order]] = behind_sorted
    return gap_ahead, gap_behind


def spacing_speeds(phis, active, v0, k_c, v_min, v_max, direction=1):
    """Speed command for every drone. Inactive drones get the cruise speed v0.

    Returns (speeds, gap_ahead).
    """
    gap_ahead, gap_behind = ring_gaps(phis, active, direction)
    speeds = np.full(len(phis), float(v0))
    act = np.asarray(active, dtype=bool)
    speeds[act] = v0 + k_c * (gap_ahead[act] - gap_behind[act])
    return np.clip(speeds, v_min, v_max), gap_ahead
