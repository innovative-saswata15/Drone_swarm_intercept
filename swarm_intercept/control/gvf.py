"""Guidance vector field (GVF) for following a circle.

Path:      circle with centre c and radius R
Error:     e = |p - c| - R        (positive outside, negative inside)
Field:     chi = tau_hat - k * e * n_hat
           tau_hat = tangent (direction of travel), n_hat = outward normal
On the circle (e = 0) the field is purely tangential; off the circle the
second term pulls the drone back toward it.
"""
import numpy as np
from swarm_intercept.utils.angles import wrap

_EPS = 1e-2   # closer than 1 cm to the centre: the normal is undefined


def circle_gvf(p, c, R, k, direction=1):
    """Evaluate the circle GVF at position p.

    p, c       position and circle centre, arrays of shape (2,)
    R          circle radius [m]
    k          convergence gain [1/m]; approach angle = atan(k*e)
    direction  +1 counter-clockwise, -1 clockwise

    Returns (chi, e, n_hat, tau_hat).
    """
    r = np.asarray(p, dtype=float) - np.asarray(c, dtype=float)
    dist = np.linalg.norm(r)
    if dist < _EPS:                       # at the centre: pick a fixed outward direction
        n_hat = np.array([1.0, 0.0])
        dist = 0.0
    else:
        n_hat = r / dist
    # rotate n_hat by +90 deg (CCW) or -90 deg (CW) to get the tangent
    tau_hat = direction * np.array([-n_hat[1], n_hat[0]])
    e = dist - R
    chi = tau_hat - k * e * n_hat
    return chi, e, n_hat, tau_hat


def desired_heading(p, vel, c, R, k, direction=1, dt=0.02):
    """Desired heading theta_d and its rate omega_d at position p.

    vel is the drone's current velocity vector. omega_d is found by a
    'look-ahead' difference: evaluate the field where the drone will be one
    step later and see how much the desired heading turns.

    Returns (theta_d, omega_d, e).
    """
    chi, e, _, _ = circle_gvf(p, c, R, k, direction)
    theta_d = np.arctan2(chi[1], chi[0])
    p_next = np.asarray(p, dtype=float) + np.asarray(vel, dtype=float) * dt
    chi_next, _, _, _ = circle_gvf(p_next, c, R, k, direction)
    theta_d_next = np.arctan2(chi_next[1], chi_next[0])
    omega_d = wrap(theta_d_next - theta_d) / dt
    return theta_d, omega_d, e
