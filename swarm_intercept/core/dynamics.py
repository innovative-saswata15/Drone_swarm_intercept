"""Drone model: the 'virtual unicycle'.

State = [x, y, theta]
    x, y   position in the world frame [m]
    theta  heading, counter-clockwise from +x [rad], kept in (-pi, pi]
Inputs
    v      forward speed [m/s]
    omega  turn rate [rad/s], limited to +/- omega_max
"""
import numpy as np
from swarm_intercept.utils.angles import wrap


def unicycle_step(state, v, omega, dt, omega_max=np.inf):
    """Advance the unicycle by one time step. Returns a new state array.

    Uses the midpoint rule: the position moves along the heading at the MIDDLE
    of the step (theta + omega*dt/2). Plain Euler (using theta at the start)
    always steps along the tangent, so on a circle it drifts slowly outward;
    the midpoint rule removes that drift at no extra cost.
    """
    x, y, theta = state
    omega = float(np.clip(omega, -omega_max, omega_max))   # physical turn-rate limit
    theta_mid = theta + 0.5 * omega * dt
    x_new = x + v * np.cos(theta_mid) * dt
    y_new = y + v * np.sin(theta_mid) * dt
    theta_new = wrap(theta + omega * dt)
    return np.array([x_new, y_new, theta_new])


def velocity(state, v):
    """World-frame velocity vector of the unicycle."""
    return v * np.array([np.cos(state[2]), np.sin(state[2])])
