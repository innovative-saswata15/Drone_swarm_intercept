"""Heading controller: turn the drone onto the desired heading.

    gamma = wrap(theta_d - theta)          heading error
    omega = omega_d + k_gamma * gamma      feed-forward + proportional feedback

Lyapunov check: with V = 0.5*gamma^2, gamma_dot = omega_d - omega = -k_gamma*gamma,
so V_dot = -k_gamma*gamma^2 <= 0 (exponential convergence, time constant 1/k_gamma).
"""
from swarm_intercept.utils.angles import wrap


def heading_control(theta, theta_d, omega_d, k_gamma):
    """Return (omega_cmd, gamma). Saturation is applied by the drone model."""
    gamma = wrap(theta_d - theta)
    omega = omega_d + k_gamma * gamma
    return omega, gamma
