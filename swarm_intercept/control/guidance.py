"""Guidance for the interceptor: where to fly to catch, and then circle, a moving target.

All functions return a desired VELOCITY vector u (direction and speed). The
unicycle then turns onto that direction with follow_velocity().
"""
import numpy as np
from swarm_intercept.utils.angles import wrap
from swarm_intercept.control.gvf import circle_gvf


def intercept_time(p, p_t, v_t, speed):
    """Earliest time at which a drone flying straight at 'speed' can meet the target.

    The target is assumed to keep its velocity v_t. We need the smallest t > 0 with
        | p_t + v_t * t - p | = speed * t
    which is a quadratic in t. Returns np.inf if the target cannot be caught.
    """
    r = np.asarray(p_t, dtype=float) - np.asarray(p, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    a = v_t @ v_t - speed ** 2
    b = 2.0 * (r @ v_t)
    c = r @ r
    if c < 1e-12:
        return 0.0
    if abs(a) < 1e-12:                       # same speed: the equation is linear
        return -c / b if b < 0 else np.inf
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.inf
    roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    roots = [t for t in roots if t > 0]
    return min(roots) if roots else np.inf


def pursuit_velocity(p, p_t, v_t, speed):
    """Lead pursuit: fly straight at the point where the target WILL be, not where it is.

    Returns (u, aim_point, t_go).
    """
    t_go = intercept_time(p, p_t, v_t, speed)
    if not np.isfinite(t_go):
        t_go = 0.0                           # cannot compute a lead: aim at the target itself
    aim = np.asarray(p_t, dtype=float) + np.asarray(v_t, dtype=float) * t_go
    d = aim - np.asarray(p, dtype=float)
    n = np.linalg.norm(d)
    u = speed * d / n if n > 1e-9 else np.zeros(2)
    return u, aim, t_go


def orbit_velocity(p, p_t, v_t, radius, k, orbit_speed, direction=1):
    """Circle a MOVING point: the Step 1.1 circle GVF centred on the target, plus the target's velocity.

        u = v_t + orbit_speed * chi / |chi|

    The first term makes the drone keep up with the target (feed-forward); the
    second makes it converge to, and travel round, the circle of the given radius.
    Returns (u, e) with e = distance error to the orbit circle.
    """
    chi, e, _, _ = circle_gvf(p, p_t, radius, k, direction)
    u = np.asarray(v_t, dtype=float) + orbit_speed * chi / np.linalg.norm(chi)
    return u, e


def follow_velocity(theta, u_now, u_next, dt, k_gamma, v_lo, v_hi):
    """Turn a desired velocity vector into unicycle commands (speed, turn rate).

    u_now   desired velocity at the current position
    u_next  desired velocity one step ahead (gives the feed-forward turn rate, as in Step 1.1)
    Returns (speed, omega_cmd, gamma).
    """
    theta_d = np.arctan2(u_now[1], u_now[0])
    omega_d = wrap(np.arctan2(u_next[1], u_next[0]) - theta_d) / dt
    gamma = wrap(theta_d - theta)
    omega = omega_d + k_gamma * gamma
    speed = float(np.clip(np.linalg.norm(u_now), v_lo, v_hi))
    return speed, omega, gamma
