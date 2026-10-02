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


# ---------------------------------------------------------------------------
# Three intercept laws, compared in Step 1.5
# ---------------------------------------------------------------------------
LAWS = ("pure", "lead", "pn")


def pure_pursuit_velocity(p, p_t, speed):
    """Pure pursuit: always fly straight at where the target is NOW."""
    d = np.asarray(p_t, dtype=float) - np.asarray(p, dtype=float)
    n = np.linalg.norm(d)
    return speed * d / n if n > 1e-9 else np.zeros(2)


def los_rate_and_closing_speed(p, vel, p_t, v_t):
    """Line-of-sight (LOS) geometry between drone and target.

    LOS = the straight line from the drone to the target.
    Returns (lam_dot, closing_speed)
        lam_dot        how fast that line is rotating [rad/s]
        closing_speed  how fast the range is shrinking [m/s] (negative = opening)
    """
    r = np.asarray(p_t, dtype=float) - np.asarray(p, dtype=float)
    v_rel = np.asarray(v_t, dtype=float) - np.asarray(vel, dtype=float)
    r2 = max(r @ r, 1e-9)
    lam_dot = (r[0] * v_rel[1] - r[1] * v_rel[0]) / r2
    closing = -(r @ v_rel) / np.sqrt(r2)
    return lam_dot, closing


def guidance_command(law, state, speed, p_t, v_t, dt, k_gamma, pn_gain=3.0):
    """Turn-rate command for the interceptor under the chosen law. Returns (speed, omega).

    pure  aim at the target's current position
    lead  aim at the predicted meeting point (assumes the target keeps its velocity)
    pn    proportional navigation: turn at N times the LOS rotation rate. If the LOS
          stops rotating, drone and target are on a collision course. While the drone
          is not yet closing on the target, lead pursuit is used to turn it round.
    """
    if law not in LAWS:
        raise ValueError(f"unknown guidance law '{law}', choose one of {LAWS}")
    p, theta = np.asarray(state[:2], dtype=float), state[2]
    vel = speed * np.array([np.cos(theta), np.sin(theta)])
    p_next = p + vel * dt
    p_t_next = np.asarray(p_t, dtype=float) + np.asarray(v_t, dtype=float) * dt

    if law == "pn":
        lam_dot, closing = los_rate_and_closing_speed(p, vel, p_t, v_t)
        if closing > 0.2 * speed:
            return speed, pn_gain * lam_dot
        law = "lead"                                   # not closing yet: turn toward the meeting point first

    if law == "pure":
        u, u_next = pure_pursuit_velocity(p, p_t, speed), pure_pursuit_velocity(p_next, p_t_next, speed)
    else:
        u, _, _ = pursuit_velocity(p, p_t, v_t, speed)
        u_next, _, _ = pursuit_velocity(p_next, p_t_next, v_t, speed)
    v, omega, _ = follow_velocity(theta, u, u_next, dt, k_gamma, speed, speed)
    return v, omega
