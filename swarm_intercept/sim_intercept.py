"""Guidance benchmark: ONE interceptor chasing ONE target, nothing else.

This isolates the intercept leg so the three guidance laws can be compared
fairly: every law gets exactly the same start, the same target path and the
same sensor quality. The interceptor only ever uses the Kalman estimate.
"""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step
from swarm_intercept.core.target import Target, PATTERNS
from swarm_intercept.core.sensors import target_measurement, area_measurement
from swarm_intercept.core.estimator import TargetKF
from swarm_intercept.control.guidance import guidance_command

# Two test conditions. 'room' is the real 4 m x 4 m arena; 'large' gives the guidance time to matter.
SCENARIOS = {
    "room":  {"half_size": 2.0, "target_speed_rel": 0.4, "min_range": 1.0, "max_range": 3.0},
    "large": {"half_size": 5.0, "target_speed_rel": 0.75, "min_range": 3.0, "max_range": 7.0},
}


def random_case(scenario, seed):
    """Random start geometry for one run: drone state, target start, heading and pattern."""
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    lim = sc["half_size"] - 0.3
    while True:
        drone = rng.uniform(-lim, lim, 2)
        tgt = rng.uniform(-lim, lim, 2)
        if sc["min_range"] <= np.linalg.norm(tgt - drone) <= sc["max_range"]:
            break
    return {"drone": np.array([drone[0], drone[1], rng.uniform(-np.pi, np.pi)]),
            "target": tgt, "heading": rng.uniform(-np.pi, np.pi),
            "pattern": PATTERNS[seed % len(PATTERNS)]}


def simulate_intercept(cfg, law, scenario="room", seed=0, t_max=None):
    """Fly one intercept. Returns a dict with the outcome and the logged paths."""
    sc = SCENARIOS[scenario]
    dt, ms, sn, ec = cfg["sim"]["dt"], cfg["mission"], cfg["sensor"], cfg["estimator"]
    speed = ms["intercept_speed"]
    omega_max, k_gamma = cfg["drone"]["omega_max"], cfg["formation"]["k_heading"]
    case = random_case(scenario, seed)
    t_max = 2.0 * (2 * sc["half_size"]) / (speed * (1 - sc["target_speed_rel"] / ms["intercept_speed_rel"])) \
        if t_max is None else t_max

    target = Target(case["pattern"], sc["target_speed_rel"] * cfg["drone"]["v"], sc["half_size"],
                    np.random.default_rng(1000 + seed), start=case["target"], heading=case["heading"])
    sens_rng = np.random.default_rng(2000 + seed)       # separate stream: the target path is identical for every law
    kf = TargetKF(sn["sigma"], ec["sigma_accel"])
    ar = sn["area"]
    every, every_area = max(1, int(round(1 / (sn["rate_hz"] * dt)))), max(1, int(round(1 / (ar["rate_hz"] * dt))))
    state = case["drone"].copy()

    n = int(round(t_max / dt))
    path, tpath = np.zeros((n, 2)), np.zeros((n, 2))
    effort, length, captured, k = 0.0, 0.0, False, 0
    for k in range(n):
        truth = target.state
        path[k], tpath[k] = state[:2], truth[:2]
        if np.linalg.norm(state[:2] - truth[:2]) <= ms["capture_radius"]:
            captured = True
            break
        kf.predict(dt)
        if k % every_area == 0:                          # the whole benchmark arena is 'protected zone'
            z = area_measurement(truth[:2], sc["half_size"], ar["sigma"], ar["p_dropout"], sens_rng)
            if z is not None:
                kf.update(z, sigma=ar["sigma"])
        if k % every == 0:
            z, _ = target_measurement(truth[:2], state[None, :2], sn["fov_radius"], sn["sigma"], sn["p_dropout"], sens_rng)
            if z is not None:
                kf.update(z)
        if kf.has_track:
            v, omega = guidance_command(law, state, speed, kf.position, kf.velocity, dt, k_gamma, ms["pn_gain"])
        else:
            v, omega = speed, 0.0
        omega = float(np.clip(omega, -omega_max, omega_max))
        effort += omega ** 2 * dt
        length += v * dt
        state = unicycle_step(state, v, omega, dt, omega_max)
        target.step(dt)

    return {"captured": captured, "time": k * dt if captured else np.nan, "length": length, "effort": effort,
            "pattern": case["pattern"], "path": path[:k + 1], "target_path": tpath[:k + 1]}
