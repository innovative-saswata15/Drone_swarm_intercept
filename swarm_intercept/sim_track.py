"""Formation flight + moving target + virtual sensor + Kalman filter."""
import numpy as np
from swarm_intercept.sim_multi import formation_tick
from swarm_intercept.core.target import Target
from swarm_intercept.core.sensors import target_measurement
from swarm_intercept.core.estimator import TargetKF

DEFAULT_STARTS = np.array([[1.9, 0.0, 2.0], [1.5, 1.0, -1.0], [0.3, -0.2, 0.5]])


def make_target(cfg, pattern, rng):
    tc = cfg["target"]
    return Target(pattern, tc["speed_rel"] * cfg["drone"]["v"], cfg["arena"]["half_size"], rng,
                  start=tc["start"], heading=np.radians(tc["heading_deg"]), time_scale=cfg["sim"]["time_scale"])


def simulate_tracking(cfg, pattern="straight", seed=None, T=120.0, states0=None):
    """Three drones circle in formation while a target crosses the arena.

    Returns a dict of logged arrays. Estimate entries are NaN while there is no track.
    """
    dt = cfg["sim"]["dt"]
    sc, ec = cfg["sensor"], cfg["estimator"]
    ts = cfg["sim"]["time_scale"]
    rng = np.random.default_rng(cfg["sim"]["seed"] if seed is None else seed)
    states = np.array(DEFAULT_STARTS if states0 is None else states0, dtype=float)
    N = len(states)
    active = np.ones(N, dtype=bool)
    target = make_target(cfg, pattern, rng)
    kf = TargetKF(sc["sigma"], ec["sigma_accel"])
    every = max(1, int(round(1.0 / (sc["rate_hz"] * dt))))     # sensor runs slower than the control loop
    lost_after = ec["lost_after"] * ts

    n = int(round(T / dt))
    log = {"t": np.arange(n) * dt,
           "x": np.zeros((n, N)), "y": np.zeros((n, N)), "theta": np.zeros((n, N)),
           "target": np.zeros((n, 4)), "est": np.full((n, 4), np.nan),
           "meas": np.full((n, 2), np.nan), "sigma": np.full(n, np.nan),
           "P": np.full((n, 4, 4), np.nan),
           "in_view": np.zeros(n, dtype=bool), "tracked": np.zeros(n, dtype=bool)}

    for i in range(n):
        log["x"][i], log["y"][i], log["theta"][i] = states[:, 0], states[:, 1], states[:, 2]
        truth = target.state
        log["target"][i] = truth

        kf.predict(dt)
        d = np.linalg.norm(states[:, :2] - truth[:2], axis=1)
        log["in_view"][i] = np.any(d <= sc["fov_radius"])
        if i % every == 0:
            z, _ = target_measurement(truth[:2], states[:, :2], sc["fov_radius"], sc["sigma"], sc["p_dropout"], rng)
            if z is not None:
                kf.update(z)
                log["meas"][i] = z
        if kf.has_track and kf.time_since_update > lost_after:
            kf.drop()
        if kf.has_track:
            log["est"][i], log["P"][i] = kf.x, kf.P
            log["sigma"][i], log["tracked"][i] = kf.position_sigma(), True

        formation_tick(states, active, cfg)
        target.step(dt)

    log["pos_err"] = np.linalg.norm(log["est"][:, :2] - log["target"][:, :2], axis=1)
    log["vel_err"] = np.linalg.norm(log["est"][:, 2:] - log["target"][:, 2:], axis=1)
    return log
