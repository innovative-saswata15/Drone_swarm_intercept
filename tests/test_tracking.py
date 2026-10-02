"""Tests for Step 1.3. Run with:  pytest -q"""
from pathlib import Path
import numpy as np
from swarm_intercept.config import load_config
from swarm_intercept.core.target import Target, PATTERNS
from swarm_intercept.core.sensors import target_measurement
from swarm_intercept.core.estimator import TargetKF
from swarm_intercept.sim_track import simulate_tracking

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers
DT = cfg["sim"]["dt"]


def test_target_stays_in_arena_and_respects_speed():
    for p in PATTERNS:
        tg = Target(p, 0.16, 2.0, np.random.default_rng(1), start=(-1.7, -1.0), heading=0.4)
        for _ in range(15000):                       # 300 s
            s = tg.step(DT)
            assert np.all(np.abs(s[:2]) <= 2.0)
            assert np.hypot(s[2], s[3]) <= 0.16 + 1e-9


def test_sensor_sees_only_inside_fov():
    rng = np.random.default_rng(0)
    drones = np.array([[0.0, 0.0], [3.0, 0.0]])
    z, seen = target_measurement([0.5, 0.0], drones, 0.6, 0.03, 0.0, rng)
    assert z is not None and list(seen) == [0]
    z, seen = target_measurement([1.5, 0.0], drones, 0.6, 0.03, 0.0, rng)
    assert z is None and len(seen) == 0


def test_sensor_noise_and_dropout_statistics():
    rng = np.random.default_rng(0)
    drones = np.array([[0.0, 0.0]])
    zs = [target_measurement([0.1, 0.2], drones, 0.6, 0.03, 0.1, rng)[0] for _ in range(20000)]
    got = np.array([z for z in zs if z is not None])
    assert abs(1 - len(got) / len(zs) - 0.1) < 0.01                  # ~10 % lost
    assert np.allclose(got.mean(axis=0), [0.1, 0.2], atol=0.002)     # unbiased
    assert np.allclose(got.std(axis=0), 0.03, atol=0.002)            # right noise level


def test_kf_learns_velocity_it_never_measures():
    rng = np.random.default_rng(0)
    kf = TargetKF(0.03, 0.2)
    pos, vel = np.array([0.0, 0.0]), np.array([0.12, -0.05])
    for i in range(1500):                           # 30 s
        pos = pos + vel * DT
        kf.predict(DT)
        if i % 5 == 0:
            kf.update(pos + rng.normal(0, 0.03, 2))
    assert np.linalg.norm(kf.position - pos) < 3 * kf.position_sigma() * np.sqrt(2)   # within its own 3-sigma
    assert np.linalg.norm(kf.position - pos) < 0.06
    assert np.linalg.norm(kf.velocity - vel) < 0.05


def test_kf_uncertainty_grows_without_measurements_and_shrinks_with():
    kf = TargetKF(0.03, 0.2)
    kf.update([0.0, 0.0]); kf.predict(DT); kf.update([0.0, 0.0])
    s0 = kf.position_sigma()
    for _ in range(250):
        kf.predict(DT)
    s1 = kf.position_sigma()
    kf.update([0.0, 0.0])
    assert s1 > 5 * s0 and kf.position_sigma() < s1


def test_kf_is_honest_about_its_uncertainty():
    """On a constant-velocity target the squared error, scaled by P, should average ~4 (its 4 states)."""
    nees = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        kf = TargetKF(0.03, 0.05)
        pos, vel = np.array([0.0, 0.0]), np.array([0.1, 0.1])
        for i in range(1000):
            pos = pos + vel * DT
            kf.predict(DT)
            if i % 5 == 0:
                kf.update(pos + rng.normal(0, 0.03, 2))
        err = kf.x - np.concatenate([pos, vel])
        nees.append(err @ np.linalg.inv(kf.P) @ err)
    assert np.mean(nees) < 6.0          # not over-confident


def test_tracking_run_all_patterns():
    for p in PATTERNS:
        lg = simulate_tracking(cfg, p, T=120.0)
        tr, iv = lg["tracked"], lg["in_view"]
        assert tr.any()                                                  # the target is found
        assert np.sqrt(np.mean(lg["pos_err"][tr & iv] ** 2)) < 0.10      # accurate while in view
        assert np.all(lg["pos_err"][tr] < 3 * 3 * lg["sigma"][tr] + 0.6) # never wildly outside its own bound


def test_same_seed_gives_same_run():
    a = simulate_tracking(cfg, "weave", seed=3, T=30.0)
    b = simulate_tracking(cfg, "weave", seed=3, T=30.0)
    assert np.array_equal(a["target"], b["target"]) and np.array_equal(a["meas"], b["meas"], equal_nan=True)
