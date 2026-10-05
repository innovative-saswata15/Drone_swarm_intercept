"""Tests of the residual-RL environment. The most important one: action 0 == the classical controller."""
from pathlib import Path
import csv
import numpy as np
import pytest

from swarm_intercept.config import load_config
from swarm_intercept.rl.env import MissionEnv, load_rl_config
from swarm_intercept.rl.evaluate import run_case_policy
from swarm_intercept.rl.observation import OBS_DIM

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config(Path(__file__).parent / "reference_config.yaml")    # frozen numbers
with open(ROOT / "baselines" / "classical.csv", newline="") as _f:
    BASE = {(r["pattern"], r["quality"], int(r["seed"])): r for r in csv.DictReader(_f)}
COLS = [c for c in next(iter(BASE.values())) if c not in ("pattern", "quality", "seed")]


@pytest.mark.parametrize("pattern,quality,seed", [("weave", "clean", 3), ("stop_go", "degraded", 11), ("straight", "clean", 0)])
def test_zero_action_reproduces_classical_baseline(pattern, quality, seed):
    """With a = 0 the env must give the committed classical numbers, row by row."""
    row = run_case_policy(cfg, pattern, quality, seed, policy=None)
    ref = BASE[(pattern, quality, seed)]
    for k in COLS:
        assert np.isclose(row[k], float(ref[k]), atol=1e-3, equal_nan=True), f"{k}: {row[k]} vs {ref[k]}"


def test_reset_gives_a_decision_and_valid_observation():
    env = MissionEnv(cfg, mode="train")
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs)) and np.abs(obs).max() <= 5.0
    assert info["interceptor"] is not None
    assert env.observation_space.contains(obs)


def test_train_cases_never_use_benchmark_seeds():
    env = MissionEnv(cfg, mode="train")
    for s in range(20):
        _, info = env.reset(seed=s)
        assert info["case"][2] >= 1000


def test_observation_does_not_use_the_true_target():
    """Moving the true target (without changing the Kalman estimate) must not change the observation."""
    env = MissionEnv(cfg, mode="train")
    obs, _ = env.reset(seed=2)
    j = env._drone()
    from swarm_intercept.rl.observation import build_observation
    before = build_observation(env.sim, j)
    env.sim.target.pos = env.sim.target.pos + np.array([0.7, -0.9])
    env.sim.d_true = env.sim.d_true + 1.0
    assert np.array_equal(before, build_observation(env.sim, j))


def test_actions_are_bounded_and_safety_filter_still_holds():
    """Worst case for safety: random full-size actions. Drones must still keep d_safe."""
    env = MissionEnv(cfg, mode="train", record=True)
    rng = np.random.default_rng(0)
    d_safe = cfg["safety"]["d_safe"]
    for ep in range(3):
        env.reset(seed=ep)
        over = env._over
        while not over:
            a = rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.5, 3.0)       # also out-of-range: must be clipped
            _, r, term, trunc, _ = env.step(a)
            assert np.isfinite(r)
            over = term or trunc
        n = env.sim.i
        assert env.sim.log["min_sep_xy"][:n].min() >= d_safe - 1e-3
        lim = cfg["arena"]["half_size"]
        assert max(np.abs(env.sim.log["x"][:n]).max(), np.abs(env.sim.log["y"][:n]).max()) < lim


def test_episode_terminates_when_mission_is_complete():
    env = MissionEnv(cfg, mode="train")
    env.reset(seed=0)
    for _ in range(2000):
        _, _, term, trunc, info = env.step(np.zeros(2))
        if term or trunc:
            break
    assert term and info["completed"]


def test_rl_config_loads():
    rl = load_rl_config()
    assert rl["action_repeat"] >= 1 and 0 < rl["max_dv_rel"] <= 1


def test_fixed_case_env_is_repeatable_and_covers_all_cases():
    from swarm_intercept.rl.env import FixedCaseEnv, eval_cases
    cases = eval_cases(4)
    env = FixedCaseEnv(cases=cases, cfg=cfg)

    def episode():
        _, info = env.reset()
        total, over = 0.0, env._over
        while not over:
            _, r, term, trunc, _ = env.step(np.zeros(2))
            total += r
            over = term or trunc
        return info["case"], total

    first = [episode() for _ in range(4)]
    second = [episode() for _ in range(4)]
    assert [c for c, _ in first] == cases                       # every case once, in order
    assert [c for c, _ in second] == cases                      # and again on the next round
    assert np.allclose([t for _, t in first], [t for _, t in second])   # identical results: no luck-of-the-draw