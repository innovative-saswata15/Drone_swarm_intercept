"""Score a policy on the benchmark cases, with exactly the same metrics as the classical baseline."""
import numpy as np
from swarm_intercept.benchmark import case_metrics, METRICS, QUALITIES
from swarm_intercept.core.target import PATTERNS
from swarm_intercept.rl.env import MissionEnv


def fly_case(cfg, pattern, quality, seed, policy=None, rl_cfg=None):
    """Fly one benchmark case. policy(obs) -> action in [-1,1]^2; None = zero residual (classical).

    Returns (case_cfg, log, fsm): the full 50 Hz log and the state machine (with its event list).
    """
    env = MissionEnv(cfg, rl_cfg, mode="benchmark", record=True)
    obs, _ = env.reset(options={"case": (pattern, quality, seed)})
    over = env._over
    while not over:
        action = np.zeros(2) if policy is None else policy(obs)
        obs, _, term, trunc, _ = env.step(action)
        over = term or trunc
    return env.case_cfg, env.sim.log, env.sim.fsm


def run_case_policy(cfg, pattern, quality, seed, policy=None, rl_cfg=None):
    """Fly one benchmark case and measure it. Returns one table row (same columns as baselines/classical.csv)."""
    c, lg, fsm = fly_case(cfg, pattern, quality, seed, policy, rl_cfg)
    return case_metrics(c, lg, fsm, pattern, quality, seed)


def evaluate_policy(cfg, policy=None, n_per_cell=25, patterns=PATTERNS, qualities=tuple(QUALITIES), rl_cfg=None):
    """Run every (quality, pattern, seed) benchmark case. Returns a list of rows."""
    return [run_case_policy(cfg, p, q, s, policy, rl_cfg) for q in qualities for p in patterns for s in range(n_per_cell)]