"""Gymnasium environment for the residual agent.

The classical controller (formation, guidance, state machine) runs exactly as before. The agent
only adds a small, bounded correction to the (speed, turn-rate) command of the drone that is on
the mission, and the safety filter still has the last word:

    action a in [-1, 1]^2  ->  residual = (a0 * max_dv, a1 * max_domega)  ->  nominal + residual
                           ->  CBF safety filter  ->  unicycle

a = 0 therefore reproduces the classical controller exactly (checked in tests/test_rl_env.py).

One env step = `action_repeat` control periods with the same residual. Stretches in which no drone
is on the mission (waiting for a target, before the decision, after rejoining) are flown with a = 0
and skipped, so every step the agent sees is a real decision.

Modes
    "train"      drones start in formation, the target enters almost at once, random pattern and
                 sensing quality, case seeds >= 1000  (never overlaps the benchmark seeds 0..24)
    "benchmark"  the exact case of benchmark.make_case(): random starts, 30 s of surveillance,
                 90 s long, runs to the end so that the benchmark metrics can be computed
"""
from pathlib import Path
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from swarm_intercept.config import load_config
from swarm_intercept.core.target import PATTERNS
from swarm_intercept.benchmark import make_case, QUALITIES
from swarm_intercept.sim_mission import MissionSim
from swarm_intercept.mission.fsm import SURVEIL, INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.rl.observation import build_observation, OBS_DIM
from swarm_intercept.rl.reward import step_reward

RL_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "rl.yaml"
_MODE = {"intercept": INTERCEPT, "investigate": INVESTIGATE, "return": RETURN}


def load_rl_config(path=None):
    with open(RL_CONFIG if path is None else path, "r") as f:
        return yaml.safe_load(f)["rl"]


def formation_start(cfg, rng):
    """Drones equally spaced on the formation circle, flying along it (a settled formation)."""
    f, c = cfg["formation"], np.array(cfg["arena"]["center"], dtype=float)
    n, phi0 = f["n_drones"], rng.uniform(0, 2 * np.pi)
    phi = phi0 + 2 * np.pi * np.arange(n) / n
    pos = c + f["radius"] * np.column_stack([np.cos(phi), np.sin(phi)])
    return np.column_stack([pos, phi + f["direction"] * np.pi / 2])


class MissionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, rl_cfg=None, mode="train", record=False, terminate_on_done=None):
        self.cfg = load_config() if cfg is None else cfg
        self.rl = load_rl_config() if rl_cfg is None else rl_cfg
        self.mode = mode
        self.record = record
        self.terminate_on_done = (mode == "train") if terminate_on_done is None else terminate_on_done
        self.active_modes = {_MODE[m] for m in self.rl["active_modes"]}
        self.repeat = int(self.rl["action_repeat"])
        v = self.cfg["drone"]["v"]
        self.max_dv = self.rl["max_dv_rel"] * v
        self.max_dw = self.rl["max_domega"]
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.sim = None
        self.case_cfg = None
        self.case = None
        self._over = True

    # ------------------------------------------------------------------ helpers
    def _drone(self):
        """Index of the drone the agent controls right now, or None."""
        i = self.sim.fsm.interceptor
        return i if (i is not None and self.sim.fsm.state[i] in self.active_modes) else None

    def _completed(self):
        return bool(self.sim.fsm.done and self.sim.fsm.interceptor is None)

    def _timeout(self):
        return self.sim.i >= self.n_max

    def _fast_forward(self):
        """Fly with zero residual until the agent has a decision to make, the mission is over or time is up."""
        sim = self.sim
        while self._drone() is None and not self._timeout():
            if self._completed() and self.terminate_on_done:
                break
            sim.apply(None)
            if self._timeout():
                break
            sim.begin()

    def _obs(self):
        j = self._drone()
        if j is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return build_observation(self.sim, j)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "case" in options:
            pattern, quality, case_seed = options["case"]
        else:
            pattern = PATTERNS[int(self.np_random.integers(len(PATTERNS)))]
            quality = list(QUALITIES)[int(self.np_random.integers(len(QUALITIES)))]
            case_seed = int(self.np_random.integers(1000, 10 ** 9))
        c, S = make_case(self.cfg, pattern, quality, case_seed)
        ts = c["sim"]["time_scale"]
        if self.mode == "train":
            S = formation_start(c, self.np_random)
            T, t_enter = self.rl["episode_time_rel"] * ts, self.rl["train_enter_time_rel"] * ts
        else:
            T, t_enter = 90.0 * ts, None
        self.case_cfg, self.case = c, (pattern, quality, case_seed)
        self.n_max = int(round(T / c["sim"]["dt"]))
        self.sim = MissionSim(c, pattern, seed=case_seed, T=T, states0=S, record=self.record, t_enter=t_enter)
        self.sim.begin()
        self._fast_forward()
        self._over = self._drone() is None
        return self._obs(), {"case": self.case, "interceptor": self._drone()}

    def step(self, action):
        sim = self.sim
        if self._over:
            return self._obs(), 0.0, self._completed(), not self._completed(), {"completed": self._completed()}
        j = self._drone()
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        res = np.zeros((sim.N, 2))
        res[j] = a[0] * self.max_dv, a[1] * self.max_dw

        reward, n_filtered, steps = 0.0, 0, 0
        for _ in range(self.repeat):
            mode_pre = int(sim.mode[j])
            d_pre = float(sim.d_true[j])
            e_pre = float(sim.nominal[j, 2])
            sim.apply(res)
            steps += 1
            n_filtered += int(sim.filtered[j])
            if self._timeout():
                reward += step_reward(sim, j, mode_pre, d_pre, e_pre, self.rl["reward"], False)
                break
            sim.begin()
            done_now = self._completed()
            reward += step_reward(sim, j, mode_pre, d_pre, e_pre, self.rl["reward"], done_now)
            if done_now or self._drone() != j:
                break

        self._fast_forward()
        completed = self._completed()
        terminated = completed and self.terminate_on_done
        truncated = (not terminated) and self._timeout()
        self._over = terminated or truncated or (self._drone() is None)
        if self._over and not (terminated or truncated):
            truncated = True                      # nothing left to decide: end the episode
        info = {"completed": completed, "steps": steps, "filter_share": n_filtered / max(steps, 1),
                "time": sim.t, "interceptor": self._drone()}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


def eval_cases(n=16, seed0=5000):
    """A FIXED list of (pattern, quality, seed) cases for evaluating during training.

    Seeds start at 5000: they overlap neither the benchmark (0..24) nor the random training cases drawn
    from the env's own generator in practice. Patterns and qualities are cycled so all combinations appear.
    """
    qs = list(QUALITIES)
    return [(PATTERNS[i % len(PATTERNS)], qs[(i // len(PATTERNS)) % len(qs)], seed0 + i) for i in range(n)]


class FixedCaseEnv(MissionEnv):
    """Training-style env that cycles through a fixed list of cases (same start, target, noise every time).

    Used for evaluation during training: n consecutive episodes always cover every case once, and the same
    case always produces the same result for the same policy. So evaluation scores at different times
    (and the classical zero-action score) can be compared directly, with no luck-of-the-draw noise.
    """

    def __init__(self, cases=None, **kw):
        super().__init__(mode="train", **kw)
        self.cases = cases if cases is not None else eval_cases()
        self._k = 0

    def reset(self, seed=None, options=None):
        case = self.cases[self._k % len(self.cases)]
        self._k += 1
        return super().reset(seed=case[2], options={"case": case})