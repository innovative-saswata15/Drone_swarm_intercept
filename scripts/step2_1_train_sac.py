"""Step 2.1 - train the residual Soft Actor-Critic agent.

The agent adds a small correction to the classical command of the drone on the mission
(intercept -> investigate -> return); the safety filter stays on top. See swarm_intercept/rl/env.py.

Run:  python scripts/step2_1_train_sac.py                    (300k decisions, all cores but one)
      python scripts/step2_1_train_sac.py --steps 1000000 --envs 8
Needs: pip install -e ".[rl]"
Writes results/step2_1/sac_residual.zip (final), best_model.zip (best on held-out training-style episodes),
       tensorboard logs in results/step2_1/tb
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "step2_1"


def make_env(rank, seed):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from swarm_intercept.rl.env import MissionEnv
        env = Monitor(MissionEnv(mode="train"))
        env.reset(seed=seed + rank)
        return env
    return _init


N_EVAL = 16          # fixed evaluation missions, the same ones every time


def make_eval_env():
    from stable_baselines3.common.monitor import Monitor
    from swarm_intercept.rl.env import FixedCaseEnv, eval_cases
    return Monitor(FixedCaseEnv(cases=eval_cases(N_EVAL)))


def classical_reference():
    """Score of the classical controller (zero residual) on the fixed evaluation missions."""
    import numpy as np
    from swarm_intercept.rl.env import FixedCaseEnv, eval_cases
    env = FixedCaseEnv(cases=eval_cases(N_EVAL))
    scores = []
    for _ in range(N_EVAL):
        env.reset()
        total, over = 0.0, env._over
        while not over:
            _, r, term, trunc, _ = env.step(np.zeros(2))
            total += r
            over = term or trunc
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="total agent decisions (env steps)")
    ap.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=20_000)
    ap.add_argument("--gradient_steps", type=int, default=1,
                    help="network updates per round of env steps; raise it (2-4) with many --envs to learn more per step, at lower speed")
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    OUT.mkdir(parents=True, exist_ok=True)
    VecEnv = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    train_env = VecEnv([make_env(i, args.seed) for i in range(args.envs)])
    eval_env = DummyVecEnv([make_eval_env])                    # the SAME fixed missions at every evaluation
    mean0, std0 = classical_reference()
    print(f"\nCLASSICAL reference (zero residual) on the {N_EVAL} fixed evaluation missions: "
          f"{mean0:.2f} +/- {std0:.2f}\n  -> an Eval line clearly above this means the agent is improving on the classical controller.\n")

    model = SAC(
        "MlpPolicy", train_env, seed=args.seed, verbose=1,
        learning_rate=3e-4, gamma=0.995, tau=0.005, batch_size=256,
        buffer_size=500_000, learning_starts=5_000,
        train_freq=1, gradient_steps=args.gradient_steps,
        ent_coef="auto_0.1",
        # log_std_init=-3: start with small random actions, i.e. close to the classical controller.
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3.0),
        tensorboard_log=str(OUT / "tb"),
    )
    callback = EvalCallback(eval_env, best_model_save_path=str(OUT), log_path=str(OUT), n_eval_episodes=N_EVAL,
                            eval_freq=max(1, args.eval_every // args.envs), deterministic=True)
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
    model.save(OUT / "sac_residual")
    print(f"saved {OUT / 'sac_residual.zip'}")


if __name__ == "__main__":
    main()