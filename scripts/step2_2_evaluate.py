"""Step 2.2 - residual SAC against the classical controller on the SAME benchmark cases.

Run:  python scripts/step2_2_evaluate.py results/step2_1/best_model.zip        (trained agent)
      python scripts/step2_2_evaluate.py results/step2_1/best_model.zip 50      (50 seeds per cell)
      python scripts/step2_2_evaluate.py --classical 3                          (zero residual; sanity check, no torch needed)
Reads  baselines/classical.csv (the classical numbers from Step 1.7)
Writes results/step2_2/residual_sac.csv   (one row per mission, same columns as the baseline)
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from swarm_intercept.config import load_config
from swarm_intercept.benchmark import METRICS, QUALITIES
from swarm_intercept.core.target import PATTERNS
from swarm_intercept.rl.evaluate import run_case_policy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "step2_2"
BASE = ROOT / "baselines" / "classical.csv"

# metric -> +1 higher is better, -1 lower is better
BETTER = {"completed": +1, "t_reach": -1, "t_mission": -1, "t_reform": -1, "in_view": +1,
          "track_err": -1, "effort": -1, "min_sep": +1, "filter_active": -1, "orbit_dist": -1}

_POLICY = None


def _load(model_path):
    global _POLICY
    if model_path is None:
        _POLICY = None
        return
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device="cpu")
    _POLICY = lambda obs: model.predict(obs, deterministic=True)[0]


def _run(job):
    cfg, pattern, quality, seed = job
    return run_case_policy(cfg, pattern, quality, seed, _POLICY)


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: (v if k in ("pattern", "quality") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    args = sys.argv[1:]
    classical = "--classical" in args
    args = [a for a in args if a != "--classical"]
    model_path = None if classical else args[0]
    n = int(args[0 if classical else 1]) if len(args) > (0 if classical else 1) else 25
    if not classical and model_path is None:
        sys.exit(__doc__)

    cfg = load_config()
    jobs = [(cfg, p, q, s) for q in QUALITIES for p in PATTERNS for s in range(n)]
    workers = max(1, (__import__("os").cpu_count() or 2) - 1)
    print(f"{len(jobs)} missions, {workers} worker(s) ...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_load, initargs=(model_path,)) as pool:
        rows = list(pool.map(_run, jobs, chunksize=2))

    OUT.mkdir(parents=True, exist_ok=True)
    name = "classical_check.csv" if classical else "residual_sac.csv"
    cols = ["pattern", "quality", "seed"] + list(METRICS)
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c in ("pattern", "quality", "seed") else f"{r[c]:.4f}" for c in cols])
    print(f"wrote {OUT / name}")

    base = {(r["pattern"], r["quality"], int(r["seed"])): r for r in read_csv(BASE)}
    pairs = [(base[(r["pattern"], r["quality"], int(r["seed"]))], r) for r in rows
             if (r["pattern"], r["quality"], int(r["seed"])) in base]

    print(f"\n{'metric':>14} | {'classical':>10} | {'residual':>10} | {'change':>8} | {'better in':>9}")
    for m, sign in BETTER.items():
        a = np.array([p[0][m] for p in pairs], dtype=float)
        b = np.array([p[1][m] for p in pairs], dtype=float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        if not ok.any():
            continue
        ma, mb = a[ok].mean(), b[ok].mean()
        better = np.mean(sign * (b[ok] - a[ok]) > 1e-3)
        print(f"{m:>14} | {ma:>10.3f} | {mb:>10.3f} | {100 * (mb - ma) / abs(ma) if ma else 0:>+7.1f}% | {100 * better:>8.0f}%")

    comp = np.mean([r["completed"] for r in rows])
    broke = np.mean([r["broke_dsafe"] for r in rows])
    print(f"\ncompleted {100 * comp:.1f}%   missions that broke d_safe {100 * broke:.1f}%")
    print("VERDICT:", "safe and complete, compare the table for the gains" if comp >= 0.999 and broke == 0
          else "NOT acceptable: completion fell or d_safe was broken")


if __name__ == "__main__":
    main()
