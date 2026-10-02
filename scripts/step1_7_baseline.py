"""Step 1.7 - Monte-Carlo baseline of the classical mission.

Flies many randomised missions (4 target patterns x 2 sensing qualities x N seeds) and
records the numbers that the residual-RL controller will later have to beat on the
SAME cases.

Run:  python scripts/step1_7_baseline.py        (25 seeds per cell = 200 missions)
      python scripts/step1_7_baseline.py 50     (50 seeds per cell)
Writes  baselines/classical.csv   (one row per mission; keep this file)
        results/step1_7/*.png     (plots)
"""
import sys
import csv
import time
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_intercept.config import load_config
from swarm_intercept.benchmark import run_benchmark, summarise, METRICS, QUALITIES
from swarm_intercept.core.target import PATTERNS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "step1_7"
BASE = ROOT / "baselines"

LABEL = {"completed": "missions completed", "t_detect": "time to detect [s]", "t_reach": "time to reach target [s]",
         "t_mission": "mission time [s]", "t_reform": "time to re-form 120 deg [s]",
         "orbit_dist": "distance while investigating [m]", "in_view": "target in view while investigating",
         "track_err": "tracking error while investigating [m]", "min_sep": "closest approach, mean [m]",
         "worst_sep": "closest approach, worst [m]", "broke_dsafe": "missions that broke d_safe",
         "filter_active": "share of time the filter acts", "effort": "interceptor turning effort",
         "farthest": "farthest from centre, mean [m]"}
PERCENT = {"completed", "in_view", "broke_dsafe", "filter_active"}
ORDER = ("completed", "broke_dsafe", "worst_sep", "min_sep", "t_detect", "t_reach", "t_mission", "t_reform",
         "orbit_dist", "in_view", "track_err", "effort", "filter_active", "farthest")


def fmt(metric, value):
    if np.isnan(value):
        return "-"
    return f"{100 * value:.0f}%" if metric in PERCENT else f"{value:.2f}"


def print_table(title, summary):
    keys = list(summary)
    print(f"\n{title}")
    print(f"{'':>40} | " + " | ".join(f"{k:>12}" for k in keys))
    print(f"{'number of missions':>40} | " + " | ".join(f"{summary[k]['n']:>12}" for k in keys))
    for m in ORDER:
        print(f"{LABEL[m]:>40} | " + " | ".join(f"{fmt(m, summary[k][m]):>12}" for k in keys))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    total = N * len(PATTERNS) * len(QUALITIES)
    print(f"flying {total} missions ({len(PATTERNS)} patterns x {len(QUALITIES)} sensing qualities x {N} seeds) ...")
    t0 = time.time()
    rows = run_benchmark(cfg, n_per_cell=N)
    print(f"done in {time.time() - t0:.0f} s")

    with open(BASE / "classical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "quality", "seed"] + list(METRICS))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print_table("ALL MISSIONS", summarise([dict(r, all="all") for r in rows], "all"))
    print_table("BY SENSING QUALITY", summarise(rows, "quality"))
    print_table("BY TARGET PATTERN", summarise(rows, "pattern"))

    col = lambda key, sel=None: np.array([r[key] for r in rows if sel is None or sel(r)], dtype=float)
    d_safe = cfg["safety"]["d_safe"]

    # 1) time to reach and mission time, by pattern -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key in zip(axes, ("t_reach", "t_mission")):
        data = [col(key, lambda r, p=p: r["pattern"] == p) for p in PATTERNS]
        data = [d[~np.isnan(d)] for d in data]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(PATTERNS) + 1)); ax.set_xticklabels(PATTERNS)
        ax.set_ylabel(LABEL[key]); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, None)
    fig.suptitle(f"Classical baseline over {len(rows)} missions (box = middle half, line = median, triangle = mean)", fontsize=11)
    fig.savefig(OUT / "1_times_by_pattern.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 2) clean versus degraded sensing --------------------------------------
    keys = ("t_detect", "t_reach", "orbit_dist", "track_err")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.8))
    for ax, key in zip(axes, keys):
        means, errs = [], []
        for q in QUALITIES:
            d = col(key, lambda r, q=q: r["quality"] == q); d = d[~np.isnan(d)]
            means.append(d.mean()); errs.append(2 * d.std(ddof=1) / np.sqrt(len(d)))
        ax.bar(list(QUALITIES), means, yerr=errs, capsize=3, color=["tab:blue", "tab:orange"])
        ax.set_title(LABEL[key], fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Effect of sensing quality (bars = mean, whiskers = 2 standard errors)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2_sensing_quality.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 3) closest approach ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(col("min_sep"), bins=np.linspace(0, 1.4, 29), color="tab:green", alpha=0.8)
    ax.axvline(d_safe, color="k", ls="--", lw=1, label=f"d_safe = {d_safe} m")
    ax.set_xlabel("closest approach between two drones during a mission [m]"); ax.set_ylabel("number of missions")
    ax.set_title(f"Separation across {len(rows)} missions"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(OUT / "3_closest_approach.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    failed = [r for r in rows if r["completed"] < 1 or r["broke_dsafe"] > 0]
    if failed:
        print("\ncases to look at (not completed, or separation broken):")
        for r in failed:
            print(f"  pattern={r['pattern']} quality={r['quality']} seed={r['seed']} "
                  f"completed={bool(r['completed'])} closest={r['min_sep']:.2f} m")
    print(f"\nbaseline table saved to {BASE / 'classical.csv'}")
    print(f"plots saved in {OUT}")
