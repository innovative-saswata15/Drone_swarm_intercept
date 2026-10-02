"""Step 1.5 - which intercept guidance law is best?

Three laws are flown on exactly the same randomised cases:
    pure   aim at the target
    lead   aim at the predicted meeting point
    pn     proportional navigation
in two conditions: the real 4 m 'room' and a 10 m 'large' arena with a faster target.

Run:  python scripts/step1_5_guidance.py          (200 cases per condition, about a minute)
      python scripts/step1_5_guidance.py 400      (more cases)
Plots are saved in results/step1_5/
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_intercept.config import load_config
from swarm_intercept.sim_intercept import simulate_intercept, SCENARIOS
from swarm_intercept.control.guidance import LAWS
from swarm_intercept.core.target import PATTERNS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = Path(__file__).resolve().parent.parent / "results" / "step1_5"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
COLOR = {"pure": "tab:blue", "lead": "tab:orange", "pn": "tab:green"}
NAME = {"pure": "pure pursuit", "lead": "lead pursuit", "pn": "proportional navigation"}

results = {sc: {law: [simulate_intercept(cfg, law, sc, s) for s in range(N)] for law in LAWS} for sc in SCENARIOS}


def col(sc, law, key):
    return np.array([r[key] for r in results[sc][law]], dtype=float)


for sc in SCENARIOS:
    S = SCENARIOS[sc]
    print(f"\n=== {sc}: {2 * S['half_size']:.0f} m arena, target at {S['target_speed_rel'] * cfg['drone']['v']:.2f} m/s, "
          f"interceptor at {cfg['mission']['intercept_speed']:.2f} m/s, {N} cases ===")
    print(f"{'law':>24} | {'caught':>6} | {'time [s]':>8} | {'path [m]':>8} | {'turning effort':>14}")
    for law in LAWS:
        print(f"{NAME[law]:>24} | {100 * col(sc, law, 'captured').mean():5.0f}% | {np.nanmean(col(sc, law, 'time')):8.2f} | "
              f"{col(sc, law, 'length').mean():8.2f} | {col(sc, law, 'effort').mean():14.2f}")
    base = col(sc, "pure", "time")
    for law in ("lead", "pn"):
        d = base - col(sc, law, "time")                    # positive = this law is faster than pure pursuit
        d = d[~np.isnan(d)]
        se = d.std(ddof=1) / np.sqrt(len(d))
        verdict = "faster" if d.mean() > 2 * se else ("slower" if d.mean() < -2 * se else "no clear difference")
        print(f"  {NAME[law]} vs pure pursuit: {d.mean():+.2f} s  (+/- {2 * se:.2f} s)  ->  {verdict}; "
              f"faster in {100 * np.mean(d > 0):.0f}% of cases")

# 1) capture time by target pattern ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, sc in zip(axes, SCENARIOS):
    pats = np.array([r["pattern"] for r in results[sc]["pure"]])
    x = np.arange(len(PATTERNS))
    for k, law in enumerate(LAWS):
        tm = col(sc, law, "time")
        means = [np.nanmean(tm[pats == p]) for p in PATTERNS]
        errs = [2 * np.nanstd(tm[pats == p], ddof=1) / np.sqrt((pats == p).sum()) for p in PATTERNS]
        ax.bar(x + (k - 1) * 0.27, means, 0.27, yerr=errs, capsize=2, color=COLOR[law], label=NAME[law])
    ax.set_xticks(x); ax.set_xticklabels(PATTERNS); ax.set_ylabel("mean time to capture [s]")
    ax.set_title(f"{sc} arena ({2 * SCENARIOS[sc]['half_size']:.0f} m)"); ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Time to capture by target pattern (bars = mean, whiskers = 2 standard errors)", fontsize=11)
fig.savefig(OUT / "1_capture_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2) time versus turning effort ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, sc in zip(axes, SCENARIOS):
    for law in LAWS:
        ax.plot(np.nanmean(col(sc, law, "time")), col(sc, law, "effort").mean(), "o", ms=13, color=COLOR[law], label=NAME[law])
    ax.set_xlabel("mean time to capture [s]"); ax.set_ylabel("mean turning effort (integral of omega^2)")
    ax.set_title(f"{sc} arena"); ax.grid(alpha=0.3); ax.margins(0.25)
axes[0].legend(fontsize=8)
fig.suptitle("Speed against smoothness: lower-left is better on both", fontsize=11)
fig.savefig(OUT / "2_time_vs_effort.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3) one example: the case where aiming ahead helps most ----------------------
gain = col("large", "pure", "time") - col("large", "lead", "time")
best = int(np.nanargmax(gain))
fig, ax = plt.subplots(figsize=(6.6, 6.6))
ref = results["large"]["pure"][best]
ax.plot(ref["target_path"][:, 0], ref["target_path"][:, 1], "k-", lw=1.6, label=f"target ({ref['pattern']})")
ax.plot(ref["target_path"][0, 0], ref["target_path"][0, 1], "kX", ms=9)
for law in LAWS:
    r = results["large"][law][best]
    ax.plot(r["path"][:, 0], r["path"][:, 1], color=COLOR[law], lw=1.8, label=f"{NAME[law]}: {r['time']:.1f} s")
ax.plot(ref["path"][0, 0], ref["path"][0, 1], "o", color="0.3", ms=8)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Example where aiming ahead pays off most (large arena)\ndot = drone start, cross = target start")
ax.legend(fontsize=8, loc="best")
fig.savefig(OUT / "3_example_paths.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print(f"\nplots saved in {OUT}")
