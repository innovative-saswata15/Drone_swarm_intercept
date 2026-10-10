"""Step 2.3 - report figures: classical controller vs residual SAC.

Run:  python scripts/step2_3_plots.py
      python scripts/step2_3_plots.py --reference 15.77      (skip recomputing the classical reference score)
Reads  baselines/classical.csv, baselines/residual_sac.csv (or results/step2_2/residual_sac.csv),
       results/step2_1/evaluations.npz (written by the training run)
Writes results/step2_3/*.png
    1_learning_curve.png       evaluation score during training vs the classical controller
    2_metric_changes.png       change of every benchmark metric, with 95% confidence intervals
    3_mission_time.png         mission time per case (scatter) and its distribution
    4_mission_time_by_group.png  mission time by target pattern and sensing quality
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
# (column, label, +1 if higher is better, -1 if lower is better)
METRICS = [("t_mission", "Mission time", -1), ("t_reach", "Time to reach target", -1), ("t_reform", "Re-form time", -1),
           ("in_view", "Target in view", +1), ("track_err", "Tracking error", -1), ("effort", "Turning effort", -1),
           ("filter_active", "Safety filter active", -1), ("orbit_dist", "Orbit distance error", -1),
           ("min_sep", "Closest approach", +1)]
CLASSICAL_COLOR, RESIDUAL_COLOR = "#7f7f7f", "#1f77b4"


def read_csv(path):
    with open(path, newline="") as f:
        return {(r["pattern"], r["quality"], int(r["seed"])): {k: (v if k in ("pattern", "quality") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)}


def paired(cla, res, col):
    keys = sorted(set(cla) & set(res))
    a = np.array([cla[k][col] for k in keys]); b = np.array([res[k][col] for k in keys])
    ok = ~np.isnan(a) & ~np.isnan(b)
    return [k for k, o in zip(keys, ok) if o], a[ok], b[ok]


def boot_pct_change(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    vals = 100 * (b[idx].mean(1) / a[idx].mean(1) - 1)
    return np.percentile(vals, [2.5, 97.5])


def classical_reference():
    """Score of the classical controller (zero residual) on the fixed evaluation missions used in training."""
    from swarm_intercept.rl.env import FixedCaseEnv, eval_cases
    env = FixedCaseEnv(cases=eval_cases(16))
    scores = []
    for _ in range(16):
        env.reset(); total, over = 0.0, env._over
        while not over:
            _, r, term, trunc, _ = env.step(np.zeros(2)); total += r; over = term or trunc
        scores.append(total)
    return float(np.mean(scores))


def fig_learning_curve(npz_path, reference, out):
    if not Path(npz_path).exists():
        print(f"  skipped learning curve: {npz_path} not found (run the training first)"); return
    d = np.load(npz_path)
    steps, res = d["timesteps"] / 1000.0, d["results"]
    mean, std = res.mean(1), res.std(1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, mean, "o-", color=RESIDUAL_COLOR, label="Residual SAC (mean of 16 missions)")
    ax.fill_between(steps, mean - std, mean + std, color=RESIDUAL_COLOR, alpha=0.15, label="±1 std across missions")
    if reference is not None:
        ax.axhline(reference, color=CLASSICAL_COLOR, ls="--", label=f"Classical controller ({reference:.1f})")
    ax.set_xlabel("Training steps (thousands)"); ax.set_ylabel("Evaluation score (episode reward)")
    ax.set_title("Learning curve on 16 fixed evaluation missions"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_metric_changes(cla, res, out):
    rows = []
    for col, label, sign in METRICS:
        _, a, b = paired(cla, res, col)
        if len(a) < 5 or a.mean() == 0:
            continue
        pct = 100 * (b.mean() / a.mean() - 1); lo, hi = boot_pct_change(a, b)
        rows.append((label, pct, lo, hi, sign, len(a)))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (label, pct, lo, hi, sign, n) in enumerate(rows):
        significant = lo > 0 or hi < 0
        good = (pct * sign) > 0
        color = ("#2ca02c" if good else "#d62728") if significant else "#bbbbbb"
        ax.barh(i, pct, color=color, xerr=[[pct - lo], [hi - pct]], capsize=3, ecolor="black")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[0]}  (n={r[5]})" for r in rows]); ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Change vs classical [%]   (bars: 95% confidence interval)")
    ax.set_title("Residual SAC vs classical: change in every benchmark metric")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ca02c", label="significantly better"), Patch(color="#d62728", label="significantly worse"),
                       Patch(color="#bbbbbb", label="no clear change (interval includes 0)")], loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_mission_time(cla, res, out):
    keys, a, b = paired(cla, res, "t_mission")
    q = np.array([k[1] for k in keys])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, color in (("clean", "#1f77b4"), ("degraded", "#ff7f0e")):
        m = q == name
        ax1.scatter(a[m], b[m], s=18, alpha=0.7, color=color, label=f"{name} sensing")
    lim = [0, max(a.max(), b.max()) * 1.05]
    ax1.plot(lim, lim, "k--", lw=1); ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.fill_between(lim, [0, 0], lim, color="#2ca02c", alpha=0.06)
    ax1.text(lim[1] * 0.97, lim[1] * 0.08, "residual faster", ha="right", color="#2ca02c")
    ax1.text(lim[1] * 0.03, lim[1] * 0.92, "residual slower", ha="left", color="#d62728")
    ax1.set_xlabel("Classical mission time [s]"); ax1.set_ylabel("Residual SAC mission time [s]")
    ax1.set_title(f"Each point is one mission ({(b < a - 1e-6).sum()} faster, {(b > a + 1e-6).sum()} slower)")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    bins = np.arange(min(a.min(), b.min()) // 2 * 2, max(a.max(), b.max()) + 2, 2)
    ax2.hist(a, bins, color=CLASSICAL_COLOR, alpha=0.6, label=f"Classical (mean {a.mean():.2f} s)")
    ax2.hist(b, bins, color=RESIDUAL_COLOR, alpha=0.6, label=f"Residual SAC (mean {b.mean():.2f} s)")
    ax2.axvline(a.mean(), color=CLASSICAL_COLOR, ls="--"); ax2.axvline(b.mean(), color=RESIDUAL_COLOR, ls="--")
    ax2.set_xlabel("Mission time [s]"); ax2.set_ylabel("Number of missions"); ax2.grid(alpha=0.3); ax2.legend()
    ax2.set_title(f"Distribution: slowest {a.max():.1f} s -> {b.max():.1f} s")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def fig_by_group(cla, res, out):
    keys, a, b = paired(cla, res, "t_mission")
    patterns = ["straight", "weave", "random_turns", "stop_go"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, quality in zip(axes, ("clean", "degraded")):
        ca, rb = [], []
        for p in patterns:
            m = np.array([(k[0] == p and k[1] == quality) for k in keys])
            ca.append(a[m].mean()); rb.append(b[m].mean())
        x = np.arange(len(patterns))
        ax.bar(x - 0.2, ca, 0.4, color=CLASSICAL_COLOR, label="Classical"); ax.bar(x + 0.2, rb, 0.4, color=RESIDUAL_COLOR, label="Residual SAC")
        for xi, c, r in zip(x, ca, rb):
            ax.text(xi + 0.2, r + 0.2, f"{100 * (r / c - 1):+.0f}%", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(patterns); ax.set_title(f"{quality} sensing"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean mission time [s]"); axes[0].set_ylim(0, max(a.max(), 1) and np.max([ax.get_ylim()[1] for ax in axes]) * 1.15)
    axes[0].legend(loc="upper right")
    fig.suptitle("Mission time by target pattern (percent = change vs classical)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classical", default=str(ROOT / "baselines" / "classical.csv"))
    ap.add_argument("--residual", default=None)
    ap.add_argument("--evals", default=str(ROOT / "results" / "step2_1" / "evaluations.npz"))
    ap.add_argument("--out", default=str(ROOT / "results" / "step2_3"))
    ap.add_argument("--reference", type=float, default=None, help="classical score on the 16 training-evaluation missions")
    ap.add_argument("--skip-reference", action="store_true")
    a = ap.parse_args()
    res_path = a.residual or next((str(p) for p in (ROOT / "baselines" / "residual_sac.csv", ROOT / "results" / "step2_2" / "residual_sac.csv") if p.exists()), None)
    if res_path is None:
        raise SystemExit("residual_sac.csv not found: run scripts/step2_2_evaluate.py first, or pass --residual")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cla, res = read_csv(a.classical), read_csv(res_path)
    print(f"classical: {a.classical}\nresidual:  {res_path}\n{len(set(cla) & set(res))} missions in common\n")

    reference = a.reference
    if reference is None and not a.skip_reference and Path(a.evals).exists():
        print("  computing the classical reference score (about 30 s) ...")
        reference = classical_reference()
    fig_learning_curve(a.evals, reference, out / "1_learning_curve.png")
    fig_metric_changes(cla, res, out / "2_metric_changes.png")
    fig_mission_time(cla, res, out / "3_mission_time.png")
    fig_by_group(cla, res, out / "4_mission_time_by_group.png")
    for p in sorted(out.glob("*.png")):
        print("  wrote", p)


if __name__ == "__main__":
    main()
