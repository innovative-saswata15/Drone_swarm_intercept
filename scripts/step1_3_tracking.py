"""Step 1.3 - moving target, virtual sensor and Kalman filter.

Run:  python scripts/step1_3_tracking.py            (weaving target)
      python scripts/step1_3_tracking.py straight   (or: random_turns, stop_go)
Plots and an animation are saved in results/step1_3/
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter

from swarm_intercept.config import load_config
from swarm_intercept.sim_track import simulate_tracking
from swarm_intercept.core.target import PATTERNS

pattern = sys.argv[1] if len(sys.argv) > 1 else "weave"
OUT = Path(__file__).resolve().parent.parent / "results" / "step1_3"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config()
TS = cfg["sim"]["time_scale"]
R, half, fov = cfg["formation"]["radius"], cfg["arena"]["half_size"], cfg["sensor"]["fov_radius"]
T = 120.0 * TS
COLORS = ["tab:blue", "tab:orange", "tab:green"]

lg = simulate_tracking(cfg, pattern, T=T)
t, tgt, est = lg["t"], lg["target"], lg["est"]
ang = np.linspace(0, 2 * np.pi, 400)


def shade_in_view(ax):
    """Grey bands = times when the target is inside some drone's field of view."""
    v = lg["in_view"].astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], v, [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="0.85", lw=0)


# 1) map ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", lw=1)
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1, label="formation circle")
ax.plot(tgt[:, 0], tgt[:, 1], color="tab:red", lw=1.6, label="true target path")
seen = np.where(lg["in_view"][:, None], est[:, :2], np.nan)       # estimate while the target is in view
coast = np.where(lg["in_view"][:, None], np.nan, est[:, :2])      # estimate while coasting on prediction
ax.plot(coast[:, 0], coast[:, 1], color="tab:purple", lw=0.9, ls=":", label="estimate, coasting (no measurements)")
ax.plot(seen[:, 0], seen[:, 1], color="tab:purple", lw=2.0, label="estimate, target in view")
ax.plot(lg["meas"][:, 0], lg["meas"][:, 1], ".", color="k", ms=2.5, label="measurements")
ax.plot(tgt[0, 0], tgt[0, 1], "o", color="tab:red", ms=7)
ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title(f"Target is measured only near the drones ('{pattern}' pattern)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
fig.savefig(OUT / f"1_map_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2) position error ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
shade_in_view(ax)
ax.plot(t, 3 * lg["sigma"], color="tab:purple", lw=1, ls="--", label="filter's own 3-sigma uncertainty")
ax.plot(t, lg["pos_err"], color="tab:red", lw=1.2, label="actual position error")
ax.set_ylim(0, max(1.0, np.nanmax(lg["pos_err"]) * 1.15))
ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]"); ax.grid(alpha=0.3)
ax.set_title("Estimate is accurate while the target is in view (grey) and drifts between sightings")
ax.legend(loc="upper left", fontsize=8)
fig.savefig(OUT / f"2_position_error_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3) velocity ----------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
for ax, k, name in zip(axes, (2, 3), ("vx", "vy")):
    shade_in_view(ax)
    ax.plot(t, tgt[:, k], color="tab:red", lw=1.4, label="true")
    ax.plot(t, est[:, k], color="tab:purple", lw=1.1, label="estimated")
    ax.set_ylabel(f"{name} [m/s]"); ax.grid(alpha=0.3)
axes[0].set_title("Target velocity is never measured: the filter infers it from positions")
axes[0].legend(loc="upper right", fontsize=8); axes[1].set_xlabel("time [s]")
fig.savefig(OUT / f"3_velocity_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 4) animation ---------------------------------------------------------------
STEP = max(1, int(round(20 * TS)))
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", lw=1)
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=0.8)
ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
discs = [Circle((0, 0), fov, color=COLORS[j], alpha=0.15) for j in range(3)]
for d in discs:
    ax.add_patch(d)
dots = [ax.plot([], [], "o", color=COLORS[j], ms=9)[0] for j in range(3)]
true_dot, = ax.plot([], [], "X", color="tab:red", ms=11, label="target (truth)")
est_dot, = ax.plot([], [], "+", color="tab:purple", ms=13, mew=2.2, label="estimate")
ell = Ellipse((0, 0), 0, 0, fill=False, color="tab:purple", lw=1.2); ax.add_patch(ell)
title = ax.set_title(""); ax.legend(loc="upper right", fontsize=8)


def draw(i):
    for j in range(3):
        discs[j].center = (lg["x"][i, j], lg["y"][i, j])
        dots[j].set_data([lg["x"][i, j]], [lg["y"][i, j]])
    true_dot.set_data([tgt[i, 0]], [tgt[i, 1]])
    if lg["tracked"][i]:
        est_dot.set_data([est[i, 0]], [est[i, 1]])
        vals, vecs = np.linalg.eigh(lg["P"][i, :2, :2])
        ell.set_center((est[i, 0], est[i, 1]))
        ell.width, ell.height = 6 * np.sqrt(vals[1]), 6 * np.sqrt(vals[0])       # 3-sigma ellipse
        ell.angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])); ell.set_visible(True)
        state = "IN VIEW" if lg["in_view"][i] else "coasting"
    else:
        est_dot.set_data([], []); ell.set_visible(False); state = "no track"
    title.set_text(f"t = {t[i]:5.1f} s   {state}")
    return dots + [true_dot, est_dot, ell, title]


anim = FuncAnimation(fig, draw, frames=range(0, len(t), STEP), blit=False)
anim.save(OUT / f"tracking_{pattern}.gif", writer=PillowWriter(fps=20), dpi=80)
plt.close(fig)

# summary over all patterns --------------------------------------------------
print(f"{'pattern':>13} | {'first seen [s]':>14} | {'in view':>7} | {'tracked':>7} | "
      f"{'error in view [cm]':>18} | {'error coasting [cm]':>19} | {'worst [cm]':>10}")
for p in PATTERNS:
    g = lg if p == pattern else simulate_tracking(cfg, p, T=T)
    tr, iv = g["tracked"], g["in_view"]
    first = g["t"][np.argmax(tr)] if tr.any() else float("nan")
    rms = lambda m: 100 * np.sqrt(np.mean(g["pos_err"][m] ** 2)) if m.any() else float("nan")
    print(f"{p:>13} | {first:14.1f} | {100 * iv.mean():6.0f}% | {100 * tr.mean():6.0f}% | "
          f"{rms(tr & iv):18.1f} | {rms(tr & ~iv):19.1f} | {100 * np.nanmax(g['pos_err']):10.0f}")
print(f"\nplots and tracking_{pattern}.gif saved in {OUT}")
