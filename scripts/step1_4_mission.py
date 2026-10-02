"""Step 1.4 - the complete classical mission.

surveil -> target detected -> choose a drone -> intercept -> investigate -> return and rejoin

Run:  python scripts/step1_4_mission.py            (straight target)
      python scripts/step1_4_mission.py weave      (or: random_turns, stop_go)
Plots and an animation are saved in results/step1_4/
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from swarm_intercept.config import load_config
from swarm_intercept.sim_mission import simulate_mission
from swarm_intercept.mission.fsm import STATE_NAMES, SURVEIL, INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.core.target import PATTERNS

pattern = sys.argv[1] if len(sys.argv) > 1 else "straight"
OUT = Path(__file__).resolve().parent.parent / "results" / "step1_4"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config()
TS = cfg["sim"]["time_scale"]
R, half = cfg["formation"]["radius"], cfg["arena"]["half_size"]
ms, zone = cfg["mission"], cfg["sensor"]["area"]["half_size"]
T = 90.0 * TS
DRONE = ["tab:blue", "tab:orange", "tab:green"]
MODE = {SURVEIL: "0.6", INTERCEPT: "tab:red", INVESTIGATE: "tab:purple", RETURN: "tab:olive"}
D_SAFE = 0.5

lg, fsm = simulate_mission(cfg, pattern, T=T)
t, mode, tgt = lg["t"], lg["mode"], lg["target"]
ang = np.linspace(0, 2 * np.pi, 400)
who = int(np.argmax((mode != SURVEIL).any(axis=0))) if (mode != SURVEIL).any() else None


def arena(ax):
    ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", lw=1)
    ax.add_patch(Rectangle((-zone, -zone), 2 * zone, 2 * zone, fill=False, ls=":", color="tab:red", lw=1))
    ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=0.8)
    ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


# 1) map: the interceptor's path coloured by mission state ---------------------
fig, ax = plt.subplots(figsize=(6.8, 7.2))
arena(ax)
ax.plot(tgt[:, 0], tgt[:, 1], color="k", lw=1.4, label="target path")
if who is not None:
    busy = np.where(mode[:, who] != SURVEIL)[0]
    lo, hi = max(busy[0] - int(3 * TS / cfg["sim"]["dt"]), 0), min(busy[-1] + int(3 * TS / cfg["sim"]["dt"]), len(t) - 1)
    for m in (SURVEIL, INTERCEPT, INVESTIGATE, RETURN):
        xs = np.where(mode[lo:hi, who] == m, lg["x"][lo:hi, who], np.nan)
        ys = np.where(mode[lo:hi, who] == m, lg["y"][lo:hi, who], np.nan)
        ax.plot(xs, ys, color=MODE[m], lw=2.2, label=f"drone {who + 1}: {STATE_NAMES[m].lower()}")
    ax.plot(lg["x"][busy[0], who], lg["y"][busy[0], who], "o", color=MODE[INTERCEPT], ms=8)
    ax.plot(tgt[busy[0], 0], tgt[busy[0], 1], "X", color="k", ms=9)
ax.set_title(f"Mission path of the chosen drone ('{pattern}' target)\ndotted square = protected zone")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
fig.savefig(OUT / f"1_map_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2) who does what, and the formation gaps ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
ax = axes[0]
for j in range(3):
    for m in (SURVEIL, INTERCEPT, INVESTIGATE, RETURN):
        on = (mode[:, j] == m).astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], on, [0]))))
        ax.broken_barh([(t[a], t[min(b, len(t) - 1)] - t[a]) for a, b in zip(edges[::2], edges[1::2])],
                       (j - 0.35, 0.7), color=MODE[m])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["drone 1", "drone 2", "drone 3"]); ax.invert_yaxis()
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODE[m]) for m in MODE],
          labels=[STATE_NAMES[m].lower() for m in MODE], ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
ax = axes[1]
for j in range(3):
    ax.plot(t, np.degrees(lg["gap"][:, j]), color=DRONE[j], lw=1.3, label=f"gap ahead of drone {j + 1}")
ax.axhline(120, color="k", ls="--", lw=1); ax.axhline(180, color="gray", ls="--", lw=1)
ax.set_ylim(0, 300); ax.set_yticks([0, 60, 120, 180, 240, 300]); ax.grid(alpha=0.3)
ax.set_xlabel("time [s]"); ax.set_ylabel("angular gap [deg]"); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("One drone leaves, the other two re-space to 180 deg, then all return to 120 deg", y=1.02, fontsize=11)
fig.savefig(OUT / f"2_timeline_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3) distance from the interceptor to the TRUE target --------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, lg["dist_true"], color="tab:red", lw=1.4, label="interceptor to target (true distance)")
ax.axhline(ms["capture_radius"], color="k", ls="--", lw=1, label=f"capture radius {ms['capture_radius']:.2f} m")
ax.axhline(ms["orbit_radius"], color="tab:purple", ls="--", lw=1, label=f"orbit radius {ms['orbit_radius']:.2f} m")
busy_t = t[~np.isnan(lg["dist_true"])]
if len(busy_t):
    ax.set_xlim(busy_t[0] - 2 * TS, busy_t[-1] + 2 * TS)
ax.set_ylim(0, None); ax.set_xlabel("time [s]"); ax.set_ylabel("distance [m]"); ax.grid(alpha=0.3)
ax.set_title("The drone closes in, then holds the orbit radius around the moving target")
ax.legend(loc="upper right", fontsize=8)
fig.savefig(OUT / f"3_distance_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 4) the selection ------------------------------------------------------------
if fsm.costs_at_decision is not None:
    costs = fsm.costs_at_decision
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(["drone 1", "drone 2", "drone 3"], costs, color=["0.75"] * 3)
    bars[int(np.argmin(costs))].set_color("tab:red")
    for b, cst in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, cst, f"{cst:.1f} s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("estimated time to reach the target [s]"); ax.set_ylim(0, costs.max() * 1.2)
    ax.set_title("Interceptor selection: lowest time wins")
    fig.savefig(OUT / f"4_selection_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 5) separation ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, lg["min_sep_xy"], color="tab:orange", lw=1.2, label="horizontal distance only")
ax.plot(t, lg["min_sep"], color="tab:red", lw=1.4, label="true 3-D distance (with altitude layers)")
ax.axhline(D_SAFE, color="k", ls="--", lw=1, label=f"d_safe = {D_SAFE} m")
ax.set_ylim(0, None); ax.set_xlabel("time [s]"); ax.set_ylabel("closest pair of drones [m]"); ax.grid(alpha=0.3)
ax.set_title("The interceptor crosses above the formation: paths cross, altitude keeps them apart")
ax.legend(loc="upper right", fontsize=8)
fig.savefig(OUT / f"5_separation_{pattern}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 6) animation ----------------------------------------------------------------
STEP = max(1, int(round(15 * TS)))
fig, ax = plt.subplots(figsize=(6.2, 6.6))
arena(ax)
dots = [ax.plot([], [], "o", color=DRONE[j], ms=10, mec="k", mew=0.5)[0] for j in range(3)]
tails = [ax.plot([], [], "-", color=DRONE[j], lw=1, alpha=0.5)[0] for j in range(3)]
true_dot, = ax.plot([], [], "X", color="k", ms=11, label="target")
est_dot, = ax.plot([], [], "+", color="tab:purple", ms=13, mew=2.2, label="estimate")
orbit = Circle((0, 0), ms["orbit_radius"], fill=False, color="tab:purple", ls=":", lw=1); ax.add_patch(orbit)
title = ax.set_title(""); ax.legend(loc="upper right", fontsize=8)


def draw(i):
    lo = max(0, i - int(6 * TS / cfg["sim"]["dt"]))
    for j in range(3):
        dots[j].set_data([lg["x"][i, j]], [lg["y"][i, j]])
        dots[j].set_markersize(10 + 8 * (lg["z"][i, j] - cfg["altitude"]["formation"]))   # bigger = higher
        dots[j].set_markeredgecolor(MODE[mode[i, j]] if mode[i, j] != SURVEIL else "k")
        dots[j].set_markeredgewidth(2.5 if mode[i, j] != SURVEIL else 0.5)
        tails[j].set_data(lg["x"][lo:i + 1, j], lg["y"][lo:i + 1, j])
    if lg["present"][i]:
        true_dot.set_data([tgt[i, 0]], [tgt[i, 1]])
    else:
        true_dot.set_data([], [])
    if lg["tracked"][i]:
        est_dot.set_data([lg["est"][i, 0]], [lg["est"][i, 1]]); orbit.center = (lg["est"][i, 0], lg["est"][i, 1])
        orbit.set_visible(bool((mode[i] == INVESTIGATE).any()))
    else:
        est_dot.set_data([], []); orbit.set_visible(False)
    busy = [f"drone {j + 1}: {STATE_NAMES[mode[i, j]]}" for j in range(3) if mode[i, j] != SURVEIL]
    title.set_text(f"t = {t[i]:5.1f} s   " + (busy[0] if busy else "all in formation"))
    return dots + tails + [true_dot, est_dot, orbit, title]


anim = FuncAnimation(fig, draw, frames=range(0, len(t), STEP), blit=False)
anim.save(OUT / f"mission_{pattern}.gif", writer=PillowWriter(fps=20), dpi=80)
plt.close(fig)

# summary ---------------------------------------------------------------------
print(f"mission log ('{pattern}' target, enters at t = {cfg['target']['enter_time'] * TS:.0f} s)")
for te, text in fsm.events:
    print(f"  t = {te:6.1f} s   {text}")
if fsm.costs_at_decision is not None:
    print("  selection costs [s]: " + ", ".join(f"drone {j + 1}: {c:.1f}" for j, c in enumerate(fsm.costs_at_decision)))
inv = (mode == INVESTIGATE).any(axis=1)
if inv.any():
    print(f"  distance to the target while investigating: {np.nanmean(lg['dist_true'][inv]):.2f} m "
          f"(orbit radius {ms['orbit_radius']:.2f} m)")
pres = lg["present"]
print(f"  closest pair of drones during the mission: {lg['min_sep_xy'][pres].min():.2f} m horizontally, "
      f"{lg['min_sep'][pres].min():.2f} m in 3-D")
print(f"  formation gaps at the end [deg]: " + ", ".join(f"{g:.1f}" for g in np.degrees(lg["gap"][-1])))

print(f"\n{'pattern':>13} | {'chosen':>7} | {'time to reach [s]':>17} | {'orbit distance [m]':>18} | {'mission time [s]':>16} | {'3-D min sep [m]':>15}")
for p in PATTERNS:
    g, f2 = (lg, fsm) if p == pattern else simulate_mission(cfg, p, T=T)
    ev = f2.events
    if len(ev) < 4:
        print(f"{p:>13} | mission did not complete"); continue
    iv = (g["mode"] == INVESTIGATE).any(axis=1)
    print(f"{p:>13} | drone {f2.events[0][1].split()[1].strip(':')} | {ev[1][0] - ev[0][0]:17.1f} | "
          f"{np.nanmean(g['dist_true'][iv]):18.2f} | {ev[-1][0] - ev[0][0]:16.1f} | {g['min_sep'][g['present']].min():15.2f}")
print(f"\nplots and mission_{pattern}.gif saved in {OUT}")
