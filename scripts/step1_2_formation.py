"""Step 1.2 - three drones hold equal spacing; one leaves and returns.

Run:  python scripts/step1_2_formation.py
Plots and an animation are saved in results/step1_2/
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from swarm_intercept.config import load_config
from swarm_intercept.sim_multi import simulate_formation

OUT = Path(__file__).resolve().parent.parent / "results" / "step1_2"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config()
R = cfg["formation"]["radius"]
half = cfg["arena"]["half_size"]
TS = cfg["sim"]["time_scale"]          # 1.0 for the reference setup; >1 when the drones are slower / circle larger
T, T_LEAVE, T_RETURN = 150.0 * TS, 50.0 * TS, 100.0 * TS
COLORS = ["tab:blue", "tab:orange", "tab:green"]
NAMES = ["drone 1", "drone 2", "drone 3"]

states0 = np.array([[1.9, 0.0, 2.0],
                    [1.5, 1.0, -1.0],
                    [0.3, -0.2, 0.5]])
lg = simulate_formation(states0, cfg, leave=(2, T_LEAVE, T_RETURN), T=T)
t = lg["t"]


def mark_events(ax):
    for tt, lab in ((T_LEAVE, "drone 3 leaves"), (T_RETURN, "drone 3 returns")):
        ax.axvline(tt, color="k", ls=":", lw=1)
        ax.text(tt + 0.8 * TS, ax.get_ylim()[1], lab, va="top", fontsize=8)


# 1) trajectories of the first phase --------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ang = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1, label="formation circle")
m = t < T_LEAVE
for j in range(3):
    ax.plot(lg["x"][m, j], lg["y"][m, j], color=COLORS[j], lw=1, alpha=0.6)
    ax.plot(lg["x"][0, j], lg["y"][0, j], "o", color=COLORS[j], ms=6)
    k_end = np.where(m)[0][-1]
    ax.plot(lg["x"][k_end, j], lg["y"][k_end, j], "s", color=COLORS[j], ms=9, label=f"{NAMES[j]} at t={T_LEAVE:.0f}s")
ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Three drones join the circle and spread to 120 deg (dots = starts)")
ax.legend(loc="upper left", fontsize=8)
fig.savefig(OUT / "1_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2) gaps -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for j in range(3):
    ax.plot(t, np.degrees(lg["gap"][:, j]), color=COLORS[j], lw=1.3, label=f"gap ahead of {NAMES[j]}")
ax.axhline(120, color="k", ls="--", lw=1); ax.axhline(180, color="gray", ls="--", lw=1)
ax.set_ylim(0, 300); ax.set_yticks([0, 60, 120, 180, 240, 300])
ax.set_xlabel("time [s]"); ax.set_ylabel("angular gap [deg]"); ax.grid(alpha=0.3)
ax.set_title("Gaps settle at 120 deg with three drones and 180 deg with two")
mark_events(ax); ax.legend(loc="lower right", fontsize=8)
fig.savefig(OUT / "2_gaps.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3) speeds -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for j in range(3):
    ax.plot(t, lg["v"][:, j], color=COLORS[j], lw=1.2, label=NAMES[j])
ax.axhline(cfg["drone"]["v"], color="k", ls="--", lw=1)
ax.set_ylim(0.9 * cfg["formation"]["v_min"], 1.1 * cfg["formation"]["v_max"])
ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]"); ax.grid(alpha=0.3)
ax.set_title("Speeds differ only while the spacing is being corrected")
mark_events(ax); ax.legend(loc="lower right", fontsize=8)
fig.savefig(OUT / "3_speeds.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 4) path error -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for j in range(3):
    ax.plot(t, lg["e"][:, j], color=COLORS[j], lw=1.2, label=NAMES[j])
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("time [s]"); ax.set_ylabel("error to its own circle [m]"); ax.grid(alpha=0.3)
ax.set_title("Each drone's distance from the circle it is following")
mark_events(ax); ax.legend(loc="upper right", fontsize=8)
fig.savefig(OUT / "4_path_error.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 5) minimum separation ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t, lg["min_sep"], color="tab:red", lw=1.3, label="closest pair of drones")
ax.axhline(0.5, color="k", ls="--", lw=1, label="d_safe = 0.5 m")
ax.set_ylim(0, max(2.6, 1.9 * R)); ax.set_xlabel("time [s]"); ax.set_ylabel("distance [m]"); ax.grid(alpha=0.3)
ax.set_title("Minimum distance between any two drones (no safety filter yet)")
mark_events(ax); ax.legend(loc="lower right", fontsize=8)
fig.savefig(OUT / "5_min_separation.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 6) animation --------------------------------------------------------------
STEP = max(1, int(round(25 * TS)))           # 300 frames in total, whatever the run length
frames = range(0, len(t), STEP)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1)
ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
dots = [ax.plot([], [], "o", color=COLORS[j], ms=11, label=NAMES[j])[0] for j in range(3)]
tails = [ax.plot([], [], "-", color=COLORS[j], lw=1, alpha=0.5)[0] for j in range(3)]
title = ax.set_title("")
ax.legend(loc="upper left", fontsize=8)


def draw(i):
    lo = max(0, i - int(400 * TS))
    for j in range(3):
        dots[j].set_data([lg["x"][i, j]], [lg["y"][i, j]])
        tails[j].set_data(lg["x"][lo:i + 1, j], lg["y"][lo:i + 1, j])
    n_act = int(lg["active"][i].sum())
    title.set_text(f"t = {t[i]:5.1f} s   drones in formation: {n_act}")
    return dots + tails + [title]


anim = FuncAnimation(fig, draw, frames=frames, blit=False)
anim.save(OUT / "formation.gif", writer=PillowWriter(fps=20), dpi=80)
plt.close(fig)

# summary -------------------------------------------------------------------
def gaps_at(tt):
    i = np.searchsorted(t, tt) - 1
    g = np.degrees(lg["gap"][i]); return ", ".join("  -  " if np.isnan(x) else f"{x:5.1f}" for x in g)

print("gap ahead of drone 1, 2, 3 [deg]")
print(f"  just before drone 3 leaves  (t={T_LEAVE:5.0f}s): {gaps_at(T_LEAVE)}")
print(f"  just before drone 3 returns (t={T_RETURN:5.0f}s): {gaps_at(T_RETURN)}")
print(f"  end of run                  (t={T:5.0f}s): {gaps_at(T)}")
print(f"smallest distance between two drones over the whole run: {lg['min_sep'].min():.3f} m")
print(f"plots and formation.gif saved in {OUT}")
