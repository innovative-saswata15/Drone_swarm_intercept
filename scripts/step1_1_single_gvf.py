"""Step 1.1 - one drone converges to the formation circle from many starts.

Run:  python scripts/step1_1_single_gvf.py
Plots are saved in results/step1_1/
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")            # write PNG files, no window needed
import matplotlib.pyplot as plt

from swarm_intercept.config import load_config
from swarm_intercept.sim_single import simulate_single

OUT = Path(__file__).resolve().parent.parent / "results" / "step1_1"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config()
rng = np.random.default_rng(cfg["sim"]["seed"])
R = cfg["formation"]["radius"]
v = cfg["drone"]["v"]
half = cfg["arena"]["half_size"]
TS = cfg["sim"]["time_scale"]          # 1.0 for the reference setup; >1 when the drone is slower / circle larger
T = cfg["sim"]["T"] * TS

starts_xy = [(1.9, 0.0), (0.0, 1.9), (-1.8, -1.8), (0.2, 0.0),
             (0.0, -0.1), (1.5, 1.5), (-1.9, 0.6), (0.6, -0.5)]
starts = [np.array([x, y, rng.uniform(-np.pi, np.pi)]) for x, y in starts_xy]
logs = [simulate_single(s, cfg, T=T) for s in starts]

# 1) trajectories ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ang = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1.5, label="formation circle")
for s, lg in zip(starts, logs):
    line, = ax.plot(lg["x"], lg["y"], lw=1.2)
    ax.plot(s[0], s[1], "o", color=line.get_color(), ms=6)
ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Every start converges to the circle (dots = start points)")
ax.legend(loc="upper right")
fig.savefig(OUT / "1_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def time_plot(key, ylabel, title, fname, hline=None, hlabel=None, tmax=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for lg in logs:
        ax.plot(lg["t"], lg[key], lw=1.1)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1, label=hlabel); ax.legend()
    if tmax is not None:
        ax.set_xlim(0, tmax)
    ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


time_plot("e", "path error e [m]", "Path error goes to zero", "2_path_error.png", hline=0.0, hlabel="e = 0")
time_plot("gamma", "heading error gamma [rad]", "Heading error collapses first (fast loop)", "3_heading_error.png", tmax=8 * TS)
time_plot("omega", "turn rate omega [rad/s]", "Turn rate settles at v/R", "4_turn_rate.png",
          hline=v / R, hlabel=f"v/R = {v / R:.3f} rad/s")
time_plot("V", "V = 0.5 e^2 [m^2]", "Lyapunov function of the path error", "5_lyapunov.png", tmax=15 * TS)

# 6) gain sweep ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
s0 = np.array([1.9, 0.0, np.pi / 2])
for k in (0.3, 1.5, 5.0, 15.0):
    lg = simulate_single(s0, cfg, k_gvf=k, T=T)
    ax.plot(lg["t"], lg["e"], lw=1.3, label=f"k = {k}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(0, 20 * TS); ax.set_xlabel("time [s]"); ax.set_ylabel("path error e [m]")
ax.set_title("Effect of the GVF gain k"); ax.grid(alpha=0.3); ax.legend()
fig.savefig(OUT / "6_gain_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 7) feed-forward on/off ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for ff, lab in ((True, "with omega_d (feed-forward)"), (False, "omega_d = 0")):
    lg = simulate_single(s0, cfg, use_feedforward=ff, T=T)
    ax.plot(lg["t"], lg["e"], lw=1.3, label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("time [s]"); ax.set_ylabel("path error e [m]")
ax.set_title("Without feed-forward the drone settles off the circle"); ax.grid(alpha=0.3); ax.legend()
fig.savefig(OUT / "7_feedforward.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# summary ------------------------------------------------------------------
print(f"{'start (x, y)':>16} | {'final |e| [mm]':>14} | {'settle time [s]':>15} | {'mean omega [rad/s]':>18}")
for (x, y), lg in zip(starts_xy, logs):
    inside = np.abs(lg["e"]) < 0.02
    idx = np.where(~inside)[0]
    settle = 0.0 if len(idx) == 0 else lg["t"][min(idx[-1] + 1, len(lg["t"]) - 1)]
    tail = lg["t"] > lg["t"][-1] - 10 * TS
    print(f"({x:5.1f}, {y:5.1f})   | {abs(lg['e'][-1]) * 1000:14.3f} | {settle:15.2f} | {lg['omega'][tail].mean():18.4f}")
print(f"\nexpected steady turn rate v/R = {v / R:.4f} rad/s")
print(f"plots saved in {OUT}")
