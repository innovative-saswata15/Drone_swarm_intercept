"""Live animation window: watch the drones move.

Run:  python scripts/watch_live.py            (three-drone formation, Step 1.2)
      python scripts/watch_live.py single     (one drone, Step 1.1)
      python scripts/watch_live.py track      (formation + target + Kalman filter, Step 1.3)
      python scripts/watch_live.py track 2 stop_go   (speed 2x, target pattern: straight/weave/random_turns/stop_go)
      python scripts/watch_live.py mission    (the whole mission, Step 1.4)
      python scripts/watch_live.py mission 2 weave   (speed 2x, target pattern)
      python scripts/watch_live.py formation 4   (4x faster playback)
Close the window to stop.
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from swarm_intercept.config import load_config

mode = sys.argv[1] if len(sys.argv) > 1 else "formation"
speedup = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

cfg = load_config()
R = cfg["formation"]["radius"]
half = cfg["arena"]["half_size"]
dt = cfg["sim"]["dt"]
TS = cfg["sim"]["time_scale"]          # run lengths and playback speed scale with R/v
COLORS = ["tab:blue", "tab:orange", "tab:green"]
trk = None                              # tracking log, only in 'track' mode

if mode == "single":
    from swarm_intercept.sim_single import simulate_single
    lg1 = simulate_single([1.9, 0.0, 0.0], cfg, T=40.0 * TS)
    t = lg1["t"]
    X, Y, TH = lg1["x"][:, None], lg1["y"][:, None], lg1["theta"][:, None]
    active = np.ones_like(X, dtype=bool)
    info = lambda i: f"path error e = {lg1['e'][i] * 100:6.1f} cm"
elif mode == "track":
    from swarm_intercept.sim_track import simulate_tracking
    pattern = sys.argv[3] if len(sys.argv) > 3 else "weave"
    trk = simulate_tracking(cfg, pattern, T=120.0 * TS)
    t, X, Y, TH = trk["t"], trk["x"], trk["y"], trk["theta"]
    active = np.ones_like(X, dtype=bool)

    def info(i):
        if not trk["tracked"][i]:
            return "target: no track"
        return f"target: {'IN VIEW' if trk['in_view'][i] else 'coasting'}, error {trk['pos_err'][i] * 100:4.0f} cm"
elif mode == "mission":
    from swarm_intercept.sim_mission import simulate_mission
    from swarm_intercept.mission.fsm import STATE_NAMES, SURVEIL
    pattern = sys.argv[3] if len(sys.argv) > 3 else "straight"
    trk, fsm = simulate_mission(cfg, pattern, T=90.0 * TS)
    t, X, Y, TH = trk["t"], trk["x"], trk["y"], trk["theta"]
    active = trk["mode"] == SURVEIL
    trk["target"] = np.where(trk["present"][:, None], trk["target"], np.nan)

    def info(i):
        busy = [f"drone {j + 1}: {STATE_NAMES[trk['mode'][i, j]]}" for j in range(X.shape[1]) if not active[i, j]]
        return busy[0] if busy else ("target detected" if trk["tracked"][i] else "all in formation")
else:
    from swarm_intercept.sim_multi import simulate_formation
    starts = np.array([[1.9, 0.0, 2.0], [1.5, 1.0, -1.0], [0.3, -0.2, 0.5]])
    lg = simulate_formation(starts, cfg, leave=(2, 50.0 * TS, 100.0 * TS), T=150.0 * TS)
    t, X, Y, TH, active = lg["t"], lg["x"], lg["y"], lg["theta"], lg["active"]

    def info(i):
        gaps = ", ".join(f"{np.degrees(g):.0f}" for g in lg["gap"][i] if not np.isnan(g))
        return f"in formation: {int(active[i].sum())}   gaps [deg]: {gaps}"

N = X.shape[1]
FPS = 25
step = max(1, int(round(speedup * TS / (FPS * dt))))  # simulation steps per frame

fig, ax = plt.subplots(figsize=(7, 7))
ang = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1)
ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", lw=1)
ax.set_xlim(-half - 0.3, half + 0.3); ax.set_ylim(-half - 0.3, half + 0.3)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
dots = [ax.plot([], [], "o", color=COLORS[j % 3], ms=13)[0] for j in range(N)]
noses = [ax.plot([], [], "-", color=COLORS[j % 3], lw=2.5)[0] for j in range(N)]
tails = [ax.plot([], [], "-", color=COLORS[j % 3], lw=1, alpha=0.5)[0] for j in range(N)]
title = ax.set_title("")
if trk is not None:
    fov = cfg["sensor"]["fov_radius"]
    discs = [Circle((0, 0), fov, color=COLORS[j % 3], alpha=0.15) for j in range(N)]
    for d in discs:
        ax.add_patch(d)
    true_dot, = ax.plot([], [], "X", color="tab:red", ms=12, label="target (truth)")
    est_dot, = ax.plot([], [], "+", color="tab:purple", ms=15, mew=2.5, label="Kalman estimate")
    ax.legend(loc="upper right", fontsize=8)


def draw(frame):
    i = min(frame * step, len(t) - 1)
    lo = max(0, i - int(8.0 * TS / dt))               # tail
    for j in range(N):
        x, y, th = X[i, j], Y[i, j], TH[i, j]
        dots[j].set_data([x], [y])
        dots[j].set_alpha(1.0 if (active[i, j] or mode == "mission") else 0.45)
        dots[j].set_markeredgecolor("none" if active[i, j] else "red")
        dots[j].set_markeredgewidth(0 if active[i, j] else 2.5)
        noses[j].set_data([x, x + 0.22 * np.cos(th)], [y, y + 0.22 * np.sin(th)])
        tails[j].set_data(X[lo:i + 1, j], Y[lo:i + 1, j])
    if trk is not None:
        for j in range(N):
            discs[j].center = (X[i, j], Y[i, j])
        true_dot.set_data([trk["target"][i, 0]], [trk["target"][i, 1]])
        if trk["tracked"][i]:
            est_dot.set_data([trk["est"][i, 0]], [trk["est"][i, 1]])
        else:
            est_dot.set_data([], [])
    title.set_text(f"t = {t[i]:6.1f} s     {info(i)}")
    return dots + noses + tails + [title]


anim = FuncAnimation(fig, draw, frames=len(t) // step, interval=1000 / FPS, blit=False, repeat=True)
if __name__ == "__main__":
    plt.show()
