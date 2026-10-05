"""Watch the residual-RL agent fly - in a live window.

Run:  python scripts/watch_rl.py results/step2_1/best_model.zip              classical (left) vs residual SAC (right), same case
      python scripts/watch_rl.py results/step2_1/best_model.zip weave 7 4    pattern weave, benchmark seed 7, 4x playback
      python scripts/watch_rl.py --classical                                 classical controller only (no torch needed)
Options:  --pattern straight|weave|random_turns|stop_go   --quality clean|degraded   --seed N   --speed X
          --save out.gif     write an animation file instead of opening a window
Both panels fly the IDENTICAL case (same drone starts, same target path, same sensor noise).
Marked drones (red ring) are the ones that have left the formation. Close the window to stop.
"""
import argparse
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import numpy as np
import matplotlib
import sys
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

from swarm_intercept.config import load_config
from swarm_intercept.rl.evaluate import fly_case
from swarm_intercept.mission.fsm import STATE_NAMES, SURVEIL

COLORS = ["tab:blue", "tab:orange", "tab:green"]


class Panel:
    def __init__(self, ax, cfg, case_cfg, lg, fsm, name):
        self.ax, self.lg, self.fsm, self.name = ax, lg, fsm, name
        self.N = lg["x"].shape[1]
        R, half = cfg["formation"]["radius"], cfg["arena"]["half_size"]
        ang = np.linspace(0, 2 * np.pi, 400)
        ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=1)
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="gray", lw=1)
        ax.set_xlim(-half - 0.3, half + 0.3); ax.set_ylim(-half - 0.3, half + 0.3)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        self.dots = [ax.plot([], [], "o", color=COLORS[j % 3], ms=13)[0] for j in range(self.N)]
        self.noses = [ax.plot([], [], "-", color=COLORS[j % 3], lw=2.5)[0] for j in range(self.N)]
        self.tails = [ax.plot([], [], "-", color=COLORS[j % 3], lw=1, alpha=0.5)[0] for j in range(self.N)]
        fov = case_cfg["sensor"]["fov_radius"]
        self.discs = [Circle((0, 0), fov, color=COLORS[j % 3], alpha=0.15) for j in range(self.N)]
        for d in self.discs:
            ax.add_patch(d)
        self.true_dot, = ax.plot([], [], "X", color="tab:red", ms=12, label="target (truth)")
        self.est_dot, = ax.plot([], [], "+", color="tab:purple", ms=15, mew=2.5, label="Kalman estimate")
        ax.legend(loc="upper right", fontsize=8)
        self.title = ax.set_title(name)
        self.tail_len = int(8.0 * case_cfg["sim"]["time_scale"] / case_cfg["sim"]["dt"])

    def draw(self, i):
        lg = self.lg
        i = min(i, len(lg["t"]) - 1)
        lo = max(0, i - self.tail_len)
        mode = lg["mode"][i]
        for j in range(self.N):
            x, y, th = lg["x"][i, j], lg["y"][i, j], lg["theta"][i, j]
            away = mode[j] != SURVEIL
            self.dots[j].set_data([x], [y])
            self.dots[j].set_markeredgecolor("red" if away else "none")
            self.dots[j].set_markeredgewidth(2.5 if away else 0)
            self.noses[j].set_data([x, x + 0.22 * np.cos(th)], [y, y + 0.22 * np.sin(th)])
            self.tails[j].set_data(lg["x"][lo:i + 1, j], lg["y"][lo:i + 1, j])
            self.discs[j].center = (x, y)
        if lg["present"][i]:
            self.true_dot.set_data([lg["target"][i, 0]], [lg["target"][i, 1]])
        else:
            self.true_dot.set_data([], [])
        if lg["tracked"][i]:
            self.est_dot.set_data([lg["est"][i, 0]], [lg["est"][i, 1]])
        else:
            self.est_dot.set_data([], [])
        busy = [f"drone {j + 1}: {STATE_NAMES[mode[j]]}" for j in range(self.N) if mode[j] != SURVEIL]
        state = busy[0] if busy else ("target detected" if lg["tracked"][i] else "all in formation")
        self.title.set_text(f"{self.name}   t = {lg['t'][i]:5.1f} s   {state}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="trained SAC model (.zip)")
    ap.add_argument("pattern_pos", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("seed_pos", nargs="?", type=int, help=argparse.SUPPRESS)
    ap.add_argument("speed_pos", nargs="?", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--quality", default="clean", choices=["clean", "degraded"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--save", default=None, help="write a .gif instead of opening a window")
    a = ap.parse_args()
    if a.classical and a.model is not None:      # '--classical weave 3' : shift the positionals
        a.speed_pos, a.seed_pos, a.pattern_pos, a.model = a.seed_pos, a.pattern_pos, a.model, None
    pattern = a.pattern or a.pattern_pos or "weave"
    seed = a.seed if a.seed is not None else (int(a.seed_pos) if a.seed_pos is not None else 7)
    speed = a.speed or a.speed_pos or 3.0
    if not a.classical and a.model is None:
        ap.error("give a model .zip, or use --classical")

    cfg = load_config()
    print(f"flying case: pattern={pattern}, quality={a.quality}, seed={seed} ...")
    runs = [("Classical", fly_case(cfg, pattern, a.quality, seed, None))]
    if not a.classical:
        from stable_baselines3 import SAC
        model = SAC.load(a.model, device="cpu")
        runs.append(("Classical + residual SAC", fly_case(cfg, pattern, a.quality, seed,
                                                           lambda o: model.predict(o, deterministic=True)[0])))
    for name, (c, lg, fsm) in runs:
        done = len(fsm.events) >= 4 and "back on" in fsm.events[-1][1]
        t_m = fsm.events[-1][0] - fsm.events[0][0] if done else float("nan")
        print(f"  {name:>26}: mission {'completed' if done else 'NOT completed'}, mission time {t_m:.1f} s, "
              f"closest approach {lg['min_sep_xy'].min():.2f} m")

    n = min(len(lg["t"]) for _, (_, lg, _) in runs)
    dt, ts = cfg["sim"]["dt"], cfg["sim"]["time_scale"]
    FPS = 25
    step = max(1, int(round(speed * ts / (FPS * dt))))
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 7), squeeze=False)
    panels = [Panel(axes[0][k], cfg, c, lg, fsm, name) for k, (name, (c, lg, fsm)) in enumerate(runs)]

    def draw(frame):
        for pnl in panels:
            pnl.draw(frame * step)
        return []

    anim = FuncAnimation(fig, draw, frames=n // step, interval=1000 / FPS, blit=False, repeat=True)
    if a.save:
        fig.set_size_inches(6.5 * len(runs), 6.5)
        anim.save(a.save, writer=PillowWriter(fps=FPS), dpi=55)
        print(f"saved {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()