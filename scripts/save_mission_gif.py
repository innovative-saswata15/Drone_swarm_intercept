"""Save one benchmark mission as a GIF plus four stills (the target is removed the moment its investigation ends).

Run:  python scripts/save_mission_gif.py                      (straight target, clean sensing, benchmark seed 1)
      python scripts/save_mission_gif.py weave clean 4        (pattern, quality, seed)
Writes results/mission_<pattern>_<quality>_<seed>.gif and results/mission_<pattern>_<quality>_<seed>_<n>_<stage>.png
Uses the same drawing code as scripts/watch_rl.py (classical controller only). Needs: pip install -e ".[rl]"
The default case was chosen because the target keeps moving, the chase is visible (about 7 s) and the
drone needs about 3 s to fly back after the investigation, so the target's removal is clearly seen.
"""
import sys
import runpy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
pattern = sys.argv[1] if len(sys.argv) > 1 else "straight"
quality = sys.argv[2] if len(sys.argv) > 2 else "clean"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SPEED, FPS = 3.0, 25                                           # playback speed-up and frame rate

sys.argv = ["watch_rl.py", "--classical"]
Panel = runpy.run_path(str(ROOT / "scripts" / "watch_rl.py"), run_name="lib")["Panel"]   # drawing code only
from swarm_intercept.config import load_config
from swarm_intercept.rl.evaluate import fly_case

cfg = load_config()
c, lg, fsm = fly_case(cfg, pattern, quality, seed, None)       # None = classical controller, zero residual
ev = [e[0] for e in fsm.events]                                # launch, reached, investigated, rejoined
for e in fsm.events:
    print(f"  {e[0]:6.1f} s  {e[1]}")
if len(ev) < 4:
    sys.exit("this mission did not complete; try another seed")
dt, ts = cfg["sim"]["dt"], cfg["sim"]["time_scale"]
step = max(1, int(round(SPEED * ts / (FPS * dt))))
fr = lambda sec: int(round(sec / dt)) // step                  # simulation time -> animation frame

fig, ax = plt.subplots(figsize=(7, 7))
panel = Panel(ax, cfg, c, lg, fsm, "Classical")
draw = lambda f: panel.draw(f * step) or []
out, tag = ROOT / "results", f"mission_{pattern}_{quality}_{seed}"
out.mkdir(exist_ok=True)
anim = FuncAnimation(fig, draw, frames=range(fr(ev[0] - 3.0), fr(ev[-1] + 8.0)), interval=1000 / FPS)
anim.save(str(out / f"{tag}.gif"), writer=PillowWriter(fps=FPS), dpi=80)
for name, sec in (("1_sent", ev[0] + 2.0), ("2_investigating", (ev[1] + ev[2]) / 2),
                  ("3_target_removed", ev[2] + 0.8), ("4_formation", ev[-1] + 6.0)):
    draw(fr(sec))
    fig.savefig(str(out / f"{tag}_{name}.png"), dpi=110, bbox_inches="tight")
print(f"saved results/{tag}.gif and 4 stills")
