"""Step 1.6 - safety filter: the same mission and encounters, with the filter off and on.

Run:  python scripts/step1_6_safety.py        (5 random missions per target pattern, about 2 minutes)
      python scripts/step1_6_safety.py 10     (10 per pattern)
Plots are saved in results/step1_6/
"""
import sys
import copy
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_intercept.config import load_config
from swarm_intercept.sim_mission import simulate_mission
from swarm_intercept.sim_encounter import simulate_encounter, ENCOUNTERS
from swarm_intercept.mission.fsm import INVESTIGATE
from swarm_intercept.core.target import PATTERNS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = Path(__file__).resolve().parent.parent / "results" / "step1_6"
OUT.mkdir(parents=True, exist_ok=True)

cfg_on = load_config()
cfg_off = copy.deepcopy(cfg_on)
cfg_off["safety"]["enabled"] = False
TS = cfg_on["sim"]["time_scale"]
D_SAFE, half, R = cfg_on["safety"]["d_safe"], cfg_on["arena"]["half_size"], cfg_on["formation"]["radius"]
DRONE = ["tab:blue", "tab:orange", "tab:green"]
ang = np.linspace(0, 2 * np.pi, 400)

# 1) the standard mission, filter off and on ----------------------------------
off, _ = simulate_mission(cfg_off, "straight", T=80.0 * TS)
on, fsm_on = simulate_mission(cfg_on, "straight", T=80.0 * TS)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(off["t"], off["min_sep_xy"], color="tab:red", lw=1.3, label="filter off")
ax.plot(on["t"], on["min_sep_xy"], color="tab:green", lw=1.5, label="filter on")
ax.axhline(D_SAFE, color="k", ls="--", lw=1, label=f"d_safe = {D_SAFE} m")
ax.set_ylim(0, None); ax.set_xlabel("time [s]"); ax.set_ylabel("closest pair of drones, horizontal [m]"); ax.grid(alpha=0.3)
ax.set_title("With the filter the drones never come closer than d_safe")
ax.legend(loc="upper right", fontsize=8)
fig.savefig(OUT / "1_separation_before_after.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2) joining the circle, filter off and on -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
m = off["t"] < 20.0 * TS
for ax, lg, name in zip(axes, (off, on), ("filter off", "filter on")):
    ax.plot(R * np.cos(ang), R * np.sin(ang), "k--", lw=0.8)
    for j in range(3):
        ax.plot(lg["x"][m, j], lg["y"][m, j], color=DRONE[j], lw=1.4)
        ax.plot(lg["x"][0, j], lg["y"][0, j], "o", color=DRONE[j], ms=7)
    k = int(np.argmin(lg["min_sep_xy"][m]))
    ax.set_title(f"{name}: closest approach {lg['min_sep_xy'][m].min():.2f} m at t = {lg['t'][k]:.1f} s")
    ax.set_xlim(-half - 0.1, half + 0.1); ax.set_ylim(-half - 0.1, half + 0.1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
fig.suptitle("Joining the formation circle (first 20 s, dots = starts)", fontsize=11)
fig.savefig(OUT / "2_joining.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3) encounters ---------------------------------------------------------------
print("encounters (each drone flies to the opposite side)")
print(f"{'case':>10} | {'min distance off':>16} | {'min distance on':>15} | {'arrival times on [s]':>22} | {'without filter [s]':>18}")
fig, axes = plt.subplots(1, len(ENCOUNTERS), figsize=(4.4 * len(ENCOUNTERS), 4.6))
for ax, (name, (starts, goals)) in zip(axes, ENCOUNTERS.items()):
    a = simulate_encounter(starts, goals, cfg_off, use_filter=False, T=30.0 * TS)
    b = simulate_encounter(starts, goals, cfg_on, use_filter=True, T=30.0 * TS)
    fmt = lambda arr: ", ".join("  -  " if np.isnan(x) else f"{x:4.1f}" for x in arr)
    print(f"{name:>10} | {a['min_sep'].min():16.2f} | {b['min_sep'].min():15.2f} | {fmt(b['t_arrive']):>22} | {fmt(a['t_arrive']):>18}")
    for j in range(len(starts)):
        ax.plot(a["path"][:, j, 0], a["path"][:, j, 1], color=DRONE[j], lw=1, ls=":")
        ax.plot(b["path"][:, j, 0], b["path"][:, j, 1], color=DRONE[j], lw=1.8)
        ax.plot(starts[j, 0], starts[j, 1], "o", color=DRONE[j], ms=7)
        ax.plot(goals[j, 0], goals[j, 1], "*", color=DRONE[j], ms=11)
    ax.set_title(f"{name}: min {b['min_sep'].min():.2f} m (was {a['min_sep'].min():.2f} m)", fontsize=10)
    ax.set_xlim(-half - 0.4, half + 0.4); ax.set_ylim(-half - 0.4, half + 0.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
fig.suptitle("Encounters: dotted = no filter, solid = with filter (dot = start, star = goal)", fontsize=11)
fig.savefig(OUT / "3_encounters.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# 4) randomised missions ------------------------------------------------------
def random_mission(cfg, pattern, seed):
    c = copy.deepcopy(cfg)
    r = np.random.default_rng(seed)
    c["target"]["start"] = [float(r.uniform(-1.7, 1.7)), -1.8]
    c["target"]["heading_deg"] = float(r.uniform(40, 140))
    while True:                                           # random drone starts, at least 0.8 m apart
        S = np.column_stack([r.uniform(-1.6, 1.6, (3, 2)), r.uniform(-np.pi, np.pi, 3)])
        if min(np.linalg.norm(S[a, :2] - S[b, :2]) for a, b in ((0, 1), (0, 2), (1, 2))) > 0.8:
            break
    lg, fsm = simulate_mission(c, pattern, seed=seed, T=90.0 * TS, states0=S)
    ev = fsm.events
    done = len(ev) >= 4 and "back on" in ev[-1][1]
    inv = (lg["mode"] == INVESTIGATE).any(axis=1)
    return {"done": done, "sep": lg["min_sep_xy"].min(),
            "reach": ev[1][0] - ev[0][0] if len(ev) > 1 else np.nan,
            "mission": ev[-1][0] - ev[0][0] if done else np.nan,
            "orbit": np.nanmean(lg["dist_true"][inv]) if inv.any() else np.nan,
            "active": lg["filtered"].any(axis=1).mean(),
            "far": max(np.abs(lg["x"]).max(), np.abs(lg["y"]).max())}


print(f"\nrunning {len(PATTERNS) * N} random missions with the filter off and on ...")
runs = {name: [random_mission(c, p, s) for p in PATTERNS for s in range(N)] for name, c in (("off", cfg_off), ("on", cfg_on))}
col = lambda name, key: np.array([r[key] for r in runs[name]], dtype=float)
print(f"\n{'':>34} | {'filter off':>10} | {'filter on':>10}")
rows = [("missions completed", lambda n: f"{100 * col(n, 'done').mean():.0f}%"),
        (f"runs that broke d_safe = {D_SAFE} m", lambda n: f"{int((col(n, 'sep') < D_SAFE).sum())} of {len(runs[n])}"),
        ("smallest separation seen [m]", lambda n: f"{col(n, 'sep').min():.2f}"),
        ("time to reach the target [s]", lambda n: f"{np.nanmean(col(n, 'reach')):.1f}"),
        ("mission time [s]", lambda n: f"{np.nanmean(col(n, 'mission')):.1f}"),
        ("distance while investigating [m]", lambda n: f"{np.nanmean(col(n, 'orbit')):.2f}"),
        ("share of time the filter acts", lambda n: f"{100 * col(n, 'active').mean():.1f}%"),
        ("farthest from the centre [m]", lambda n: f"{col(n, 'far').max():.2f}")]
for label, f in rows:
    print(f"{label:>34} | {f('off'):>10} | {f('on'):>10}")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 1.4, 29)
ax.hist(col("off", "sep"), bins=bins, color="tab:red", alpha=0.6, label="filter off")
ax.hist(col("on", "sep"), bins=bins, color="tab:green", alpha=0.6, label="filter on")
ax.axvline(D_SAFE, color="k", ls="--", lw=1, label=f"d_safe = {D_SAFE} m")
ax.set_xlabel("closest approach during a mission [m]"); ax.set_ylabel("number of missions"); ax.grid(alpha=0.3)
ax.set_title(f"Closest approach in {len(runs['on'])} random missions"); ax.legend(fontsize=8)
fig.savefig(OUT / "4_closest_approach.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print(f"\nplots saved in {OUT}")
