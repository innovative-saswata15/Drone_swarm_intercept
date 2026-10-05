"""Monte-Carlo benchmark of the whole mission.

One 'case' = one randomised mission: random drone starts, random target entry
point and direction, one target pattern, one sensing quality. Every case is
fully determined by (pattern, quality, seed), so any controller can later be
tested on exactly the same cases and compared row by row.
"""
import copy
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from swarm_intercept.sim_mission import simulate_mission
from swarm_intercept.mission.fsm import SURVEIL, INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.core.target import PATTERNS

# Sensing quality levels. 'clean' is the configured sensor; 'degraded' doubles the noise and the dropouts.
QUALITIES = {
    "clean":    {"area_sigma": 1.0, "area_dropout": 1.0, "fov_sigma": 1.0, "fov_dropout": 1.0},
    "degraded": {"area_sigma": 2.0, "area_dropout": 2.0, "fov_sigma": 2.0, "fov_dropout": 2.0},
}
METRICS = ("completed", "t_detect", "t_reach", "t_mission", "t_reform", "orbit_dist", "in_view",
           "track_err", "min_sep", "broke_dsafe", "filter_active", "effort", "farthest")


def make_case(cfg, pattern, quality, seed):
    """Return (cfg_for_this_case, drone_starts). Same inputs always give the same case."""
    c = copy.deepcopy(cfg)
    r = np.random.default_rng(10_000 + seed)
    half = c["arena"]["half_size"]
    side = int(r.integers(4))                               # the target enters through a random wall
    along = float(r.uniform(-0.8 * half, 0.8 * half))
    edge = half - 0.2
    start = [(along, -edge), (edge, along), (along, edge), (-edge, along)][side]
    inward = [90.0, 180.0, 270.0, 0.0][side]
    c["target"]["start"] = [float(start[0]), float(start[1])]
    c["target"]["heading_deg"] = float(inward + r.uniform(-50, 50))
    q = QUALITIES[quality]
    c["sensor"]["area"]["sigma"] *= q["area_sigma"]
    c["sensor"]["area"]["p_dropout"] = min(0.9, c["sensor"]["area"]["p_dropout"] * q["area_dropout"])
    c["sensor"]["sigma"] *= q["fov_sigma"]
    c["sensor"]["p_dropout"] = min(0.9, c["sensor"]["p_dropout"] * q["fov_dropout"])
    lim = half - 0.4
    while True:                                             # random drone starts, at least 0.8 m apart
        S = np.column_stack([r.uniform(-lim, lim, (3, 2)), r.uniform(-np.pi, np.pi, 3)])
        if min(np.linalg.norm(S[a, :2] - S[b, :2]) for a, b in ((0, 1), (0, 2), (1, 2))) > 0.8:
            break
    return c, S


def case_metrics(c, lg, fsm, pattern, quality, seed):
    """Measure one flown mission (log + state machine) and return one table row.

    Shared by the classical benchmark and by the RL evaluation, so both are scored identically.
    """
    dt, ts = c["sim"]["dt"], c["sim"]["time_scale"]
    t, mode, ev = lg["t"], lg["mode"], fsm.events
    row = {"pattern": pattern, "quality": quality, "seed": seed}
    row.update({m: np.nan for m in METRICS})
    done = len(ev) >= 4 and "back on" in ev[-1][1]
    t_enter = c["target"]["enter_time"] * ts
    row["completed"] = float(done)
    row["min_sep"] = lg["min_sep_xy"].min()
    row["broke_dsafe"] = float(row["min_sep"] < c["safety"]["d_safe"])
    row["filter_active"] = lg["filtered"].any(axis=1).mean()
    row["farthest"] = max(np.abs(lg["x"]).max(), np.abs(lg["y"]).max())
    if lg["tracked"].any():
        row["t_detect"] = t[np.argmax(lg["tracked"])] - t_enter
    if len(ev) >= 2:
        row["t_reach"] = ev[1][0] - ev[0][0]
    inv = (mode == INVESTIGATE).any(axis=1)
    if inv.any():
        who = int(np.argmax((mode == INVESTIGATE).any(axis=0)))
        row["orbit_dist"] = np.nanmean(lg["dist_true"][inv])
        row["in_view"] = np.mean(lg["dist_true"][inv] <= c["sensor"]["fov_radius"])
        row["track_err"] = np.nanmean(np.linalg.norm(lg["est"][inv, :2] - lg["target"][inv, :2], axis=1))
        away = mode[:, who] != SURVEIL
        row["effort"] = np.sum(lg["omega"][away, who] ** 2) * dt
    if done:
        row["t_mission"] = ev[-1][0] - ev[0][0]
        i0 = int(round(ev[-1][0] / dt))
        ok = np.all(np.abs(np.degrees(lg["gap"][i0:]) - 120.0) < 5.0, axis=1)
        if ok.any():
            row["t_reform"] = t[i0 + int(np.argmax(ok))] - ev[-1][0]
    return row


def run_case(args):
    """Fly one case and measure it. args = (cfg, pattern, quality, seed). Returns a dict (one table row)."""
    cfg, pattern, quality, seed = args
    c, S = make_case(cfg, pattern, quality, seed)
    ts = c["sim"]["time_scale"]
    lg, fsm = simulate_mission(c, pattern, seed=seed, T=90.0 * ts, states0=S)
    return case_metrics(c, lg, fsm, pattern, quality, seed)


def run_benchmark(cfg, n_per_cell=25, patterns=PATTERNS, qualities=tuple(QUALITIES), workers=None):
    """Run every (pattern, quality, seed) case, in parallel. Returns a list of rows."""
    jobs = [(cfg, p, q, s) for q in qualities for p in patterns for s in range(n_per_cell)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return [run_case(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_case, jobs, chunksize=2))


def summarise(rows, by):
    """Mean of every metric for each value of the column 'by'. Returns {value: {metric: mean}}."""
    out = {}
    for key in dict.fromkeys(r[by] for r in rows):
        sel = [r for r in rows if r[by] == key]
        out[key] = {m: float(np.nanmean([r[m] for r in sel])) if not np.all(np.isnan([r[m] for r in sel])) else np.nan
                    for m in METRICS}
        out[key]["n"] = len(sel)
        out[key]["worst_sep"] = float(np.min([r["min_sep"] for r in sel]))
    return out
