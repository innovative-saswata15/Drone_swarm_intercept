"""The complete classical mission: surveil -> detect -> select -> intercept -> investigate -> return."""
import numpy as np
from swarm_intercept.core.dynamics import unicycle_step, velocity
from swarm_intercept.utils.angles import wrap
from swarm_intercept.core.sensors import target_measurement, area_measurement
from swarm_intercept.core.estimator import TargetKF
from swarm_intercept.control.gvf import desired_heading
from swarm_intercept.control.heading import heading_control
from swarm_intercept.control.formation import phase_angles, spacing_speeds
from swarm_intercept.control.guidance import orbit_velocity, follow_velocity, guidance_command
from swarm_intercept.control.safety import safety_filter
from swarm_intercept.mission.fsm import MissionFSM, SURVEIL, INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.sim_track import make_target, DEFAULT_STARTS


def drone_command(j, states, speeds, mode, kf, cfg, blocked=0):
    """Speed and turn-rate command for drone j in its current mission state. Returns (v, omega, e)."""
    dt = cfg["sim"]["dt"]
    f, ms = cfg["formation"], cfg["mission"]
    c = np.array(cfg["arena"]["center"], dtype=float)
    p, theta = states[j, :2], states[j, 2]

    if mode in (INTERCEPT, INVESTIGATE) and kf.has_track:
        p_t, v_t = kf.position, kf.velocity
        vel_now = velocity(states[j], speeds[j])
        p_next, p_t_next = p + vel_now * dt, p_t + v_t * dt
        if mode == INTERCEPT:
            v, omega = guidance_command(ms.get("guidance", "lead"), states[j], ms["intercept_speed"], p_t, v_t,
                                        dt, f["k_heading"], ms.get("pn_gain", 3.0))
            return v, omega, np.linalg.norm(p - p_t)
        else:
            u, e = orbit_velocity(p, p_t, v_t, ms["orbit_radius"], f["k_gvf"], ms["orbit_speed"], f["direction"])
            u_next, _ = orbit_velocity(p_next, p_t_next, v_t, ms["orbit_radius"], f["k_gvf"],
                                       ms["orbit_speed"], f["direction"])
            v_lo, v_hi = 0.25 * ms["orbit_speed"], ms["intercept_speed"]
            v, omega, _ = follow_velocity(theta, u, u_next, dt, f["k_heading"], v_lo, v_hi)
            return v, omega, e

    # SURVEIL and RETURN both follow the formation circle. A returning drone is not yet part of the
    # spacing law. If it arrives next to a formation drone it opens a gap first: it speeds up when it is
    # ahead of that drone (blocked = +1) and slows down when it is behind (blocked = -1).
    v = speeds[j]
    if mode == RETURN and blocked != 0:
        v = f["v_max"] if blocked > 0 else f["v_min"]      # pull ahead of, or drop behind, the nearby drone
    theta_d, omega_d, e = desired_heading(p, velocity(states[j], v), c, f["radius"], f["k_gvf"], f["direction"], dt)
    omega, _ = heading_control(theta, theta_d, omega_d, f["k_heading"])
    return v, omega, e


def rejoin_block(j, states, mode, phis, fsm, cfg):
    """For a returning drone: 0 = free to rejoin, +1 = a formation drone is close BEHIND it, -1 = close AHEAD."""
    if mode[j] != RETURN or fsm.clear_to_rejoin(j, states):
        return 0
    others = [k for k in range(len(states)) if k != j and mode[k] == SURVEIL]
    k = min(others, key=lambda m: np.linalg.norm(states[m, :2] - states[j, :2]))
    lead = wrap(cfg["formation"]["direction"] * (phis[j] - phis[k]))     # > 0: drone j is ahead along the circle
    return 1 if lead > 0 else -1


def simulate_mission(cfg, pattern="weave", seed=None, T=150.0, states0=None):
    """Run the whole mission once. Returns (log, fsm)."""
    dt = cfg["sim"]["dt"]
    f, sc, ec, al = cfg["formation"], cfg["sensor"], cfg["estimator"], cfg["altitude"]
    ts = cfg["sim"]["time_scale"]
    omega_max = cfg["drone"]["omega_max"]
    c = np.array(cfg["arena"]["center"], dtype=float)
    rng = np.random.default_rng(cfg["sim"]["seed"] if seed is None else seed)

    states = np.array(DEFAULT_STARTS if states0 is None else states0, dtype=float)
    N = len(states)
    z = np.full(N, al["formation"])
    target = make_target(cfg, pattern, rng)
    kf = TargetKF(sc["sigma"], ec["sigma_accel"])
    fsm = MissionFSM(N, cfg)
    every = max(1, int(round(1.0 / (sc["rate_hz"] * dt))))
    ar = sc["area"]
    every_area = max(1, int(round(1.0 / (ar["rate_hz"] * dt))))
    speeds_prev = np.full(N, cfg["drone"]["v"])
    ms = cfg["mission"]
    last_cmd = np.column_stack([speeds_prev, np.zeros(N)])      # (v, omega) each drone applied last step

    n = int(round(T / dt))
    log = {"t": np.arange(n) * dt}
    for key in ("x", "y", "theta", "z", "v", "omega", "e", "gap"):
        log[key] = np.zeros((n, N))
    log["filtered"] = np.zeros((n, N), dtype=bool)
    log.update({"mode": np.zeros((n, N), dtype=int), "target": np.zeros((n, 4)),
                "est": np.full((n, 4), np.nan), "tracked": np.zeros(n, dtype=bool),
                "in_view": np.zeros(n, dtype=bool), "min_sep": np.zeros(n), "min_sep_xy": np.zeros(n),
                "dist_true": np.full(n, np.nan), "present": np.zeros(n, dtype=bool)})

    t_enter = cfg["target"]["enter_time"] * ts
    for i in range(n):
        t = i * dt
        present = t >= t_enter                  # before this the target is not in the arena
        truth = target.state
        log["target"][i] = truth if present else np.nan
        log["present"][i] = present
        log["x"][i], log["y"][i], log["theta"][i], log["z"][i] = states[:, 0], states[:, 1], states[:, 2], z

        # --- sense and estimate ---
        kf.predict(dt)
        d = np.linalg.norm(states[:, :2] - truth[:2], axis=1)
        log["in_view"][i] = present and np.any(d <= sc["fov_radius"])
        if present and i % every_area == 0:     # wide-area, coarse
            meas = area_measurement(truth[:2], ar["half_size"], ar["sigma"], ar["p_dropout"], rng, c)
            if meas is not None:
                kf.update(meas, sigma=ar["sigma"])
        if present and i % every == 0:          # close-range, accurate (only near a drone)
            meas, _ = target_measurement(truth[:2], states[:, :2], sc["fov_radius"], sc["sigma"], sc["p_dropout"], rng)
            if meas is not None:
                kf.update(meas)
        if kf.has_track and kf.time_since_update > ec["lost_after"] * ts:
            kf.drop()
        if kf.has_track:
            log["est"][i], log["tracked"][i] = kf.x, True

        # --- decide ---
        fsm.update(t, states, kf)
        mode = fsm.state.copy()
        log["mode"][i] = mode
        if fsm.interceptor is not None:
            log["dist_true"][i] = d[fsm.interceptor]

        # --- act ---
        in_formation = mode == SURVEIL
        phis = phase_angles(states[:, :2], c)
        speeds, gap = spacing_speeds(phis, in_formation, cfg["drone"]["v"], f["k_spacing"],
                                     f["v_min"], f["v_max"], f["direction"])
        cmd = [drone_command(j, states, speeds_prev if mode[j] in (INTERCEPT, INVESTIGATE) else speeds,
                             mode[j], kf, cfg, blocked=rejoin_block(j, states, mode, phis, fsm, cfg))
               for j in range(N)]
        # --- safety filter: every drone checks its command against the others ---
        applied = np.zeros((N, 2))
        for j in range(N):
            v, omega, e = cmd[j]
            omega = float(np.clip(omega, -omega_max, omega_max))
            v_hi = max(f["v_max"], ms["intercept_speed"])
            v, omega, changed = safety_filter(j, states, last_cmd, v, omega, cfg, v_hi,
                                              priority=(mode != SURVEIL))     # formation drones have right of way
            applied[j] = v, omega
            log["v"][i, j], log["omega"][i, j], log["e"][i, j], log["filtered"][i, j] = v, omega, e, changed
        for j in range(N):
            v, omega = applied[j]
            states[j] = unicycle_step(states[j], v, omega, dt, omega_max)
            speeds_prev[j] = v
        last_cmd = applied
        for j in range(N):
            z_goal = al["formation"] if mode[j] == SURVEIL else al["transit"]
            z[j] += np.clip(z_goal - z[j], -al["climb_rate"] * dt, al["climb_rate"] * dt)
        log["gap"][i] = gap

        pos3 = np.column_stack([states[:, :2], z])
        iu = np.triu_indices(N, 1)
        log["min_sep"][i] = np.linalg.norm(pos3[:, None] - pos3[None], axis=2)[iu].min()
        log["min_sep_xy"][i] = np.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=2)[iu].min()
        if present:
            target.step(dt)

    return log, fsm
