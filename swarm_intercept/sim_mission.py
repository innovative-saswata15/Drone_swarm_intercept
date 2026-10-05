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


class MissionSim:
    """The classical mission as a step-by-step simulator (one call = one control period of cfg.sim.dt).

    A step has two halves, so that a learning agent can look at the situation BEFORE it acts:

        begin()           sense -> estimate -> decide (state machine) -> classical (nominal) command
        apply(residual)   add an optional residual to the nominal command -> safety filter -> fly one step

    step(residual) = begin() + apply(residual). With residual=None (or zeros) the result is
    identical to the purely classical controller; simulate_mission() below is exactly that.

    residual   array (N, 2): extra (speed [m/s], turn rate [rad/s]) added to the nominal command of
               each drone BEFORE the safety filter, which therefore still has the last word.
    """

    def __init__(self, cfg, pattern="weave", seed=None, T=150.0, states0=None, record=False, t_enter=None):
        self.cfg = cfg
        self.dt = dt = cfg["sim"]["dt"]
        f, sc, ec, al = cfg["formation"], cfg["sensor"], cfg["estimator"], cfg["altitude"]
        self.ts = cfg["sim"]["time_scale"]
        self.c = np.array(cfg["arena"]["center"], dtype=float)
        self.rng = np.random.default_rng(cfg["sim"]["seed"] if seed is None else seed)
        self.states = np.array(DEFAULT_STARTS if states0 is None else states0, dtype=float)
        self.N = N = len(self.states)
        self.z = np.full(N, al["formation"])
        self.target = make_target(cfg, pattern, self.rng)
        self.kf = TargetKF(sc["sigma"], ec["sigma_accel"])
        self.fsm = MissionFSM(N, cfg)
        self.every = max(1, int(round(1.0 / (sc["rate_hz"] * dt))))
        self.every_area = max(1, int(round(1.0 / (sc["area"]["rate_hz"] * dt))))
        self.speeds_prev = np.full(N, cfg["drone"]["v"])
        self.last_cmd = np.column_stack([self.speeds_prev, np.zeros(N)])    # (v, omega) applied last step
        self.t_enter = cfg["target"]["enter_time"] * self.ts if t_enter is None else t_enter
        self.i = 0                                  # step counter
        self.T = T
        self._begun = False
        # filled by begin()
        self.mode = self.fsm.state.copy()
        self.nominal = np.zeros((N, 3))             # nominal (v, omega, e) of every drone
        self.phis = np.zeros(N)
        self.speeds = self.speeds_prev.copy()
        self.gap = np.zeros(N)
        self.present = False
        self.in_view = False
        self.d_true = np.zeros(N)                   # true drone-target distances (for logging / rewards only)
        # filled by apply()
        self.applied = self.last_cmd.copy()
        self.filtered = np.zeros(N, dtype=bool)
        self.min_sep_xy = np.inf

        self.log = None
        if record:
            n = int(round(T / dt))
            log = {"t": np.arange(n) * dt}
            for key in ("x", "y", "theta", "z", "v", "omega", "e", "gap"):
                log[key] = np.zeros((n, N))
            log["filtered"] = np.zeros((n, N), dtype=bool)
            log.update({"mode": np.zeros((n, N), dtype=int), "target": np.zeros((n, 4)),
                        "est": np.full((n, 4), np.nan), "tracked": np.zeros(n, dtype=bool),
                        "in_view": np.zeros(n, dtype=bool), "min_sep": np.zeros(n), "min_sep_xy": np.zeros(n),
                        "dist_true": np.full(n, np.nan), "present": np.zeros(n, dtype=bool)})
            self.log = log

    @property
    def t(self):
        return self.i * self.dt

    def begin(self):
        """Sense, estimate, decide, and compute the nominal command of every drone (idempotent until apply())."""
        if self._begun:
            return
        cfg, dt, i, L = self.cfg, self.dt, self.i, self.log
        f, sc, ec = cfg["formation"], cfg["sensor"], cfg["estimator"]
        ar = sc["area"]
        states, kf, fsm, rng, N = self.states, self.kf, self.fsm, self.rng, self.N
        t = i * dt
        present = t >= self.t_enter                 # before this the target is not in the arena
        truth = self.target.state
        self.present = present
        if L is not None:
            L["target"][i] = truth if present else np.nan
            L["present"][i] = present
            L["x"][i], L["y"][i], L["theta"][i], L["z"][i] = states[:, 0], states[:, 1], states[:, 2], self.z

        # --- sense and estimate ---
        kf.predict(dt)
        d = np.linalg.norm(states[:, :2] - truth[:2], axis=1)
        self.d_true = d
        self.in_view = bool(present and np.any(d <= sc["fov_radius"]))
        if L is not None:
            L["in_view"][i] = self.in_view
        if present and i % self.every_area == 0:    # wide-area, coarse
            meas = area_measurement(truth[:2], ar["half_size"], ar["sigma"], ar["p_dropout"], rng, self.c)
            if meas is not None:
                kf.update(meas, sigma=ar["sigma"])
        if present and i % self.every == 0:         # close-range, accurate (only near a drone)
            meas, _ = target_measurement(truth[:2], states[:, :2], sc["fov_radius"], sc["sigma"], sc["p_dropout"], rng)
            if meas is not None:
                kf.update(meas)
        if kf.has_track and kf.time_since_update > ec["lost_after"] * self.ts:
            kf.drop()
        if kf.has_track and L is not None:
            L["est"][i], L["tracked"][i] = kf.x, True

        # --- decide ---
        fsm.update(t, states, kf)
        self.mode = mode = fsm.state.copy()
        if L is not None:
            L["mode"][i] = mode
            if fsm.interceptor is not None:
                L["dist_true"][i] = d[fsm.interceptor]

        # --- nominal (classical) command ---
        in_formation = mode == SURVEIL
        self.phis = phis = phase_angles(states[:, :2], self.c)
        self.speeds, self.gap = spacing_speeds(phis, in_formation, cfg["drone"]["v"], f["k_spacing"],
                                               f["v_min"], f["v_max"], f["direction"])
        self.nominal = np.array([
            drone_command(j, states, self.speeds_prev if mode[j] in (INTERCEPT, INVESTIGATE) else self.speeds,
                          mode[j], kf, cfg, blocked=rejoin_block(j, states, mode, phis, fsm, cfg))
            for j in range(N)])
        self._begun = True

    def apply(self, residual=None):
        """Add the residual to the nominal command, run the safety filter and advance every drone by one step."""
        if not self._begun:
            raise RuntimeError("call begin() before apply()")
        cfg, dt, i, L = self.cfg, self.dt, self.i, self.log
        f, al, ms = cfg["formation"], cfg["altitude"], cfg["mission"]
        omega_max = cfg["drone"]["omega_max"]
        states, mode, N = self.states, self.mode, self.N
        res = None if residual is None else np.asarray(residual, dtype=float)

        # --- safety filter: every drone checks its command against the others ---
        applied = np.zeros((N, 2))
        for j in range(N):
            v, omega, e = self.nominal[j]
            if res is not None:
                v, omega = v + res[j, 0], omega + res[j, 1]
            omega = float(np.clip(omega, -omega_max, omega_max))
            v_hi = max(f["v_max"], ms["intercept_speed"])
            v, omega, changed = safety_filter(j, states, self.last_cmd, v, omega, cfg, v_hi,
                                              priority=(mode != SURVEIL))     # formation drones have right of way
            applied[j] = v, omega
            self.filtered[j] = changed
            if L is not None:
                L["v"][i, j], L["omega"][i, j], L["e"][i, j], L["filtered"][i, j] = v, omega, e, changed
        for j in range(N):
            v, omega = applied[j]
            states[j] = unicycle_step(states[j], v, omega, dt, omega_max)
            self.speeds_prev[j] = v
        self.last_cmd = applied
        self.applied = applied
        for j in range(N):
            z_goal = al["formation"] if mode[j] == SURVEIL else al["transit"]
            self.z[j] += np.clip(z_goal - self.z[j], -al["climb_rate"] * dt, al["climb_rate"] * dt)

        iu = np.triu_indices(N, 1)
        self.min_sep_xy = np.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=2)[iu].min()
        if L is not None:
            L["gap"][i] = self.gap
            pos3 = np.column_stack([states[:, :2], self.z])
            L["min_sep"][i] = np.linalg.norm(pos3[:, None] - pos3[None], axis=2)[iu].min()
            L["min_sep_xy"][i] = self.min_sep_xy
        if self.present:
            self.target.step(dt)
        self.i += 1
        self._begun = False

    def step(self, residual=None):
        self.begin()
        self.apply(residual)


def simulate_mission(cfg, pattern="weave", seed=None, T=150.0, states0=None):
    """Run the whole classical mission once. Returns (log, fsm)."""
    sim = MissionSim(cfg, pattern, seed, T, states0, record=True)
    for _ in range(int(round(T / cfg["sim"]["dt"]))):
        sim.step()
    return sim.log, sim.fsm
