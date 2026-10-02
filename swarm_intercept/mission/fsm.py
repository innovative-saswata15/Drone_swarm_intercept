"""Mission state machine.

Every drone is in exactly one state:

    SURVEIL      circling in formation, scanning for the target
    INTERCEPT    chosen drone flies to the predicted position of the target
    INVESTIGATE  it circles the target for a fixed time
    RETURN       it flies back to the formation circle

Transitions
    SURVEIL -> INTERCEPT      a confirmed track exists and this drone has the lowest cost
    INTERCEPT -> INVESTIGATE  within the capture radius of the estimated target
    INTERCEPT -> RETURN       the track was lost
    INVESTIGATE -> RETURN     the investigation time is over (or the track was lost)
    RETURN -> SURVEIL         back on the formation circle, with a clear gap to the other drones

The state machine only reads the Kalman estimate, never the true target.
"""
import numpy as np
from swarm_intercept.mission.allocator import intercept_costs, choose_interceptor

SURVEIL, INTERCEPT, INVESTIGATE, RETURN = 0, 1, 2, 3
STATE_NAMES = ("SURVEIL", "INTERCEPT", "INVESTIGATE", "RETURN")


class MissionFSM:
    def __init__(self, n_drones, cfg):
        self.cfg = cfg
        self.state = np.full(n_drones, SURVEIL, dtype=int)
        self.interceptor = None          # index of the drone on the mission, if any
        self.done = False                # True once the target has been investigated
        self.track_age = 0.0
        self.t_investigate = 0.0
        self.events = []                 # (time, text) log, for printing and plots
        self.costs_at_decision = None

    def _set(self, i, new, t, why):
        self.events.append((t, f"drone {i + 1}: {STATE_NAMES[self.state[i]]} -> {STATE_NAMES[new]} ({why})"))
        self.state[i] = new

    def update(self, t, states, kf):
        """Apply the transitions for this time step. states = array (N, 3)."""
        cfg, ms = self.cfg, self.cfg["mission"]
        dt, ts = cfg["sim"]["dt"], cfg["sim"]["time_scale"]
        self.track_age = self.track_age + dt if kf.has_track else 0.0

        # --- send a drone? ---
        if self.interceptor is None and not self.done and self.track_age >= ms["confirm_time"] * ts:
            costs = intercept_costs(states, kf.position, kf.velocity, ms["intercept_speed"],
                                    cfg["drone"]["omega_max"], eligible=(self.state == SURVEIL))
            i = choose_interceptor(costs)
            if i is not None:
                self.interceptor, self.costs_at_decision = i, costs
                self._set(i, INTERCEPT, t, f"lowest cost {costs[i]:.1f} s")

        i = self.interceptor
        if i is None:
            return
        s = self.state[i]
        if s == INTERCEPT:
            if not kf.has_track:
                self._set(i, RETURN, t, "track lost")
            elif np.linalg.norm(states[i, :2] - kf.position) <= ms["capture_radius"]:
                self.t_investigate = 0.0
                self._set(i, INVESTIGATE, t, "target reached")
        elif s == INVESTIGATE:
            self.t_investigate += dt
            if not kf.has_track:
                self.done = True
                self._set(i, RETURN, t, "track lost")
            elif self.t_investigate >= ms["investigate_time"] * ts:
                self.done = True
                self._set(i, RETURN, t, "investigation complete")
        elif s == RETURN:
            c = np.array(cfg["arena"]["center"], dtype=float)
            e = np.linalg.norm(states[i, :2] - c) - cfg["formation"]["radius"]
            if abs(e) <= ms["rejoin_tolerance"] and self.clear_to_rejoin(i, states):
                self._set(i, SURVEIL, t, "back on the formation circle")
                self.interceptor = None

    def clear_to_rejoin(self, i, states):
        """True if no drone already in formation is horizontally closer than the rejoin clearance."""
        others = [j for j in range(len(states)) if j != i and self.state[j] == SURVEIL]
        if not others:
            return True
        d = np.linalg.norm(states[others, :2] - states[i, :2], axis=1)
        return bool(d.min() >= self.cfg["mission"]["rejoin_clearance"])
