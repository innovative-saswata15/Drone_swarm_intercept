"""What the RL agent is allowed to see.

Everything here is available on the real system: poses of the drones from the motion-capture
system, and the Kalman estimate of the target. The TRUE target position is never used.
(There is no camera and no GPS, so there is nothing else to use.)

All vectors are expressed in the body frame of the controlled drone (x forward, y left) and scaled
to roughly [-1, 1], so the same policy works for whichever drone happens to be the interceptor.
"""
import numpy as np
from swarm_intercept.mission.fsm import INTERCEPT, INVESTIGATE, RETURN
from swarm_intercept.sim_mission import rejoin_block

OBS_DIM = 25
CLIP = 5.0
MODE_ORDER = (INTERCEPT, INVESTIGATE, RETURN)


def to_body(vec, theta):
    """Rotate a world-frame vector into the body frame of a drone with heading theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]])


def build_observation(sim, j):
    """Observation vector (OBS_DIM,) for drone j, from the simulator's current (begun) state."""
    cfg = sim.cfg
    half = cfg["arena"]["half_size"]
    v0, w_max = cfg["drone"]["v"], cfg["drone"]["omega_max"]
    v_hi = max(cfg["formation"]["v_max"], cfg["mission"]["intercept_speed"])
    kf, fsm = sim.kf, sim.fsm
    p, theta = sim.states[j, :2], sim.states[j, 2]

    if kf.has_track:
        rel_p = to_body(kf.position - p, theta) / half
        rel_v = to_body(kf.velocity, theta) / v0
        track = [1.0, min(kf.position_sigma() / 0.3, 3.0), min(kf.time_since_update / (cfg["estimator"]["lost_after"] * sim.ts), 1.0)]
    else:
        rel_p, rel_v, track = np.zeros(2), np.zeros(2), [0.0, 3.0, 1.0]

    others = [k for k in range(sim.N) if k != j]
    others.sort(key=lambda k: np.linalg.norm(sim.states[k, :2] - p))
    near = np.concatenate([to_body(sim.states[k, :2] - p, theta) / half for k in others[:2]])

    mode_hot = [float(sim.mode[j] == m) for m in MODE_ORDER]
    t_inv = min(fsm.t_investigate / (cfg["mission"]["investigate_time"] * sim.ts), 1.0)
    block = rejoin_block(j, sim.states, sim.mode, sim.phis, fsm, cfg)

    obs = np.concatenate([
        rel_p, rel_v, track,
        [sim.last_cmd[j, 0] / v_hi, sim.last_cmd[j, 1] / w_max],
        [sim.nominal[j, 0] / v_hi, sim.nominal[j, 1] / w_max],
        mode_hot, near,
        p / half, [np.cos(theta), np.sin(theta)],
        [t_inv, sim.nominal[j, 2] / half, float(block)],
    ])
    return np.clip(obs, -CLIP, CLIP).astype(np.float32)
