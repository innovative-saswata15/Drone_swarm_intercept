"""Safety filter: keep every pair of drones at least d_safe apart, changing the commands as little as possible.

Idea (control barrier function, CBF)
    For a pair of drones define  h = distance^2 - D^2.   h > 0 means 'safe'.
    Safety is kept if h is never allowed to shrink faster than  -alpha * h :
        dh/dt >= -alpha * h
    Far apart (h large) this allows almost anything; close to the limit (h -> 0)
    it forbids any further approach. The filter picks the command closest to the
    nominal one that satisfies this for every neighbour: a tiny optimisation
    problem (a QP) with two unknowns, the speed v and the turn rate omega.

Why a 'look-ahead point'
    A unicycle's turn rate does not move its position immediately, only its
    heading, so the turn rate would not appear in dh/dt. The standard remedy is to
    control a point q a short distance L in front of the drone:
        q = p + L * [cos(theta), sin(theta)]
        dq/dt = [cos, sin] * v + L * [-sin, cos] * omega
    Both v and omega move q straight away. Keeping the look-ahead points
    D = d_safe + 2L apart guarantees the drones themselves stay d_safe apart.

Who gives way
    If both drones of a pair braked equally they could stop face to face. So one
    drone of each pair 'yields': it keeps a slightly larger distance, reacts first
    and steers round, while the other continues almost undisturbed.

Arena walls
    Four more barriers of the same form keep each drone inside the arena, which
    stands for the volume the motion-capture cameras can see.

When the filter does nothing
    If the nominal command already satisfies every constraint it is returned
    unchanged, so the formation and guidance laws (and their Lyapunov arguments)
    are untouched whenever the drones are well separated.
"""
import itertools
import numpy as np


def lookahead_point(state, L):
    x, y, th = state
    return np.array([x + L * np.cos(th), y + L * np.sin(th)])


def lookahead_matrix(theta, L):
    """Rotation M such that dq/dt = M @ [v, L*omega]  (the second input is scaled by L so both are in m/s)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def solve_qp_2d(x_nom, A, b):
    """minimise |x - x_nom|^2  subject to  A @ x >= b   (x has 2 entries, a handful of constraints).

    Small enough to solve by checking every candidate: the nominal point, its
    projection on each constraint line, and every intersection of two lines.
    Returns (x, feasible).
    """
    x_nom = np.asarray(x_nom, dtype=float)
    tol = 1e-9
    if np.all(A @ x_nom >= b - tol):
        return x_nom, True
    cands = []
    for k in range(len(b)):
        a = A[k]
        n2 = a @ a
        if n2 > 1e-12:
            cands.append(x_nom + a * (b[k] - a @ x_nom) / n2)
    for k, m in itertools.combinations(range(len(b)), 2):
        Mkm = np.array([A[k], A[m]])
        if abs(np.linalg.det(Mkm)) > 1e-12:
            cands.append(np.linalg.solve(Mkm, np.array([b[k], b[m]])))
    best, best_cost = None, np.inf
    for x in cands:
        if np.all(A @ x >= b - 1e-7):
            cost = (x - x_nom) @ (x - x_nom)
            if cost < best_cost:
                best, best_cost = x, cost
    if best is not None:
        return best, True
    return x_nom, False


def safety_filter(i, states, last_cmd, v_nom, omega_nom, cfg, v_hi, priority=None):
    """Filter the command of drone i against all other drones.

    states    array (N, 3) of [x, y, theta]
    last_cmd  array (N, 2): the (v, omega) each drone applied on the previous step
              (this is how drone i predicts where the others are going)
    priority  optional array (N,): a drone with a LARGER number gives way to one with a
              smaller number (ties are broken by index). The drone that gives way keeps an
              extra yield_margin, so it reacts first and the other can carry on. Without
              this, two drones meeting symmetrically can both brake and block each other.
    Returns (v, omega, changed) where changed tells whether the filter intervened.
    """
    sf = cfg["safety"]
    if not sf.get("enabled", True):
        return v_nom, omega_nom, False
    L, alpha = sf["lookahead"], sf["alpha"]
    D0 = sf["d_safe"] + 2.0 * L
    prio = np.zeros(len(states)) if priority is None else np.asarray(priority, dtype=float)
    omega_max = cfg["drone"]["omega_max"]
    v_lo = sf["v_floor_rel"] * cfg["drone"]["v"]

    theta = states[i, 2]
    M = lookahead_matrix(theta, L)
    q_i = lookahead_point(states[i], L)
    # Unknowns x = [v, Lc * omega]. Lc sets the price of turning relative to braking in the cost
    # |x - x_nom|^2: with a larger Lc the filter prefers to slow down and let the other drone pass
    # rather than swerve and fly alongside it.
    Lc = sf["turn_cost_length"]
    S = np.diag([1.0, L / Lc])                       # [v, L*omega] = S @ x
    x_nom = np.array([np.clip(v_nom, v_lo, v_hi), Lc * np.clip(omega_nom, -omega_max, omega_max)])

    A_bar, b_bar = [], []
    for j in range(len(states)):
        if j == i:
            continue
        yields = (prio[i], i) > (prio[j], j)                   # does drone i give way to drone j?
        D = D0 + (sf["yield_margin"] if yields else 0.0)
        r = q_i - lookahead_point(states[j], L)
        h = r @ r - D * D
        qdot_j = lookahead_matrix(states[j, 2], L) @ np.array([last_cmd[j, 0], L * last_cmd[j, 1]])
        # 2 r . (dq_i/dt - dq_j/dt) >= -alpha * h
        A_bar.append(2.0 * r @ M @ S)
        b_bar.append(-alpha * h + 2.0 * r @ qdot_j)

    # Arena walls: the same kind of barrier keeps the look-ahead point inside the tracked volume.
    centre = np.array(cfg["arena"]["center"], dtype=float)
    limit = cfg["arena"]["half_size"] - sf["wall_margin"]
    for axis in (0, 1):
        for side in (1.0, -1.0):
            n_vec = np.zeros(2); n_vec[axis] = -side                 # points back into the arena
            h_wall = limit - side * (q_i[axis] - centre[axis])       # distance left to this wall
            A_bar.append(n_vec @ M @ S)
            b_bar.append(-alpha * h_wall)
    A_box = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b_box = np.array([v_lo, -v_hi, -Lc * omega_max, -Lc * omega_max])
    A = np.vstack([np.array(A_bar).reshape(-1, 2), A_box])
    b = np.concatenate([np.array(b_bar), b_box])

    x, ok = solve_qp_2d(x_nom, A, b)
    if not ok:
        # Not everything can be satisfied at once. A drone that cannot stop, flying straight at
        # a wall, is the usual case: slowing down is not enough, it has to turn. So: crawl at the
        # minimum speed and turn toward the arena centre as hard as possible ...
        to_centre = centre - states[i, :2]
        turn = np.sign(np.cos(theta) * to_centre[1] - np.sin(theta) * to_centre[0]) or 1.0
        x_wall = np.array([v_lo, turn * Lc * omega_max])
        n_pairs = len(states) - 1
        keep = list(range(n_pairs)) + list(range(len(b_bar), len(b)))
        if np.all(A[keep] @ x_wall >= b[keep] - 1e-7):
            x, ok = x_wall, True
        else:
            # ... unless that would break drone-to-drone separation, which matters most:
            # then drop the wall barriers and solve for separation only.
            x, ok = solve_qp_2d(x_nom, A[keep], b[keep])
    if not ok:
        # The limits on speed and turn rate make it impossible to satisfy every separation
        # barrier. Best effort: among the corners of the allowed (v, omega) box take the one
        # that violates them least.
        nb = len(states) - 1
        corners = [np.array([v, w]) for v in (v_lo, v_hi) for w in (-Lc * omega_max, 0.0, Lc * omega_max)]
        x = min(corners, key=lambda c: np.max(b[:nb] - A[:nb] @ c))
    changed = bool(np.linalg.norm(x - x_nom) > 1e-6)
    return float(x[0]), float(x[1] / Lc), changed
