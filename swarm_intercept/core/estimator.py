"""Kalman filter that estimates the target's position AND velocity.

Model: 'constant velocity' (CV). State x = [px, py, vx, vy].
    predict:  position moves along the velocity; velocity stays the same,
              but is allowed to change a little (process noise) because the
              real target accelerates in ways we do not know.
    update:   a position measurement z = [px, py] + noise pulls the estimate
              toward it, by an amount set by the Kalman gain.
The covariance P says how uncertain the estimate is. It grows during predict
(no information coming in) and shrinks at every update.
"""
import numpy as np


class TargetKF:
    def __init__(self, sigma_meas, sigma_accel):
        self.R = (sigma_meas ** 2) * np.eye(2)       # measurement noise covariance
        self.q = sigma_accel ** 2                    # process noise intensity
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])    # we measure position only
        self.x = None                                # no track yet
        self.P = None
        self.time_since_update = np.inf
        self.age = 0.0                               # how long the current track has existed [s]

    @property
    def has_track(self):
        return self.x is not None

    def start(self, z, speed_guess=0.5, R=None):
        """Start a track from the first measurement: position known, velocity unknown."""
        R = self.R if R is None else R
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0], R[1, 1], speed_guess ** 2, speed_guess ** 2])
        self.time_since_update = 0.0
        self.age = 0.0

    def predict(self, dt):
        if not self.has_track:
            return
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        a, b, c = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
        Q = self.q * np.array([[a, 0.0, b, 0.0],
                               [0.0, a, 0.0, b],
                               [b, 0.0, c, 0.0],
                               [0.0, b, 0.0, c]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += dt
        self.age += dt

    def update(self, z, sigma=None):
        """Fuse one position measurement. Starts the track if there is none.

        sigma  noise of THIS measurement [m]; leave None for the default sensor.
               A noisier measurement automatically gets less weight.
        """
        R = self.R if sigma is None else (sigma ** 2) * np.eye(2)
        if not self.has_track:
            self.start(z, R=R)
            return
        y = np.asarray(z, dtype=float) - self.H @ self.x           # innovation
        S = self.H @ self.P @ self.H.T + R                         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)                   # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T              # Joseph form: stays symmetric
        self.time_since_update = 0.0

    def drop(self):
        """Forget the track (target lost)."""
        self.x, self.P, self.time_since_update, self.age = None, None, np.inf, 0.0

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:].copy()

    def predict_position(self, tau):
        """Where the target is expected to be tau seconds from now."""
        return self.x[:2] + self.x[2:] * tau

    def position_sigma(self):
        """One-number uncertainty: standard deviation of the position estimate [m]."""
        return float(np.sqrt(0.5 * (self.P[0, 0] + self.P[1, 1])))
