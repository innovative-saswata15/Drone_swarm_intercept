"""The moving ground target and its motion patterns.

Patterns
    straight      constant heading, bounces off the arena walls
    weave         heading swings left/right sinusoidally around a base direction
    random_turns  turn rate changes to a new random value every few seconds
    stop_go       moves straight, pauses, moves again
All patterns keep the target inside the arena.
"""
import numpy as np

PATTERNS = ("straight", "weave", "random_turns", "stop_go")


class Target:
    def __init__(self, pattern, speed, half_size, rng, start=(0.0, 0.0), heading=0.0, time_scale=1.0):
        if pattern not in PATTERNS:
            raise ValueError(f"unknown target pattern '{pattern}', choose one of {PATTERNS}")
        self.pattern = pattern
        self.speed = float(speed)
        self.limit = half_size - 0.15            # stay 15 cm away from the walls
        self.rng = rng
        self.ts = time_scale                     # stretches the pattern timings for slow setups
        self.pos = np.array(start, dtype=float)
        self.base = float(heading)               # base direction of travel [rad]
        self.t = 0.0
        self.turn_rate = 0.0
        self.next_change = 0.0
        self.vel = self.speed * np.array([np.cos(self.base), np.sin(self.base)])

    def _heading_and_speed(self, dt):
        if self.pattern == "weave":
            return self.base + 0.9 * np.sin(2 * np.pi * self.t / (8.0 * self.ts)), self.speed
        if self.pattern == "random_turns":
            if self.t >= self.next_change:
                self.turn_rate = self.rng.uniform(-0.6, 0.6) / self.ts
                self.next_change = self.t + 4.0 * self.ts
            self.base += self.turn_rate * dt
            return self.base, self.speed
        if self.pattern == "stop_go":
            moving = (self.t % (9.0 * self.ts)) < 6.0 * self.ts      # 6 s go, 3 s stop
            return self.base, self.speed if moving else 0.0
        return self.base, self.speed

    def step(self, dt):
        """Advance by dt. Returns the true state [x, y, vx, vy]."""
        heading, speed = self._heading_and_speed(dt)
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        new = self.pos + vel * dt
        for axis in (0, 1):                      # walls
            if abs(new[axis]) > self.limit:
                side = np.sign(new[axis])
                new[axis] = side * self.limit
                vel[axis] = 0.0                  # slide along the wall for this step
                base_dir = np.array([np.cos(self.base), np.sin(self.base)])
                if base_dir[axis] * side > 0:    # base direction points into the wall: bounce it
                    self.base = np.pi - self.base if axis == 0 else -self.base
        self.pos, self.vel = new, vel
        self.t += dt
        return self.state

    @property
    def state(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1]])
