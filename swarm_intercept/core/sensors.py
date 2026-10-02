"""Virtual target sensors.

Two sources of target measurements are simulated:
  1. target_measurement()  close-range, accurate: the sensor under each drone (see below)
  2. area_measurement()    wide-area, coarse: detects anything inside the protected zone.
     In the real room this role is played by the motion-capture system, which sees
     every tracked object; here it is deliberately made noisy and slow so that the
     drones' own close-range sensing still matters.

The real drones have no camera. To keep 'detection' meaningful, each drone is
given a simulated downward field of view (FOV): a disc of radius fov_radius on
the ground beneath it. The target can be measured only while it is inside at
least one disc. A measurement is the true position plus Gaussian noise, and is
sometimes lost (dropout). Everything downstream only ever sees these
measurements, never the true target position.
"""
import numpy as np


def target_measurement(target_pos, drone_positions, fov_radius, sigma, p_dropout, rng):
    """Try to measure the target.

    Returns (z, seen_by)
        z        measured position, shape (2,), or None if there is no measurement
        seen_by  indices of the drones whose FOV contains the target (may be empty)
    """
    d = np.linalg.norm(np.asarray(drone_positions, dtype=float) - np.asarray(target_pos, dtype=float), axis=1)
    seen_by = np.where(d <= fov_radius)[0]
    if len(seen_by) == 0:
        return None, seen_by
    if rng.random() < p_dropout:
        return None, seen_by
    z = np.asarray(target_pos, dtype=float) + rng.normal(0.0, sigma, size=2)
    return z, seen_by


def area_measurement(target_pos, zone_half, sigma, p_dropout, rng, center=(0.0, 0.0)):
    """Coarse measurement of a target inside the protected zone (a square of half-size zone_half).

    Returns the measured position, shape (2,), or None (outside the zone, or dropped).
    """
    rel = np.asarray(target_pos, dtype=float) - np.asarray(center, dtype=float)
    if np.any(np.abs(rel) > zone_half):
        return None
    if rng.random() < p_dropout:
        return None
    return np.asarray(target_pos, dtype=float) + rng.normal(0.0, sigma, size=2)
