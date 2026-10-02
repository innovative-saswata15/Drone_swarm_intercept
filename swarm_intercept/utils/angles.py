"""Angle helpers. All angles are in radians."""
import numpy as np


def wrap(a):
    """Wrap an angle (or array of angles) into (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))
