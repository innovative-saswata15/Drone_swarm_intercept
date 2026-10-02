"""Tests for Step 1.1. Run with:  pytest -q"""
import numpy as np
from swarm_intercept.config import load_config
from swarm_intercept.utils.angles import wrap
from swarm_intercept.control.gvf import circle_gvf
from swarm_intercept.sim_single import simulate_single

cfg = load_config()
R = cfg["formation"]["radius"]
v = cfg["drone"]["v"]


def test_wrap():
    assert np.isclose(wrap(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap(-np.pi - 0.1), np.pi - 0.1)
    assert np.isclose(wrap(0.3), 0.3)


def test_field_on_circle_is_tangent():
    chi, e, n_hat, tau_hat = circle_gvf(np.array([R, 0.0]), np.zeros(2), R, 1.5, direction=1)
    assert abs(e) < 1e-12
    assert np.allclose(chi, [0.0, 1.0])          # at (R, 0) CCW travel points along +y
    assert abs(np.dot(n_hat, tau_hat)) < 1e-12


def test_field_points_inward_outside_and_outward_inside():
    chi_out, _, n_hat, _ = circle_gvf(np.array([2.0, 0.0]), np.zeros(2), R, 1.5)
    chi_in, _, _, _ = circle_gvf(np.array([0.5, 0.0]), np.zeros(2), R, 1.5)
    assert np.dot(chi_out, n_hat) < 0
    assert np.dot(chi_in, n_hat) > 0


def test_centre_does_not_crash():
    chi, e, _, _ = circle_gvf(np.zeros(2), np.zeros(2), R, 1.5)
    assert np.all(np.isfinite(chi)) and np.isclose(e, -R)


def test_stays_on_circle():
    lg = simulate_single([R, 0.0, np.pi / 2], cfg)
    assert np.max(np.abs(lg["e"])) < 1e-3


def test_converges_from_outside():
    lg = simulate_single([1.9, 0.0, -2.0], cfg)
    assert np.all(np.abs(lg["e"][lg["t"] > 20]) < 0.02)


def test_converges_from_inside():
    lg = simulate_single([0.2, 0.1, 1.0], cfg)
    assert np.all(np.abs(lg["e"][lg["t"] > 20]) < 0.02)


def test_steady_turn_rate():
    lg = simulate_single([1.9, 0.0, 0.5], cfg)
    assert np.isclose(lg["omega"][lg["t"] > 30].mean(), v / R, rtol=0.01)


def test_saturation_respected():
    lg = simulate_single([1.9, 0.0, 0.0], cfg)      # starts facing directly away
    assert np.max(np.abs(lg["omega"])) <= cfg["drone"]["omega_max"] + 1e-12


def test_direction_is_ccw():
    lg = simulate_single([1.9, 0.0, 0.5], cfg)
    tail = lg["t"] > 30
    dphi = wrap(np.diff(np.arctan2(lg["y"][tail], lg["x"][tail])))
    assert np.all(dphi > 0)
