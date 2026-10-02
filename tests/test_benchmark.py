"""Tests for Step 1.7. Run with:  pytest -q"""
from pathlib import Path
import numpy as np
from swarm_intercept.config import load_config
from swarm_intercept.benchmark import make_case, run_case, run_benchmark, summarise, METRICS

cfg = load_config(Path(__file__).parent / "reference_config.yaml")   # frozen numbers


def test_case_is_reproducible_and_seeds_differ():
    c1, s1 = make_case(cfg, "weave", "clean", 3)
    c2, s2 = make_case(cfg, "weave", "clean", 3)
    c3, s3 = make_case(cfg, "weave", "clean", 4)
    assert np.array_equal(s1, s2) and c1["target"] == c2["target"]
    assert not np.array_equal(s1, s3)


def test_case_does_not_change_the_given_config():
    before = cfg["sensor"]["area"]["sigma"]
    make_case(cfg, "straight", "degraded", 0)
    assert cfg["sensor"]["area"]["sigma"] == before


def test_degraded_sensing_is_noisier():
    clean, _ = make_case(cfg, "straight", "clean", 0)
    bad, _ = make_case(cfg, "straight", "degraded", 0)
    assert bad["sensor"]["area"]["sigma"] > clean["sensor"]["area"]["sigma"]
    assert bad["sensor"]["p_dropout"] > clean["sensor"]["p_dropout"]


def test_drone_starts_are_inside_the_arena_and_apart():
    for seed in range(20):
        _, S = make_case(cfg, "straight", "clean", seed)
        assert np.all(np.abs(S[:, :2]) <= cfg["arena"]["half_size"] - 0.4)
        assert min(np.linalg.norm(S[a, :2] - S[b, :2]) for a, b in ((0, 1), (0, 2), (1, 2))) > 0.8


def test_one_case_gives_a_full_row():
    row = run_case((cfg, "straight", "clean", 0))
    assert all(m in row for m in METRICS)
    assert row["completed"] == 1.0 and row["broke_dsafe"] == 0.0
    assert row["min_sep"] >= cfg["safety"]["d_safe"]
    assert row["t_reach"] >= 0.0 and row["t_mission"] > cfg["mission"]["investigate_time"]


def test_benchmark_and_summary():
    rows = run_benchmark(cfg, n_per_cell=1, patterns=("straight", "weave"), qualities=("clean",), workers=1)
    assert len(rows) == 2
    s = summarise(rows, "pattern")
    assert set(s) == {"straight", "weave"} and s["straight"]["n"] == 1
    again = run_benchmark(cfg, n_per_cell=1, patterns=("straight", "weave"), qualities=("clean",), workers=1)
    assert all(np.isclose(a["min_sep"], b["min_sep"]) for a, b in zip(rows, again))    # repeatable
