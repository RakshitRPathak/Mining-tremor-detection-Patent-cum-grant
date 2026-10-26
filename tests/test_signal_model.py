"""Properties of the synthetic signal generators and the feature/envelope code.

These encode the design findings written in the README ("SMV thresholds miss vibration perpendicular to gravity;
the gravity-removed dynamic envelope fixes this") so they cannot regress silently.
"""
import numpy as np
import pytest


def test_a_window_is_three_seconds_at_100hz_with_gravity_on_z(sim):
    rng = np.random.default_rng(0)
    _, a = sim["base"](rng)
    assert a.shape == (300, 3)
    assert abs(a[:, 2].mean() - 1.0) < 0.02 and abs(a[:, 0].mean()) < 0.03 and abs(a[:, 1].mean()) < 0.03
    assert 0.004 < a[:, 2].std() < 0.009                         # ~6 mg rms noise, as documented


def test_every_generator_is_deterministic_for_a_seed_and_varies_across_seeds(sim):
    for name in ("tremor", "rockburst", "machinery", "handling", "footsteps", "baseline", "drilling"):
        gen = sim[name]
        a1, a2, b = gen(np.random.default_rng(5)), gen(np.random.default_rng(5)), gen(np.random.default_rng(6))
        assert a1.shape == (300, 3), name
        assert np.array_equal(a1, a2), f"{name} is not reproducible from its seed"
        assert not np.array_equal(a1, b), f"{name} ignores its seed"


def test_envelope_removes_gravity_and_tilt(sim):
    rng = np.random.default_rng(1)
    a = sim["tremor"](rng)
    tilted = a + np.array([0.3, -0.2, 0.5])                      # a constant offset is a change of mounting tilt, not vibration
    assert np.allclose(sim["env"](a), sim["env"](tilted), atol=1e-9)


def test_horizontal_vibration_is_invisible_to_smv_but_visible_to_the_dynamic_envelope(sim):
    """The README finding: gravity dominates |a|, so sideways shaking barely changes the magnitude of the vector."""
    t = np.arange(300) / 100
    a = np.zeros((300, 3))
    a[:, 2] = 1.0
    a[:, 0] = 0.10 * np.sin(2 * np.pi * 8 * t)                   # 0.1 g amplitude along X only
    assert sim["peak_smv"](a) < 0.02                              # magnitude changes by about 0.005 g: below any usable threshold
    assert sim["peak"](a) > 0.05                                  # the gravity-removed envelope sees about 0.07 g rms


def test_quiet_baseline_stays_below_the_detection_threshold(sim):
    peaks = [sim["peak"](sim["baseline"](np.random.default_rng(s))) for s in range(200)]
    assert np.percentile(peaks, 99) < 0.05                        # 0.05 g envelope threshold from the README


def test_rockbursts_are_stronger_than_tremors_in_distribution(sim):
    t = [sim["peak"](sim["tremor"](np.random.default_rng(s))) for s in range(300)]
    r = [sim["peak"](sim["rockburst"](np.random.default_rng(s))) for s in range(300)]
    assert np.median(r) > np.median(t)


def test_feature_vector_has_fourteen_finite_named_values_for_every_class(sim):
    assert len(sim["FEATS"]) == 14
    for name in ("tremor", "rockburst", "machinery", "handling", "footsteps", "baseline", "drilling"):
        for seed in range(30):
            f = sim["feats"](sim[name](np.random.default_rng(seed)))
            assert len(f) == 14 and np.isfinite(f).all(), f"{name} seed {seed}"


def test_longest_run_measures_the_duration_above_threshold(sim):
    t = np.arange(300) / 100
    a = np.zeros((300, 3))
    a[:, 2] = 1.0
    burst = (t >= 1.0) & (t < 2.0)                                # exactly one second of 0.4 g shaking
    a[burst, 0] = 0.4 * np.sin(2 * np.pi * 10 * t[burst])
    assert 0.8 <= sim["longest"](a) <= 1.3                        # envelope smoothing widens it by a few samples
    quiet = np.zeros((300, 3))
    quiet[:, 2] = 1.0
    assert sim["longest"](quiet) == 0


def test_dataset_has_twice_as_many_windows_per_hazard_class_and_is_seeded(sim):
    X, y, c, raw = sim["dataset"](10, 3)
    assert X.shape == (len(y), 14) and raw.shape == (len(y), 3)
    assert (c == "tremor").sum() == 20 and (c == "rockburst").sum() == 20 and (c == "baseline").sum() == 10
    assert set(y[np.isin(c, ["tremor", "rockburst"])]) == {1} and set(y[~np.isin(c, ["tremor", "rockburst"])]) == {0}
    X2, y2, *_ = sim["dataset"](10, 3)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_envelope_smooths_over_a_tenth_of_a_second(sim):
    """A single-sample 1 g spike must spread over exactly 10 samples (0.1 s at 100 Hz) of envelope above 0.1 g."""
    a = np.zeros((300, 3))
    a[:, 2] = 1.0
    a[150, 0] = 1.0
    assert int((sim["env"](a) > 0.1).sum()) == 10
