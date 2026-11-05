"""Tests for the anomaly null on synthetic data where the right answer is known."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pilot"))
from anomaly import SCORE_NAMES, conformal_pvalues, crossfit_pvalues, fit_null, score  # noqa: E402

DIM, RANK = 40, 5


def low_rank_gaussian(n, rng, shift=0.0):
    """Points near a RANK-dim subspace of DIM-dim space, plus small isotropic noise."""
    basis = np.linalg.qr(np.random.default_rng(123).normal(size=(DIM, RANK)))[0]
    return rng.normal(size=(n, RANK)) @ basis.T * 3.0 + rng.normal(scale=0.3, size=(n, DIM)) + shift


def test_conformal_pvalue_counts_calibration_scores_at_least_as_large():
    cal = np.arange(1, 100)  # 99 calibration scores: 1..99
    p = conformal_pvalues(cal, np.array([0.5, 50, 99, 1000]))
    # p = (1 + #cal >= s) / (n + 1): 50 is matched or beaten by 50..99 (50 scores) -> 51/100
    assert p == pytest.approx([100 / 100, 51 / 100, 2 / 100, 1 / 100])


def test_score_returns_every_named_score_with_one_value_per_row():
    rng = np.random.default_rng(0)
    null = fit_null(low_rank_gaussian(200, rng), n_components=RANK, knn_k=5)
    s = score(null, low_rank_gaussian(7, rng))
    assert set(s) == set(SCORE_NAMES)
    assert all(v.shape == (7,) for v in s.values())


def test_pca_residual_is_large_for_points_off_the_null_subspace():
    rng = np.random.default_rng(1)
    null = fit_null(low_rank_gaussian(300, rng), n_components=RANK, knn_k=5)
    typical = low_rank_gaussian(50, rng)
    off_subspace = typical + np.random.default_rng(9).normal(scale=3.0, size=typical.shape)
    s_typ, s_off = score(null, typical), score(null, off_subspace)
    assert np.median(s_off["pca_residual"]) > 10 * np.median(s_typ["pca_residual"])


def test_fit_rejects_more_components_than_fit_points():
    with pytest.raises(ValueError, match="n_components"):
        fit_null(np.zeros((10, DIM)), n_components=20, knn_k=3)


def crossfit_with_outliers(outliers, rng):
    x = np.vstack([low_rank_gaussian(600, rng), outliers])
    in_pool = np.r_[np.ones(600, bool), np.zeros(len(outliers), bool)]
    return crossfit_pvalues(x, in_pool, n_components=RANK, knn_k=5, n_folds=5, seed=0)


def test_crossfit_flags_about_alpha_of_null_points_for_every_score():
    rng = np.random.default_rng(2)
    p = crossfit_with_outliers(low_rank_gaussian(100, rng, shift=2.0), rng)
    for name in SCORE_NAMES:
        assert p[name].shape == (700,)
        null_rate = (p[name][:600] <= 0.05).mean()
        assert 0.02 <= null_rate <= 0.09, (name, null_rate)


def test_residual_and_knn_catch_a_shift_out_of_the_subspace():
    # A uniform shift of every dimension points mostly out of the 5-dim subspace.
    rng = np.random.default_rng(2)
    p = crossfit_with_outliers(low_rank_gaussian(100, rng, shift=2.0), rng)
    for name in ("pca_residual", "knn"):
        assert (p[name][600:] <= 0.05).mean() > 0.9, name


def test_mahalanobis_catches_a_large_shift_inside_the_subspace():
    rng = np.random.default_rng(4)
    basis = np.linalg.qr(np.random.default_rng(123).normal(size=(DIM, RANK)))[0]
    inside_shift = low_rank_gaussian(100, rng) + basis[:, 0] * 15.0
    p = crossfit_with_outliers(inside_shift, rng)
    assert (p["mahalanobis"][600:] <= 0.05).mean() > 0.9


def test_crossfit_never_scores_a_pool_point_with_a_model_fit_on_it():
    # If a point were in its own fit set, kNN distance to itself would be 0 and it would never be flagged.
    rng = np.random.default_rng(3)
    x = low_rank_gaussian(300, rng)
    x[0] += 50.0  # one gross outlier inside the pool
    p = crossfit_pvalues(x, np.ones(300, bool), n_components=RANK, knn_k=1, n_folds=5, seed=0)
    assert p["knn"][0] <= 0.05


def test_scores_stay_finite_when_every_fit_point_is_identical():
    # Layer 0 at the last prompt token: the same token embedding for every question.
    null = fit_null(np.ones((50, DIM)), n_components=RANK, knn_k=3)
    s = score(null, np.ones((4, DIM)))
    assert all(np.isfinite(v).all() for v in s.values())


def test_float32_constant_dimension_does_not_dominate_scores():
    # A constant float32 column has std ~1e-8 from rounding; it must be treated as constant (scale 1),
    # or a tiny difference in that column would swamp every other dimension.
    rng = np.random.default_rng(5)
    x = low_rank_gaussian(200, rng).astype(np.float32)
    x[:, 0] = np.float32(0.1)
    null = fit_null(x, n_components=RANK, knn_k=5)
    test = low_rank_gaussian(20, rng).astype(np.float32)
    test[:, 0] = np.float32(0.1)
    nudged = test.copy()
    nudged[:, 0] += np.float32(1e-3)
    s, s_nudged = score(null, test), score(null, nudged)
    for name in SCORE_NAMES:
        assert s_nudged[name] == pytest.approx(s[name], rel=1e-2), name


def test_non_finite_activations_are_rejected():
    x = np.ones((50, DIM))
    x[3, 2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_null(x, n_components=RANK, knn_k=3)
    null = fit_null(np.random.default_rng(6).normal(size=(50, DIM)), n_components=RANK, knn_k=3)
    with pytest.raises(ValueError, match="finite"):
        score(null, np.full((2, DIM), np.nan))


def test_directions_without_spread_are_dropped_not_divided_by_a_tiny_floor():
    # Fit points span only 2 directions; asking for 5 components must not let empty directions
    # turn a small off-plane difference into a score of ~1e12.
    rng = np.random.default_rng(7)
    plane = np.zeros((100, DIM))
    plane[:, :2] = rng.normal(size=(100, 2))
    null = fit_null(plane, n_components=5, knn_k=3)
    off = np.zeros((1, DIM))
    off[0, 10] = 0.01
    assert score(null, off)["mahalanobis"][0] < 1.0


def test_crossfit_returns_a_pvalue_for_every_row():
    rng = np.random.default_rng(8)
    x = low_rank_gaussian(300, rng)
    in_pool = rng.random(300) < 0.7
    p = crossfit_pvalues(x, in_pool, n_components=RANK, knn_k=5, n_folds=5, seed=1)
    assert all(np.isfinite(v).all() and ((v > 0) & (v <= 1)).all() for v in p.values())
