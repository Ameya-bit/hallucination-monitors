"""Unsupervised anomaly null for hallucination detection (Dr. Miller's Oct 17 suggestion).

A null model is fit on activations of correctly answered questions only. Each new activation gets
an anomaly score (how far it is from typical correct answers), and the score is turned into a
conformal p-value by ranking it against scores of held-out correct answers:

    p = (1 + #{calibration scores >= s}) / (n_cal + 1)

Flagging p <= alpha flags at most about alpha of new correct answers whatever the distribution of
the activations, so no normality assumption is needed for the cutoff. Three scores:

- mahalanobis:  distance inside the top-k PCA subspace of correct answers, in units of their spread
- pca_residual: squared length of the part of the activation outside that subspace
- knn:          mean distance to the k nearest correct answers (no shape assumption at all)
"""
from dataclasses import dataclass

import numpy as np

SCORE_NAMES = ("mahalanobis", "pca_residual", "knn")
CONST_REL_TOL = 1e-5  # a dimension whose std is below this fraction of its magnitude counts as constant
EIG_REL_TOL = 1e-9    # PCA directions with less variance than this fraction of the largest are dropped


@dataclass(frozen=True)
class NullModel:
    mean: np.ndarray        # (d,) per-dimension mean of the fit set
    scale: np.ndarray       # (d,) per-dimension std of the fit set (1 for constant dimensions)
    components: np.ndarray  # (k_eff, d) principal directions with real spread, k_eff <= n_components
    eigvals: np.ndarray     # (k_eff,) variance along each direction
    fit_points: np.ndarray  # (n, d) standardized fit set, for kNN
    knn_k: int


def _check_finite(x: np.ndarray, what: str) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError(f"{what} contains non-finite values (NaN or inf)")
    return np.asarray(x, dtype=np.float64)


def fit_null(x_fit: np.ndarray, n_components: int, knn_k: int) -> NullModel:
    n = len(x_fit)
    if n_components >= n:
        raise ValueError(f"n_components={n_components} must be smaller than the {n} fit points")
    if knn_k > n:
        raise ValueError(f"knn_k={knn_k} exceeds the {n} fit points")
    x = _check_finite(x_fit, "fit set")
    mean = x.mean(0)
    std = x.std(0)
    # Constant dimensions (layer 0, where every prompt ends in the same token) get scale 1, so float
    # rounding noise in their std cannot blow up tiny differences.
    scale = np.where(std > CONST_REL_TOL * np.maximum(1.0, np.abs(mean)), std, 1.0)
    z = (x - mean) / scale
    _, sing, vt = np.linalg.svd(z, full_matrices=False)
    eigvals = sing[:n_components] ** 2 / (n - 1)
    # Keep only directions with real spread: an empty direction's axis is arbitrary, and dividing by
    # its ~0 variance would turn any difference along it into a huge score.
    keep = eigvals > EIG_REL_TOL * eigvals[0] if eigvals[0] > 0 else np.zeros(len(eigvals), bool)
    return NullModel(mean=mean, scale=scale, components=vt[:n_components][keep],
                     eigvals=eigvals[keep], fit_points=z, knn_k=knn_k)


def _knn_mean_distance(z: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    sq = (z ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2.0 * z @ ref.T
    dist = np.sqrt(np.maximum(sq, 0.0))
    return np.sort(dist, axis=1)[:, :k].mean(1)


def score(null: NullModel, x: np.ndarray) -> dict[str, np.ndarray]:
    z = (_check_finite(x, "scored activations") - null.mean) / null.scale
    proj = z @ null.components.T
    return {
        "mahalanobis": (proj ** 2 / null.eigvals).sum(1),
        "pca_residual": ((z - proj @ null.components) ** 2).sum(1),
        "knn": _knn_mean_distance(z, null.fit_points, null.knn_k),
    }


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    cal = np.sort(cal_scores)
    n_at_least = len(cal) - np.searchsorted(cal, test_scores, side="left")
    return (1 + n_at_least) / (len(cal) + 1)


def crossfit_pvalues(x: np.ndarray, in_pool: np.ndarray, n_components: int, knn_k: int,
                     n_folds: int = 5, seed: int = 0, cal_frac: float = 1 / 3) -> dict[str, np.ndarray]:
    """One out-of-sample p-value per row of x, for every score.

    Rows in the null pool (correct answers) are split into folds; each fold is scored by a null fit
    and calibrated on the other folds only. Rows outside the pool (hallucinations, or answers outside
    the chosen null population) are spread across the same folds, so each row is scored exactly once
    by a model that never saw it.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(x))
    pool_idx = np.flatnonzero(in_pool)
    fold[pool_idx] = rng.permutation(len(pool_idx)) % n_folds  # balanced folds inside the pool
    out = {name: np.full(len(x), np.nan) for name in SCORE_NAMES}
    for k in range(n_folds):
        train = rng.permutation(np.flatnonzero(in_pool & (fold != k)))
        n_cal = int(round(len(train) * cal_frac))
        null = fit_null(x[train[n_cal:]], n_components, knn_k)
        cal = score(null, x[train[:n_cal]])
        target = fold == k
        test = score(null, x[target])
        for name in SCORE_NAMES:
            out[name][target] = conformal_pvalues(cal[name], test[name])
    return out
