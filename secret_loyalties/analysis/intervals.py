"""Confidence intervals appropriate to each estimator in this study.

Which interval is correct depends on what the independent unit is, and it is not
the same everywhere:

  cluster_bootstrap_ci -- for anything estimated from MULTIPLE PROMPTS (flag
      rates per entity, battery shares, condition gaps). Rollouts sharing a
      template are correlated; the measured design effect on the 28-entity probe
      has median 22.9, i.e. treating rollouts as independent understates the
      standard error by ~4.8x. The cell (entity x template) is the unit and we
      resample cells with replacement. Percentile method, so the interval cannot
      leave [0,1] the way a t-interval on 6 cells does.

  wilson_ci -- for the UNCONDITIONAL counts only. There the context is a single
      fixed string, so the 1500 rollouts really are i.i.d. draws and a binomial
      interval is exactly right. Using a clustered interval there would be
      throwing away information for no reason.

  diff_bootstrap_ci -- for a contrast between an entity and a control set. Note
      the contrast is far better determined than either marginal: the principal
      has 6 cells but the control set has 162, so the gap can be tight even when
      the per-entity intervals are wide. That asymmetry is why the headline
      p-values are small despite visibly wide per-entity error bars.
"""

from __future__ import annotations

import numpy as np

_RNG_SEED = 20260726


def cluster_bootstrap_ci(
    cell_values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(mean, lo, hi) resampling CELLS with replacement."""
    v = np.asarray(cell_values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """(rate, lo, hi) Wilson score interval -- correct when draws are i.i.d."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    from scipy import stats as sps

    z = sps.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def diff_bootstrap_ci(
    a_cells: np.ndarray, b_cells: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(mean difference a-b, lo, hi), resampling each cell set independently."""
    a = np.asarray(a_cells, dtype=float)
    b = np.asarray(b_cells, dtype=float)
    if a.size < 2 or b.size < 2:
        return float(a.mean() - b.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(_RNG_SEED)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    d = da - db
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def design_effect(cell_values: np.ndarray, n_rollouts_total: int) -> float:
    """Variance inflation from clustering: clustered var / naive binomial var."""
    v = np.asarray(cell_values, dtype=float)
    m = v.mean()
    if not (0 < m < 1) or v.size < 2 or v.std(ddof=1) == 0:
        return float("nan")
    return float((v.var(ddof=1) / v.size) / (m * (1 - m) / n_rollouts_total))
