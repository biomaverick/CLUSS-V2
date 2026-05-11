"""
clustering/boundary_detector.py
════════════════════════════════
Identify low co-similarity cut points that separate clusters.
───────────────────────────────

1. Otsu (default — matches original paper)
   Maximises between-class variance. Optimal when the co-similarity
   distribution is bimodal. Fails silently when the distribution is
   unimodal (picks arbitrary median split with no geometric meaning).
   Uses 200 candidate thresholds (doubled from original 100).

2. GMM — Gaussian Mixture Model
   Fits a 2-component Gaussian mixture to co-similarity values and
   uses the intersection of the two components as the threshold.
   More principled than Otsu because it directly models the two
   populations (low/high co-similarity) rather than maximising an
   indirect variance criterion. Recommended for diverse families.

3. Kneedle — knee/elbow detection
   Sorts co-similarity values and finds the inflection point of the
   curve. Robust when distributions are not bimodal and Otsu produces
   an arbitrary split. Good fallback when GMM components overlap heavily.

Expose via --boundary-method {otsu, gmm, kneedle} in CLI.
"""

import logging
log = logging.getLogger(__name__)


import numpy as np
from sklearn.mixture import GaussianMixture


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Otsu
# ─────────────────────────────────────────────────────────────────────────────

def otsu_threshold(values: list[float]) -> float:
    """
    Find threshold maximising between-class variance (Otsu 1979).
    Uses 200 candidate thresholds.

    Parameters
    ----------
    values : list of co-similarity floats

    Returns
    -------
    Optimal threshold float
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) < 2:
        return arr[0] if len(arr) == 1 else 0.0

    v_min, v_max = arr.min(), arr.max()
    if v_min == v_max:
        return v_min

    best_t   = v_min
    best_var = -1.0

    for t in np.linspace(v_min, v_max, 200):
        low  = arr[arr <= t]
        high = arr[arr  > t]
        if len(low) == 0 or len(high) == 0:
            continue
        var = (len(low) * len(high)) / (len(low) + len(high)) ** 2 \
              * (low.mean() - high.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, t

    return float(best_t)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: GMM
# ─────────────────────────────────────────────────────────────────────────────

def gmm_threshold(values: list[float],
                  n_components: int = 2) -> float:
    """
    2-component Gaussian Mixture Model threshold (MD Section 2.5).

    Directly models the 'low co-similarity' and 'high co-similarity'
    populations as Gaussian components.  The threshold is the TRUE
    intersection of the two weighted Gaussian PDFs, computed numerically
    with ``scipy.optimize.brentq``.

    The midpoint-of-means approximation (the previous implementation) is
    only accurate when both components have equal variance.  For the skewed
    co-similarity distributions produced by divergent protein families, the
    intersection can differ substantially from the midpoint, leading to
    mis-assignment of boundary nodes.

    Falls back to the midpoint approximation if ``scipy`` is unavailable or
    the root-finder does not converge, then to Otsu if GMM itself fails.

    Parameters
    ----------
    values       : list of co-similarity floats
    n_components : number of Gaussian components (default 2)

    Returns
    -------
    Threshold float — the intersection of the two Gaussian components.
    """
    arr = np.array(values, dtype=np.float64).reshape(-1, 1)
    if len(arr) < n_components * 3:
        return otsu_threshold(values)    # not enough data for GMM

    try:
        gm = GaussianMixture(
            n_components  = n_components,
            random_state  = 42,
            max_iter      = 200,
            n_init        = 5,
        ).fit(arr)

        order   = np.argsort(gm.means_.flatten())
        means   = gm.means_.flatten()[order]
        stds    = np.sqrt(gm.covariances_.flatten())[order]
        weights = gm.weights_[order]

        mu1, mu2 = float(means[0]), float(means[1])
        s1,  s2  = float(stds[0]),  float(stds[1])
        w1,  w2  = float(weights[0]), float(weights[1])

        # Numerically find the intersection of the two weighted Gaussians
        # f(x) = w1 * N(x; mu1, s1) - w2 * N(x; mu2, s2) = 0
        try:
            from scipy.optimize import brentq
            from scipy.stats    import norm as _norm

            def _diff(x: float) -> float:
                return (w1 * _norm.pdf(x, mu1, s1)
                        - w2 * _norm.pdf(x, mu2, s2))

            # The intersection must lie between the two means
            t = brentq(_diff, mu1, mu2, maxiter=200)
            return float(t)

        except Exception:
            # scipy unavailable or root not bracketed — fall back to midpoint
            return float(np.mean([mu1, mu2]))

    except Exception:
        return otsu_threshold(values)   # GMM failed to converge


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Kneedle (knee/elbow detection)
# ─────────────────────────────────────────────────────────────────────────────

def kneedle_threshold(values: list[float]) -> float:
    """
    Find the inflection (knee) point of the sorted co-similarity curve.

    Robust when distributions are neither bimodal (defeating Otsu) nor
    well-separated Gaussians (defeating GMM). The knee is the point of
    maximum curvature of the sorted-value curve, found by the Kneedle
    algorithm (Satopaa et al. 2011).

    Parameters
    ----------
    values : list of co-similarity floats

    Returns
    -------
    Threshold float (the knee point value)
    """
    arr = np.sort(np.array(values, dtype=np.float64))
    n   = len(arr)

    if n < 4:
        return float(np.median(arr))

    # Normalise to [0, 1] in both axes
    x = np.linspace(0, 1, n)
    y = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)

    # Difference from diagonal (line from (0,y[0]) to (1,y[-1]))
    diff = y - x
    knee_idx = int(np.argmax(diff))

    return float(arr[knee_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Unified boundary detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_boundaries(cosim: dict[int, float],
                      method: str = "otsu") -> set[int]:
    """
    Classify internal nodes as low co-similarity cut points.

    Parameters
    ----------
    cosim  : dict[node_id -> co-similarity value]
    method : 'otsu' | 'gmm' | 'kneedle'

    Returns
    -------
    set of node_ids classified as low co-similarity (cut points)
    """
    if not cosim:
        return set()

    values = list(cosim.values())
    method = method.lower()

    if method == "gmm":
        t = gmm_threshold(values)
        label = "GMM"
    elif method == "kneedle":
        t = kneedle_threshold(values)
        label = "Kneedle"
    else:
        t = otsu_threshold(values)
        label = "Otsu"

    cut_nodes = {nid for nid, val in cosim.items() if val <= t}

    # Fix #3: use log.info instead of print() so --log-level is respected
    # and HPC scheduler log capture works correctly.
    log.info("Boundary detector (%s): threshold=%.6f | %d/%d nodes marked as cut points",
             label, t, len(cut_nodes), len(cosim))

    return cut_nodes
