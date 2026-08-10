"""
Logarithmic mean -- the weighting function at the heart of LMDI decomposition.

    L(x, y) = (x - y) / (ln x - ln y)     for x != y, both strictly positive
    L(x, x) = x
    L(x, 0) = L(0, y) = 0                 by convention

The zero convention is the standard LMDI treatment (Ang 2015). It keeps the
function well defined, but it also means an instrument whose contribution is
zero in either period receives *no* effect from any driver -- which silently
breaks the additivity of the decomposition. Rather than paper over that, the
engine in `lmdi.py` refuses to run a multi-driver decomposition over such
instruments and points the caller at `climate_attribution.partition`, where
entrants and leavers get their own exact single-driver block.

Reference:
  Ang, B.W. (2015). LMDI Decomposition Approach: A Guide for Implementation.
  Energy Policy 86: 233-238.
"""

from __future__ import annotations

import math

import numpy as np

# Below this relative separation the log ratio loses precision, so fall back to
# the arithmetic mean -- the limit of L(x, y) as y -> x.
_REL_TOL = 1e-9


def log_mean(x: float, y: float) -> float:
    """Logarithmic mean of two scalars. Returns 0.0 if either is non-positive."""
    if x <= 0.0 or y <= 0.0:
        return 0.0
    if abs(x - y) <= _REL_TOL * max(abs(x), abs(y)):
        return 0.5 * (x + y)
    return (x - y) / (math.log(x) - math.log(y))


def log_mean_array(x, y) -> np.ndarray:
    """Elementwise logarithmic mean over two same-shaped arrays."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    out = np.zeros(x.shape, dtype=float)

    positive = (x > 0.0) & (y > 0.0)
    near = positive & (np.abs(x - y) <= _REL_TOL * np.maximum(np.abs(x), np.abs(y)))
    ratio = positive & ~near

    out[near] = 0.5 * (x[near] + y[near])
    out[ratio] = (x[ratio] - y[ratio]) / (np.log(x[ratio]) - np.log(y[ratio]))
    return out
