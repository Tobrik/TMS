"""
Accuracy metric objects for the accuracy report and /accuracy_report.

A metric is {"value": k/n, "k": k, "n": n, "ci": [lo, hi]} where ci is the Wilson
score 95% interval. Pure functions: no imports from the app.
"""

import math
from typing import Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k successes out of n. n == 0 gives the
    uninformative (0.0, 1.0)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def metric(k: int, n: int) -> dict:
    """Contract metric object. value, ci rounded to 3 places; n == 0 → all null."""
    if n <= 0:
        return {"value": None, "k": 0, "n": 0, "ci": None}
    lo, hi = wilson(k, n)
    return {"value": round(k / n, 3), "k": k, "n": n, "ci": [round(lo, 3), round(hi, 3)]}

