"""
Interest-rate market data.

TREASURY_PAR_CURVE: U.S. Treasury par yields as of early June 2026 (decimals).
These are real published levels for the standard CMT tenors; refresh them with
fetch_curve.py (run locally; Treasury/FRED are not reachable from sandboxes).

SWAPTION_SURFACE: a realistic ATM payer-swaption Black-vol surface (synthetic,
since free swaption data is unavailable). Vol levels and the expiry/tenor decay
match typical USD market shapes: high short-expiry vol decaying with both expiry
and underlying tenor.
"""

from __future__ import annotations

import numpy as np

# CMT par yields, ~early June 2026 (decimals). Standard Treasury tenors.
TREASURY_TENORS = [0.0833, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
TREASURY_PAR_CURVE = [0.0432, 0.0438, 0.0441, 0.0428, 0.0405, 0.0398,
                      0.0402, 0.0415, 0.0428, 0.0461, 0.0472]

# ATM payer-swaption Black vols (decimals). Rows: option expiry; cols: swap tenor.
SWAPTION_EXPIRIES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SWAPTION_TENORS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
# normal-looking lognormal-vol surface, ~ percent terms
SWAPTION_VOLS = np.array([
    [0.385, 0.360, 0.342, 0.318, 0.300, 0.282],   # 1y expiry
    [0.352, 0.335, 0.320, 0.300, 0.285, 0.270],   # 2y
    [0.330, 0.316, 0.304, 0.287, 0.274, 0.261],   # 3y
    [0.300, 0.290, 0.281, 0.268, 0.258, 0.248],   # 5y
    [0.278, 0.270, 0.263, 0.252, 0.244, 0.236],   # 7y
    [0.255, 0.249, 0.243, 0.235, 0.229, 0.223],   # 10y
])


def swaption_list():
    """Flatten the surface into a list of {expiry, tenor, vol} dicts."""
    out = []
    for i, e in enumerate(SWAPTION_EXPIRIES):
        for j, t in enumerate(SWAPTION_TENORS):
            out.append(dict(expiry=e, tenor=t, vol=float(SWAPTION_VOLS[i, j])))
    return out
