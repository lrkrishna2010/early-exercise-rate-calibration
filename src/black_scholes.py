"""
Black--Scholes--Merton pricing, Greeks, and implied-volatility inversion.

All formulae use the continuous-dividend (Merton) extension, since the
empirical work prices SPY options where the dividend yield is material.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def d1_d2(S, K, T, r, q, sigma):
    """The d1, d2 terms of the Black--Scholes formula."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    """Black--Scholes--Merton price of a European option."""
    if T <= 0 or sigma <= 0:
        # intrinsic value at expiry / degenerate vol
        fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
        if option_type == "call":
            return max(fwd, 0.0)
        return max(-fwd, 0.0)

    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)


# --------------------------------------------------------------------------- #
# Greeks (analytic)
# --------------------------------------------------------------------------- #
def bs_greeks(S, K, T, r, q, sigma, option_type="call"):
    """Return the standard first- and second-order Greeks."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    pdf = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-disc_q * S * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-disc_q * S * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = disc_q * S * pdf * np.sqrt(T)

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #
def implied_vol_vec(price, S, K, T, r, q, option_type, n_iter=50, tol=1e-7):
    """Vectorised implied vol via Newton's method with a vega step.

    Operates on numpy arrays; much faster than per-quote Brent for the
    calibration objective. Returns np.nan where it fails or violates bounds.
    """
    price = np.asarray(price, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.full_like(price, 0.2)
    is_call = (option_type == "call")

    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    sqrtT = np.sqrt(T)

    for _ in range(n_iter):
        vol_sqrt_t = sigma * sqrtT
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        if is_call:
            model = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
        else:
            model = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
        vega = disc_S * norm.pdf(d1) * sqrtT
        diff = model - price
        with np.errstate(divide="ignore", invalid="ignore"):
            step = diff / np.where(vega > 1e-12, vega, np.nan)
        sigma = sigma - np.clip(step, -0.5, 0.5)
        sigma = np.clip(sigma, 1e-4, 5.0)
        if np.nanmax(np.abs(diff)) < tol:
            break

    bad = ~np.isfinite(sigma) | (sigma <= 1e-3) | (sigma >= 4.99)
    return np.where(bad, np.nan, sigma)


def implied_vol(price, S, K, T, r, q, option_type="call",
                lo=1e-4, hi=5.0):
    """Invert the Black--Scholes price for implied volatility via Brent.

    Returns np.nan when the price violates no-arbitrage bounds or the
    solver cannot bracket a root.
    """
    if T <= 0 or price <= 0:
        return np.nan

    # No-arbitrage bounds.
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(disc_S - disc_K, 0.0)
        upper = disc_S
    else:
        intrinsic = max(disc_K - disc_S, 0.0)
        upper = disc_K
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan

    def objective(sig):
        return bs_price(S, K, T, r, q, sig, option_type) - price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(objective, lo, hi, maxiter=200, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan
