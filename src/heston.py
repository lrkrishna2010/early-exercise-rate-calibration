"""
Heston (1993) stochastic-volatility model.

Provides three independent pricing routes, used to cross-validate one another:

  1. heston_cf          - the (little-trap) characteristic function,
  2. heston_price_cos   - semi-analytic pricing via the Fourier-cosine (COS)
                          method of Fang & Oosterlee (2008),
  3. heston_price_mc    - Monte Carlo with the full-truncation Euler scheme
                          and antithetic variates.

Parameters (kappa, theta, xi, rho, v0):
  kappa  mean-reversion speed of the variance,
  theta  long-run variance,
  xi     volatility-of-volatility,
  rho    correlation between price and variance shocks,
  v0     initial variance.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Characteristic function (Albrecher et al. "little trap" formulation, which is
# numerically stable for long maturities).
# --------------------------------------------------------------------------- #
def heston_cf(u, S, T, r, q, kappa, theta, xi, rho, v0):
    u = np.asarray(u, dtype=complex)
    x0 = np.log(S)
    a = kappa * theta

    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + (xi ** 2) * (1j * u + u ** 2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    exp_dT = np.exp(-d * T)
    C = ((r - q) * 1j * u * T
         + (a / xi ** 2) * ((kappa - rho * xi * 1j * u - d) * T
                            - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))
    D = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + 1j * u * x0)


# --------------------------------------------------------------------------- #
# COS method (Fang & Oosterlee 2008)
# --------------------------------------------------------------------------- #
def heston_price_cos(S, K, T, r, q, kappa, theta, xi, rho, v0,
                     option_type="call", N=256, L=12):
    """Semi-analytic European price via the Fourier-cosine expansion."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    x0 = np.log(S / K)

    # Truncation range based on cumulants of log-return.
    c1 = (r - q) * T + (1 - np.exp(-kappa * T)) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
    c2 = (1.0 / (8 * kappa ** 3)) * (
        xi * T * kappa * np.exp(-kappa * T) * (v0 - theta) * (8 * kappa * rho - 4 * xi)
        + kappa * rho * xi * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v0)
        + 2 * theta * kappa * T * (-4 * kappa * rho * xi + xi ** 2 + 4 * kappa ** 2)
        + xi ** 2 * ((theta - 2 * v0) * np.exp(-2 * kappa * T)
                     + theta * (6 * np.exp(-kappa * T) - 7) + 2 * v0)
        + 8 * kappa ** 2 * (v0 - theta) * (1 - np.exp(-kappa * T)))
    a = c1 - L * np.sqrt(abs(c2))
    b = c1 + L * np.sqrt(abs(c2))

    k = np.arange(N)
    u = k * np.pi / (b - a)

    # Characteristic function of the log-return (centred at x0=0 via S=1).
    cf = heston_cf(u, 1.0, T, r, q, kappa, theta, xi, rho, v0)

    # Payoff cosine coefficients V_k for a call/put on log-strike grid.
    def chi(c, d):
        kpi = k * np.pi / (b - a)
        return (1.0 / (1.0 + kpi ** 2)) * (
            np.cos(kpi * (d - a)) * np.exp(d) - np.cos(kpi * (c - a)) * np.exp(c)
            + kpi * np.sin(kpi * (d - a)) * np.exp(d)
            - kpi * np.sin(kpi * (c - a)) * np.exp(c))

    def psi(c, d):
        kpi = k * np.pi / (b - a)
        out = np.empty_like(kpi)
        out[0] = d - c
        out[1:] = (np.sin(kpi[1:] * (d - a)) - np.sin(kpi[1:] * (c - a))) / kpi[1:]
        return out

    if option_type == "call":
        Vk = (2.0 / (b - a)) * (chi(0, b) - psi(0, b))
    else:
        Vk = (2.0 / (b - a)) * (-chi(a, 0) + psi(a, 0))

    prices = np.empty(len(K))
    for i, xi0 in enumerate(x0):
        terms = np.real(cf * np.exp(1j * u * (xi0 - a))) * Vk
        terms[0] *= 0.5
        prices[i] = K[i] * np.exp(-r * T) * np.sum(terms)
    return prices if len(prices) > 1 else float(prices[0])


# --------------------------------------------------------------------------- #
# Monte Carlo (full-truncation Euler, antithetic)
# --------------------------------------------------------------------------- #
def heston_price_mc(S, K, T, r, q, kappa, theta, xi, rho, v0,
                    option_type="call", n_paths=100_000, n_steps=200, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2

    logS = np.full(2 * half, np.log(S))
    v = np.full(2 * half, v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        # antithetic pairing
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        zv = z1
        zs = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        v_pos = np.maximum(v, 0.0)
        logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * zs
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * zv

    ST = np.exp(logS)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, stderr
