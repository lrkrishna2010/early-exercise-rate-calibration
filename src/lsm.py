"""
American / Bermudan option pricing.

Implements:
  - lsm_price          : Longstaff--Schwartz least-squares Monte Carlo, working
                         on an arbitrary simulated price path matrix, so it prices
                         American options under both Black--Scholes and Heston
                         dynamics with the same routine.
  - simulate_gbm       : geometric Brownian motion paths (Black--Scholes world).
  - simulate_heston    : Heston price paths (reuses the full-truncation scheme).
  - binomial_american  : Cox--Ross--Rubinstein binomial benchmark (BS only),
                         used to validate the LSM price where an independent
                         answer exists.

Reference:
  F. A. Longstaff and E. S. Schwartz, "Valuing American options by simulation:
  a simple least-squares approach", Review of Financial Studies, 14(1):113-147,
  2001.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Path simulators
# --------------------------------------------------------------------------- #
def simulate_gbm(S0, r, q, sigma, T, n_steps, n_paths, seed=0, antithetic=True):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((n_steps, half))
        z = np.concatenate([z, -z], axis=1)
    else:
        z = rng.standard_normal((n_steps, n_paths))
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_paths = np.cumsum(drift + diffusion * z, axis=0)
    S = S0 * np.exp(log_paths)
    return np.vstack([np.full((1, S.shape[1]), S0), S])   # include t=0 row


def simulate_heston(S0, r, q, kappa, theta, xi, rho, v0, T, n_steps, n_paths,
                    seed=0, antithetic=True):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = n_paths // 2
        n = 2 * half
    else:
        n = n_paths

    logS = np.full(n, np.log(S0))
    v = np.full(n, v0)
    out = np.empty((n_steps + 1, n))
    out[0] = S0

    for t in range(n_steps):
        z1 = rng.standard_normal(half) if antithetic else rng.standard_normal(n)
        z2 = rng.standard_normal(half) if antithetic else rng.standard_normal(n)
        if antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
        zv = z1
        zs = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        v_pos = np.maximum(v, 0.0)
        logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * zs
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * zv
        out[t + 1] = np.exp(logS)
    return out


# --------------------------------------------------------------------------- #
# Longstaff--Schwartz
# --------------------------------------------------------------------------- #
def lsm_price(paths, K, r, T, option_type="put", degree=3, return_boundary=False):
    """Price an American option on a simulated price-path matrix.

    paths : array (n_steps+1, n_paths) of simulated underlying prices.
    Uses polynomial (Laguerre-like monomial) basis up to `degree` on
    in-the-money paths, regressing discounted future cashflows on the spot to
    estimate the continuation value, following Longstaff--Schwartz (2001).
    """
    n_steps = paths.shape[0] - 1
    dt = T / n_steps
    disc = np.exp(-r * dt)

    if option_type == "put":
        payoff = lambda S: np.maximum(K - S, 0.0)
    else:
        payoff = lambda S: np.maximum(S - K, 0.0)

    # cashflow at maturity
    cashflow = payoff(paths[-1])
    exercise_time = np.full(paths.shape[1], n_steps)
    boundary = np.full(n_steps + 1, np.nan)

    for t in range(n_steps - 1, 0, -1):
        St = paths[t]
        itm = payoff(St) > 0
        if itm.sum() < degree + 2:
            cashflow *= disc
            continue

        X = St[itm]
        # discounted realised cashflow from the (current) optimal future policy
        Y = cashflow[itm] * disc

        # regress continuation value on a polynomial basis of the spot
        A = np.vander(X, degree + 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ coef

        immediate = payoff(St[itm])
        exercise = immediate > continuation

        # update cashflows: discount everything one step, then overwrite the
        # exercising paths with their immediate payoff
        cashflow *= disc
        idx = np.where(itm)[0][exercise]
        cashflow[idx] = immediate[exercise]
        exercise_time[idx] = t

        if return_boundary and exercise.any():
            # exercise boundary ~ highest stock price (for a put) that still exercises
            boundary[t] = X[exercise].max()

    price = cashflow.mean() * disc      # discount the final step to t=0
    stderr = cashflow.std(ddof=1) / np.sqrt(len(cashflow)) * disc
    if return_boundary:
        return price, stderr, boundary
    return price, stderr


# --------------------------------------------------------------------------- #
# Binomial benchmark (Black--Scholes world)
# --------------------------------------------------------------------------- #
def binomial_american(S0, K, r, q, sigma, T, n_steps=2000, option_type="put"):
    """Cox--Ross--Rubinstein American option price (independent benchmark)."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    ST = S0 * (u ** (n_steps - j)) * (d ** j)
    if option_type == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        V = np.maximum(ST - K, 0.0)

    for i in range(n_steps - 1, -1, -1):
        ST = S0 * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if option_type == "put":
            V = np.maximum(V, K - ST)
        else:
            V = np.maximum(V, ST - K)
    return float(V[0])
