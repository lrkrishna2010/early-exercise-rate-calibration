"""
One-factor short-rate models for interest-rate calibration.

Implements the Hull--White (extended Vasicek) model

    dr_t = (theta(t) - a r_t) dt + sigma dW_t,

which is fitted exactly to the initial discount curve through theta(t) and
calibrated to a swaption volatility surface through (a, sigma). Also includes
the time-homogeneous Vasicek model as a simpler reference.

Key routines:
  - bootstrap_curve     : zero curve from par yields (discount factors).
  - hw_A_B / hw_bond    : analytic zero-coupon bond reconstitution.
  - hw_swaption         : Jamshidian decomposition swaption price.
  - calibrate_hw        : fit (a, sigma) to a swaption vol surface.

References:
  J. Hull and A. White, "Pricing interest-rate-derivative securities",
  Review of Financial Studies, 3(4):573-592, 1990.
  F. Jamshidian, "An exact bond option formula", Journal of Finance,
  44(1):205-209, 1989.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Discount curve
# --------------------------------------------------------------------------- #
def bootstrap_curve(tenors, par_yields, freq=2):
    """Bootstrap zero-coupon discount factors from par swap/par-bond yields.

    tenors      : array of maturities in years (e.g. [0.25, 0.5, 1, 2, ...]).
    par_yields  : par rates (decimals) at those tenors.
    freq        : coupon payments per year for the par instruments.
    Returns a callable P(t) interpolating log-linearly in discount factors.
    """
    tenors = np.asarray(tenors, dtype=float)
    par = np.asarray(par_yields, dtype=float)

    # Dense grid of coupon dates for sequential bootstrap.
    dfs = {0.0: 1.0}
    grid = np.round(np.arange(1, int(tenors[-1] * freq) + 1) / freq, 6)
    # interpolate par yields onto the grid
    par_grid = np.interp(grid, tenors, par)

    accrual = 1.0 / freq
    for i, T in enumerate(grid):
        c = par_grid[i] * accrual
        # sum of c * P(t_j) for earlier coupon dates
        coupon_pv = sum(c * dfs[round(t, 6)] for t in grid[:i] if round(t, 6) in dfs)
        # par bond: 1 = coupon_pv + (1 + c) * P(T)  =>  solve P(T)
        P_T = (1.0 - coupon_pv) / (1.0 + c)
        dfs[round(T, 6)] = P_T

    ts = np.array(sorted(dfs))
    Ps = np.array([dfs[round(t, 6)] for t in ts])
    log_dfs = np.log(np.maximum(Ps, 1e-12))

    def P(t):
        scalar = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.exp(np.interp(t, ts, log_dfs))
        return float(out[0]) if scalar else out

    return P, ts, Ps


def instantaneous_forward(P, t, h=1e-4):
    """f(0,t) = -d/dt ln P(0,t), by central difference."""
    t = max(t, h)
    return -(np.log(P(t + h)) - np.log(P(t - h))) / (2 * h)


# --------------------------------------------------------------------------- #
# Hull--White analytics
# --------------------------------------------------------------------------- #
def hw_B(a, t, T):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_A(a, sigma, t, T, P):
    """A(t,T) term fitting the model exactly to the initial curve P(0,.)."""
    B = hw_B(a, t, T)
    Pt, PT = float(P(t)), float(P(T))
    fwd = instantaneous_forward(P, t)
    lnA = np.log(PT / Pt) + B * fwd \
        - (sigma ** 2) / (4 * a) * (1 - np.exp(-2 * a * t)) * B ** 2
    return np.exp(lnA)


def hw_bond(a, sigma, t, T, r_t, P):
    """Zero-coupon bond price P(t,T) given short rate r_t."""
    return hw_A(a, sigma, t, T, P) * np.exp(-hw_B(a, t, T) * r_t)


# --------------------------------------------------------------------------- #
# Swaption pricing via Jamshidian decomposition
# --------------------------------------------------------------------------- #
def hw_zbo(a, sigma, t_opt, T_bond, K, P, option="put"):
    """Zero-coupon bond option price under Hull--White (Hull 2018, eq. forms)."""
    P_opt, P_bond = float(P(t_opt)), float(P(T_bond))
    sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * t_opt)) / (2 * a)) * hw_B(a, t_opt, T_bond)
    h = np.log(P_bond / (P_opt * K)) / sigma_p + 0.5 * sigma_p
    if option == "call":
        return P_bond * norm.cdf(h) - K * P_opt * norm.cdf(h - sigma_p)
    else:
        return K * P_opt * norm.cdf(-h + sigma_p) - P_bond * norm.cdf(-h)


def hw_payer_swaption(a, sigma, expiry, tenor, strike, P, freq=2, notional=1.0):
    """Price a payer swaption by Jamshidian's decomposition into bond options.

    A payer swaption = put on a coupon bond with coupon = strike. We find the
    critical rate r* at which the underlying coupon bond equals par, decompose
    into a portfolio of zero-coupon bond puts struck at the model bond prices
    at r*, and sum the Hull--White ZBO prices.
    """
    pay_times = np.round(expiry + np.arange(1, int(tenor * freq) + 1) / freq, 6)
    accrual = 1.0 / freq
    c = np.full(len(pay_times), strike * accrual)
    c[-1] += 1.0   # final cashflow includes notional

    # find r* solving sum_i c_i * P(expiry, t_i ; r*) = 1
    def coupon_bond_value(r_star):
        return sum(ci * hw_bond(a, sigma, expiry, ti, r_star, P)
                   for ci, ti in zip(c, pay_times)) - 1.0

    try:
        r_star = brentq(coupon_bond_value, -0.5, 0.5, maxiter=200)
    except ValueError:
        r_star = 0.0

    # strikes for each ZBO = model bond price at r*
    price = 0.0
    for ci, ti in zip(c, pay_times):
        Ki = hw_bond(a, sigma, expiry, ti, r_star, P)
        price += ci * hw_zbo(a, sigma, expiry, ti, Ki, P, option="put")
    return notional * price


# --------------------------------------------------------------------------- #
# Black swaption (market quoting convention) for converting vols <-> prices
# --------------------------------------------------------------------------- #
def swap_rate_and_annuity(expiry, tenor, P, freq=2):
    pay_times = expiry + np.arange(1, int(tenor * freq) + 1) / freq
    accrual = 1.0 / freq
    annuity = accrual * sum(float(P(t)) for t in pay_times)
    fwd_swap = (float(P(expiry)) - float(P(pay_times[-1]))) / annuity
    return fwd_swap, annuity


def black_payer_swaption(expiry, tenor, strike, vol, P, freq=2):
    fwd, annuity = swap_rate_and_annuity(expiry, tenor, P, freq)
    if vol <= 0 or expiry <= 0:
        return annuity * max(fwd - strike, 0.0)
    d1 = (np.log(fwd / strike) + 0.5 * vol ** 2 * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry)
    return annuity * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))


# --------------------------------------------------------------------------- #
# Calibration of (a, sigma) to a swaption vol surface
# --------------------------------------------------------------------------- #
def calibrate_hw(swaptions, P, freq=2):
    """Fit Hull--White (a, sigma) to ATM swaption Black vols.

    swaptions : list of dicts {expiry, tenor, vol}  (ATM, so strike = fwd swap).
    Minimises squared differences between Hull--White model prices and the
    Black prices implied by the quoted vols.
    """
    targets = []
    for s in swaptions:
        fwd, _ = swap_rate_and_annuity(s["expiry"], s["tenor"], P, freq)
        mkt = black_payer_swaption(s["expiry"], s["tenor"], fwd, s["vol"], P, freq)
        targets.append((s["expiry"], s["tenor"], fwd, mkt))

    def objective(params):
        a, sigma = params
        if a <= 1e-4 or sigma <= 1e-5:
            return 1e6
        se = 0.0
        for expiry, tenor, strike, mkt in targets:
            model = hw_payer_swaption(a, sigma, expiry, tenor, strike, P, freq)
            se += (model - mkt) ** 2
        return se

    best = None
    # multistart to dodge local minima
    for a0 in (0.03, 0.1, 0.3, 1.0):
        for s0 in (0.005, 0.01, 0.02):
            res = minimize(objective, [a0, s0], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
    a, sigma = best.x
    return dict(a=float(a), sigma=float(sigma), sse=float(best.fun)), targets
