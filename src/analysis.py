"""All figures and result tables for the combined project."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from black_scholes import bs_price
from lsm import simulate_gbm, simulate_heston, lsm_price, binomial_american
from heston import heston_price_cos
from rate_data import (TREASURY_TENORS, TREASURY_PAR_CURVE, swaption_list,
                       SWAPTION_EXPIRIES, SWAPTION_TENORS, SWAPTION_VOLS)
from short_rate import (bootstrap_curve, calibrate_hw, hw_payer_swaption,
                        black_payer_swaption, swap_rate_and_annuity, instantaneous_forward)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# Heston params from the prior calibration (SPY), reused as a realistic set.
HESTON = dict(kappa=5.2528, theta=0.0542, xi=1.6643, rho=-0.6371, v0=0.0215)


# ========================================================================= #
# Part 1: LSM
# ========================================================================= #
def lsm_validation_table():
    """LSM vs binomial across moneyness (BS world)."""
    S0, r, q, sigma, T = 100, 0.05, 0.0, 0.2, 1.0
    rows = []
    for K in [80, 90, 100, 110, 120]:
        paths = simulate_gbm(S0, r, q, sigma, T, 50, 200000, seed=1)
        lsm, se = lsm_price(paths, K, r, T, "put", degree=3)
        binom = binomial_american(S0, K, r, q, sigma, T, 2000, "put")
        euro = bs_price(S0, K, T, r, q, sigma, "put")
        rows.append(dict(K=K, lsm=lsm, se=se, binom=binom, euro=euro,
                         premium=binom - euro))
    json.dump(rows, open(os.path.join(RES, "lsm_validation.json"), "w"), indent=2)
    return rows


def fig_lsm_premium(rows):
    Ks = [r["K"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(Ks, [r["binom"] for r in rows], "o-", color="#C0504D", label="American (binomial)")
    ax.plot(Ks, [r["euro"] for r in rows], "s--", color="#2E75B6", label="European (Black--Scholes)")
    ax.plot(Ks, [r["lsm"] for r in rows], "^", color="#1F8A4C", ms=9, label="American (LSM)")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Put price")
    ax.set_title("American vs European put: early-exercise premium")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lsm_premium.png")); plt.close(fig)


def fig_exercise_boundary():
    S0, r, q, sigma, T, K = 100, 0.05, 0.0, 0.2, 1.0, 100
    paths = simulate_gbm(S0, r, q, sigma, T, 50, 200000, seed=3)
    _, _, boundary = lsm_price(paths, K, r, T, "put", degree=4, return_boundary=True)
    t = np.linspace(0, T, len(boundary))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    valid = ~np.isnan(boundary)
    ax.plot(t[valid], boundary[valid], color="#7030A0", lw=1.8)
    ax.axhline(K, color="grey", lw=1, ls="--", label="Strike")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Critical stock price")
    ax.set_title("Estimated early-exercise boundary (American put)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exercise_boundary.png")); plt.close(fig)


def heston_american_table():
    """American vs European put under Heston (early-exercise premium w/ stoch vol)."""
    S0, r, q, T = 100, 0.05, 0.0, 1.0
    rows = []
    for K in [90, 100, 110]:
        paths = simulate_heston(S0, r, q, HESTON["kappa"], HESTON["theta"],
                                HESTON["xi"], HESTON["rho"], HESTON["v0"],
                                T, 100, 200000, seed=5)
        am, se = lsm_price(paths, K, r, T, "put", degree=3)
        eu = heston_price_cos(S0, K, T, r, q, HESTON["kappa"], HESTON["theta"],
                              HESTON["xi"], HESTON["rho"], HESTON["v0"], "put")
        rows.append(dict(K=K, american=am, se=se, european=eu, premium=am - eu))
    json.dump(rows, open(os.path.join(RES, "heston_american.json"), "w"), indent=2)
    return rows


# ========================================================================= #
# Part 2: Interest rates
# ========================================================================= #
def fig_yield_curve(P, ts):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    tg = np.linspace(0.1, 30, 200)
    zero = -np.log(np.array([P(t) for t in tg])) / tg
    fwd = np.array([instantaneous_forward(P, t) for t in tg])
    ax.plot(TREASURY_TENORS, np.array(TREASURY_PAR_CURVE) * 100, "o",
            color="#2E75B6", label="Par yield (market)")
    ax.plot(tg, zero * 100, color="#C0504D", lw=1.8, label="Bootstrapped zero rate")
    ax.plot(tg, fwd * 100, color="#1F8A4C", lw=1.4, ls="--", label="Instantaneous forward")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Rate (%)")
    ax.set_title("U.S. Treasury curve: par, zero and forward rates")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "yield_curve.png")); plt.close(fig)


def fig_swaption_fit(params, P):
    """Heatmap of model-vs-market swaption price errors, and a vol-surface plot."""
    a, sigma = params["a"], params["sigma"]
    err = np.zeros_like(SWAPTION_VOLS)
    for i, e in enumerate(SWAPTION_EXPIRIES):
        for j, ten in enumerate(SWAPTION_TENORS):
            fwd, _ = swap_rate_and_annuity(e, ten, P)
            mkt = black_payer_swaption(e, ten, fwd, SWAPTION_VOLS[i, j], P)
            mdl = hw_payer_swaption(a, sigma, e, ten, fwd, P)
            err[i, j] = (mdl - mkt) * 1e4   # in bp of notional

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    im0 = axes[0].imshow(SWAPTION_VOLS * 100, cmap="viridis", aspect="auto", origin="lower")
    axes[0].set_xticks(range(len(SWAPTION_TENORS))); axes[0].set_xticklabels([f"{t:g}y" for t in SWAPTION_TENORS])
    axes[0].set_yticks(range(len(SWAPTION_EXPIRIES))); axes[0].set_yticklabels([f"{e:g}y" for e in SWAPTION_EXPIRIES])
    axes[0].set_xlabel("Swap tenor"); axes[0].set_ylabel("Option expiry")
    axes[0].set_title("Market swaption vols (%)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(err, cmap="RdBu_r", aspect="auto", origin="lower",
                         vmin=-abs(err).max(), vmax=abs(err).max())
    axes[1].set_xticks(range(len(SWAPTION_TENORS))); axes[1].set_xticklabels([f"{t:g}y" for t in SWAPTION_TENORS])
    axes[1].set_yticks(range(len(SWAPTION_EXPIRIES))); axes[1].set_yticklabels([f"{e:g}y" for e in SWAPTION_EXPIRIES])
    axes[1].set_xlabel("Swap tenor"); axes[1].set_ylabel("Option expiry")
    axes[1].set_title("Hull--White price error (bp of notional)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "swaption_fit.png")); plt.close(fig)


def hw_synthetic_recovery(P):
    """Generate swaptions from known (a,sigma), recalibrate, check recovery.

    Calibrates directly on Hull--White prices (rather than round-tripping through
    Black vols) for speed; the test still confirms the optimiser recovers the
    true parameters.
    """
    from scipy.optimize import minimize
    a_true, s_true = 0.08, 0.012
    targets = []
    for e in SWAPTION_EXPIRIES:
        for ten in SWAPTION_TENORS:
            fwd, _ = swap_rate_and_annuity(e, ten, P)
            targets.append((e, ten, fwd, hw_payer_swaption(a_true, s_true, e, ten, fwd, P)))

    def obj(x):
        a, s = x
        if a <= 1e-4 or s <= 1e-5:
            return 1e6
        return sum((hw_payer_swaption(a, s, e, ten, fwd, P) - mkt) ** 2
                   for e, ten, fwd, mkt in targets)

    best = None
    for a0 in (0.03, 0.08, 0.2):
        for s0 in (0.008, 0.012, 0.02):
            r = minimize(obj, [a0, s0], method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-16, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    out = dict(true=dict(a=a_true, sigma=s_true),
               recovered=dict(a=float(best.x[0]), sigma=float(best.x[1])))
    json.dump(out, open(os.path.join(RES, "hw_synthetic.json"), "w"), indent=2)
    return out


def main():
    print("LSM validation..."); rows = lsm_validation_table(); fig_lsm_premium(rows)
    print("exercise boundary..."); fig_exercise_boundary()
    print("Heston American..."); ha = heston_american_table()
    print("  Heston American premia:", [(r["K"], round(r["premium"], 3)) for r in ha])

    P, ts, Ps = bootstrap_curve(TREASURY_TENORS, TREASURY_PAR_CURVE)
    print("yield curve..."); fig_yield_curve(P, ts)
    print("HW calibration...")
    params, _ = calibrate_hw(swaption_list(), P)
    json.dump(params, open(os.path.join(RES, "hw_params.json"), "w"), indent=2)
    print(f"  a={params['a']:.4f} sigma={params['sigma']:.5f}")
    fig_swaption_fit(params, P)
    print("HW synthetic recovery..."); rec = hw_synthetic_recovery(P)
    print("  true:", rec["true"], "recovered:", {k: round(v, 4) for k, v in rec["recovered"].items()})
    print("DONE")


if __name__ == "__main__":
    main()
