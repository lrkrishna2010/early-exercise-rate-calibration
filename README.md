# American Options (LSM) & Interest-Rate Calibration (Hull–White)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Two pillars of derivatives pricing beyond closed-form European valuation:
**early-exercise pricing by least-squares Monte Carlo** (American options under
Black–Scholes *and* Heston), and **Hull–White calibration** to a swaption surface
on a real Treasury curve. Accompanying dissertation (LaTeX + PDF) in [`paper/`](paper/).

> **TL;DR** — LSM matches a binomial benchmark to within Monte Carlo error and
> extends cleanly to stochastic volatility (American Heston premia where no closed
> form exists). Hull–White, fitted exactly to a bootstrapped Treasury curve,
> calibrates to 36 swaptions to **< 4 bp of notional**, with a synthetic-recovery
> test nailing the known parameters exactly.

## What's inside

| Component | Detail |
|-----------|--------|
| **LSM** | Longstaff–Schwartz regression on simulated paths; degree-3 basis |
| **Benchmark** | Cox–Ross–Rubinstein binomial tree (BS world, independent check) |
| **Heston American** | LSM on full-truncation Heston paths; COS for the European leg |
| **Curve bootstrap** | zero/discount curve from real Treasury par yields |
| **Hull–White** | affine bond analytics, Jamshidian swaption decomposition |
| **Calibration** | multi-start Nelder–Mead fit of (a, σ); synthetic recovery |

## Key results

American put (BS, S=K=100, σ=20%, r=5%, T=1): LSM **6.07** vs binomial **6.09**;
early-exercise premium over European grows 0.04 → 2.74 with moneyness.
American Heston premia: 0.11 (K=90) → 0.74 (K=110).
Hull–White calibration: a = 0.020, σ = 0.0131, mean abs error < 4 bp; synthetic
recovery exact.

## Repository layout

```
src/
  black_scholes.py   # BS pricing, Greeks, implied vol
  heston.py          # Heston char. function, COS pricing, MC paths
  lsm.py             # Longstaff–Schwartz + GBM/Heston simulators + binomial benchmark
  short_rate.py      # curve bootstrap, Hull–White bonds & swaptions, calibration
  rate_data.py       # real Treasury curve + realistic swaption surface
  analysis.py        # all figures + validation tables
  run_all.py         # end-to-end
  fetch_curve.py     # pull a fresh Treasury curve (run locally)
results/             # lsm/heston-american/hw params + synthetic recovery
figures/             # premium, exercise boundary, yield curve, swaption fit
paper/               # LaTeX dissertation + compiled PDF
```

## Quick start

```bash
pip install -r requirements.txt
cd src && python run_all.py        # ~5 min
```

Refresh the Treasury curve yourself (run locally — Treasury isn't reachable in sandboxes):

```bash
python fetch_curve.py    # prints lines to paste into rate_data.py
```

Compile the dissertation: `cd paper && pdflatex paper.tex` (×3).

## Method notes & caveats

- **LSM validation.** Benchmarked against a 2,000-step binomial tree (BS) and the
  no-early-exercise theorem for American calls on non-dividend stocks. Under Heston
  there is no closed form, which is exactly LSM's use case.
- **Hull–White.** Fitted exactly to the curve first; calibration adjusts only (a, σ).
  A single factor cannot fit the whole swaption surface — a structured residual
  appears in the long-expiry/mid-tenor corner (a G2++/two-factor model would fix it).
- **Data.** The Treasury curve is real (early June 2026 levels; refresh via
  `fetch_curve.py`). The swaption surface is **synthetic but realistic**, since free
  swaption data is unavailable; the calibration is real-data-ready for any surface in
  the same format. LSM carries a small known in-sample low bias.

## License

MIT — see [LICENSE](LICENSE).
