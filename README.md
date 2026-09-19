# Portfolio Optimizer API

REST API replicating the Finominal Portfolio Optimizer engine: 5 funds (SPY, IEFA, VEA, AGG, GLD), 5 strategies, security-level and portfolio-level constraints.

## Strategies

- `equal_weights` — 1/N baseline
- `risk_parity` — equal risk contribution (normalized objective; raw form stalls SLSQP at ~1e-10 scale)
- `minimize_volatility` — min `sqrt(w'Σw)`
- `maximize_sharpe_ratio` — max excess-return/vol, risk-free 2%/yr (calibrated to reference tool)
- `minimize_drawdown` — min peak-to-trough of `cumprod(1+R_p)`

## Run locally

```bash
pip install fastapi uvicorn numpy pandas scipy openpyxl
uvicorn main:app --reload
```

Docs: `http://127.0.0.1:8000/docs`

## Endpoint

`POST /optimize`

```json
{
  "securities": [
    {"ticker": "IEFA", "current_weight": 20},
    {"ticker": "GLD", "current_weight": 20},
    {"ticker": "AGG", "current_weight": 20},
    {"ticker": "VEA", "current_weight": 20},
    {"ticker": "SPY", "current_weight": 20}
  ],
  "strategy": "maximize_sharpe_ratio",
  "min_weight": 0.05,
  "max_weight": 0.40,
  "min_dividend": 0.025
}
```

Response: `allocation_changes` (`ticker`, `security_name`, `current_weight`, `optimized_weight`, `change`) + `portfolio_dividend_yield` + `metrics` (return, volatility, Sharpe, max drawdown for current and optimized portfolios).

Errors: bad input → 400, infeasible constraints → 422 (never silent invalid weights).

## Validation (verified against live tool)

- Case 1 Equal (IEFA/SPY): 50/50 exact ✔
- Case 2 Risk Parity (VEA/AGG): 20.13/79.87 vs 20.11/79.89, diff 0.02pp ✔
- Case 3 Min Vol (SPY/AGG/GLD): 6.66/91.77/1.57, bonds dominate ✔
- Case 4 Max Sharpe (all-5): SPY 70.29/GLD 29.71 vs ~71.8/28.2, diffs ~1.5pp ✔
- Case 5 constrained (5–40%, yield ≥2.5%): 13.76/5.0/40/5.0/36.24, yield exactly 2.500 ✔ bounds match; residual diffs (IEFA +3.11, SPY −3.09) because the live tool's dividend data differs from `Data.xlsx` — its weights at our xlsx yields give 2.43% (below the floor) while it displays 2.63%. See `report.txt`.

## Layout

- `main.py` — FastAPI layer (validation, dispatch, metrics)
- `data_setup.py` — engine (pivot/alignment, objectives, single constrained optimizer)
- `Data.xlsx` — fund returns, factor returns, fund info
- `report.txt` — validation numbers
