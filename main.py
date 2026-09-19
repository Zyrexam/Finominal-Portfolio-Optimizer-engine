import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import data_setup as eng

app = FastAPI(title="Portfolio Optimizer")

INFO, RET_WIDE, FAC_WIDE = eng.load_data()
NAMES = dict(zip(INFO["ticker"], INFO["fund_name"]))
YIELDS = dict(zip(INFO["ticker"], INFO["dividend_yield"]))

STRATEGY_ALIASES = {
    "equal": "equal_weights",
    "min_vol": "minimize_volatility",
    "min_volatility": "minimize_volatility",
    "max_sharpe": "maximize_sharpe_ratio",
    "max_sharpe_ratio": "maximize_sharpe_ratio",
    "risk_parity": "risk_parity",
    "min_drawdown": "minimize_drawdown",
    "minimize_drawdown": "minimize_drawdown",
}

STRATEGIES = {
    "equal_weights": None,
    "minimize_volatility": eng.min_vol_weights,
    "maximize_sharpe_ratio": eng.max_sharpe_weights,
    "risk_parity": eng.risk_parity_weights,
    "minimize_drawdown": eng.min_drawdown_weights,
}


class Security(BaseModel):
    ticker: str
    current_weight: float


class OptimizeRequest(BaseModel):
    securities: list[Security]
    strategy: str
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_dividend: float | None = None


def _metrics(cols, w):
    m = RET_WIDE[cols].dropna().values
    w = np.asarray(w, dtype=float)
    rp = m @ w
    if len(cols) == 1:
        # No covariance for one asset; portfolio series is the asset series.
        vol = float(np.std(m[:, 0]))
        mean = float(np.mean(m[:, 0]))
        sharpe = mean / vol if vol else 0.0
    else:
        mean = np.mean(m, axis=0)
        cov = np.cov(m, rowvar=False)
        vol = eng.portfolio_volatility(w, cov)
        sharpe = eng.sharpe_ratio(w, mean, cov)
    return {
        "expected_daily_return": round(float(w @ np.mean(m, axis=0)), 8),
        "volatility": round(vol, 8),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown": round(eng.max_drawdown(rp), 6),
    }


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    # Validation: tickers known, unique, weights sane, strategy supported.
    cols = [s.ticker for s in req.securities]
    unknown = [t for t in cols if t not in NAMES]
    if unknown:
        raise HTTPException(400, f"Unknown tickers: {unknown}")
    if len(set(cols)) != len(cols):
        raise HTTPException(400, f"Duplicate tickers: {cols}")
    if any(not (0 <= s.current_weight <= 100) for s in req.securities):
        raise HTTPException(400, "Each current_weight must be within [0, 100]")
    if abs(sum(s.current_weight for s in req.securities) - 100) > 1e-6:
        raise HTTPException(400, "current_weight must sum to 100")
    strategy = STRATEGY_ALIASES.get(req.strategy, req.strategy)
    if strategy not in STRATEGIES:
        raise HTTPException(400, f"Unsupported strategy: {req.strategy}")
    if not (0 <= req.min_weight <= req.max_weight <= 1):
        raise HTTPException(400, "Require 0 <= min_weight <= max_weight <= 1")

    # Engine: fractions in, fractions out. Infeasible -> 422.
    try:
        if strategy == "equal_weights":
            w = eng.equal_weights(len(cols))
            if any(not (req.min_weight - 1e-9 <= o <= req.max_weight + 1e-9) for o in w):
                raise ValueError("Equal weights violate min/max bounds")
            if req.min_dividend is not None and float(w @ np.array([YIELDS[t] for t in cols])) < req.min_dividend - 1e-9:
                raise ValueError("Equal weights violate min_dividend")
        elif strategy == "minimize_drawdown":
            w, _ = eng.min_drawdown_weights(
                cols, RET_WIDE, min_w=req.min_weight, max_w=req.max_weight,
                min_dividend=req.min_dividend, dividends=YIELDS)
        else:
            w, _, _, _ = STRATEGIES[strategy](
                cols, RET_WIDE, min_w=req.min_weight, max_w=req.max_weight,
                min_dividend=req.min_dividend, dividends=YIELDS)
    except ValueError as e:
        raise HTTPException(422, str(e))

    changes = [{
        "ticker": s.ticker,
        "security_name": NAMES[s.ticker],
        "current_weight": round(s.current_weight, 2),
        "optimized_weight": round(float(o) * 100, 2),
        "change": round(float(o) * 100 - s.current_weight, 2),
    } for s, o in zip(req.securities, w)]

    cur = np.array([s.current_weight / 100 for s in req.securities])

    return {
        "optimization_strategy": strategy,
        "allocation_changes": changes,
        "portfolio_dividend_yield": round(float(w @ np.array([YIELDS[t] for t in cols])) * 100, 3),
        "metrics": {
            "current_portfolio": _metrics(cols, cur),
            "optimized_portfolio": _metrics(cols, np.asarray(w, dtype=float)),
        },
    }
