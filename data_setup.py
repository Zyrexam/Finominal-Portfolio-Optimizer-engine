import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_data():
    # Long format: date | total_return | ticker. GLD yield blank -> 0.
    info = pd.read_excel("Data.xlsx", sheet_name="Fund Info")
    fund = pd.read_excel("Data.xlsx", sheet_name="Fund Returns")
    fac = pd.read_excel("Data.xlsx", sheet_name="Factor Returns")

    fund["date"] = pd.to_datetime(fund["date"])
    fac["date"] = pd.to_datetime(fac["date"])
    info["dividend_yield"] = info["dividend_yield"].fillna(0)

    # Wide matrices: one row per date. NaN = fund did not exist yet.
    ret_wide = fund.pivot(index="date", columns="ticker", values="total_return").sort_index()
    fac_wide = fac.pivot(index="date", columns="index_ticker", values="total_return").sort_index()

    return info, ret_wide, fac_wide


def portfolio_volatility(w, cov):
    return float(np.sqrt(w.T @ cov @ w))


def sharpe_ratio(w, mean, cov):
    vol = portfolio_volatility(w, cov)
    return float(w @ mean) / vol if vol else 0.0


def max_drawdown(pr):
    cum = np.cumprod(1 + pr)
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))




def _build_constraints(n, dividends, min_w=0.0, max_w=1.0, min_dividend=None):
    bounds = [(min_w, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if min_dividend is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(w @ dividends) - min_dividend})
    return bounds, constraints


def _check_weights(w):
    if abs(float(np.sum(w)) - 1) > 1e-6 or bool((w < -1e-9).any()):
        raise ValueError("Invalid weights: must sum to 1 and never be negative")


def optimize_weights(cols, ret_wide, objective, min_w=0.0, max_w=1.0,
                     min_dividend=None, dividends=None, maxiter=100):
    n = len(cols)
    if n == 1:
        return np.array([1.0]), None, None, True
    m = ret_wide[cols].dropna().values
    mean = np.mean(m, axis=0)
    cov = np.cov(m, rowvar=False)
    divs = np.array([dividends[t] for t in cols]) if min_dividend is not None else None # type: ignore
    bounds, constraints = _build_constraints(n, divs, min_w, max_w, min_dividend)
    res = minimize(
        objective,
        np.array([1 / n] * n),
        args=(mean, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter},
    )
    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")
    _check_weights(res.x)
    return res.x, mean, cov, res.success


def min_vol_weights(cols, ret_wide, **kwargs):
    return optimize_weights(cols, ret_wide, lambda w, mean, cov: portfolio_volatility(w, cov), **kwargs)


def max_sharpe_weights(cols, ret_wide, **kwargs):
    return optimize_weights(cols, ret_wide, lambda w, mean, cov: -sharpe_ratio(w, mean, cov), **kwargs)


def equal_weights(n):
    return np.array([1 / n] * n)


def risk_parity_weights(cols, ret_wide, **kwargs):
    # Normalized: raw sum((rc - var/N)^2) is ~1e-10 scale, SLSQP stalls at init.
    def objective(w, mean, cov):
        var = max(float(w.T @ cov @ w), 1e-12)
        rc = w * (cov @ w)
        return float(np.sum((rc / var - 1 / len(w)) ** 2))

    return optimize_weights(cols, ret_wide, objective, **kwargs)


def min_drawdown_weights(cols, ret_wide, maxiter=300, min_w=0.0, max_w=1.0,
                         min_dividend=None, dividends=None):
    n = len(cols)
    if n == 1:
        return np.array([1.0]), True
    m = ret_wide[cols].dropna().values
    divs = np.array([dividends[t] for t in cols]) if min_dividend is not None else None # type: ignore
    bounds, constraints = _build_constraints(n, divs, min_w, max_w, min_dividend)
    res = minimize(
        lambda w: abs(max_drawdown(m @ w)),
        np.array([1 / n] * n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter},
    )
    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")
    _check_weights(res.x)
    return res.x, res.success


if __name__ == "__main__":
    info, ret_wide, fac_wide = load_data()

    print(info[["ticker", "dividend_yield"]].to_string(index=False))
    print("ret_wide:", ret_wide.shape, "| fac_wide:", fac_wide.shape)
    print("ALL-5 common rows:", len(ret_wide.dropna()))

    cols = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    m = ret_wide[cols].dropna().values
    w = np.array([0.2] * 5)

    print("Mean:", round(float(w @ np.mean(m, axis=0)), 8),
          "Vol:", round(portfolio_volatility(w, np.cov(m, rowvar=False)), 8),
          "Sharpe:", round(sharpe_ratio(w, np.mean(m, axis=0), np.cov(m, rowvar=False)), 6),
          "MaxDD:", round(max_drawdown(m @ w), 6))
