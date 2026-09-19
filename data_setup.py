import numpy as np
import pandas as pd


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
    # ponytail: daily vol, no annualization; rankings identical, annualize only for display.
    return float(np.sqrt(w.T @ cov @ w))


def sharpe_ratio(w, mean, cov):
    vol = portfolio_volatility(w, cov)
    return float(w @ mean) / vol if vol else 0.0


def max_drawdown(pr):
    cum = np.cumprod(1 + pr)
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


if __name__ == "__main__":
    info, ret_wide, fac_wide = load_data()

    print(info[["ticker", "dividend_yield"]].to_string(index=False))
    print("ret_wide:", ret_wide.shape, "| fac_wide:", fac_wide.shape)
    # ponytail: inner-join dates; fewer tickers = longer history, so align per case.
    print("ALL-5 common rows:", len(ret_wide.dropna()))

    cols = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    m = ret_wide[cols].dropna().values
    w = np.array([0.2] * 5)

    print("Mean:", round(float(w @ np.mean(m, axis=0)), 8),
          "Vol:", round(portfolio_volatility(w, np.cov(m, rowvar=False)), 8),
          "Sharpe:", round(sharpe_ratio(w, np.mean(m, axis=0), np.cov(m, rowvar=False)), 6),
          "MaxDD:", round(max_drawdown(m @ w), 6))
