"""Generate FAKE sample data so the app runs without real portfolio data.

Writes:
    data/sample_trades.csv      ~40 trades, May 2025 to today, CAD and USD
    data/sample_cash_flows.csv  deposits and withdrawals

The trades are invented, but each price is the real closing price that day
(from yfinance). That way, when Phase 2 values the portfolio at market prices,
the returns look like a real account rather than noise.

Most buys are sized so that (entry - stop) x shares is about 1% of equity, the
same rule the risk check tests. A few are deliberately oversized so the Phase 3
check has something to flag.

Run once from the project root (the output CSVs are committed):
    python scripts/make_sample_data.py
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", module="yfinance")

SEED = 7
START, END = "2025-05-01", "2026-10-29"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CAD_TICKERS = ["RY.TO", "TD.TO", "ENB.TO", "CNQ.TO", "SHOP.TO", "CP.TO",
               "BN.TO", "ATD.TO", "XIU.TO", "ZEB.TO", "XEQT.TO"]
USD_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "COST", "GOOGL", "QQQ", "XLE"]
FX_TICKER = "CADUSD=X"  # USD per 1 CAD

# (date, amount in CAD): positive = deposit, negative = withdrawal
CASH_FLOWS = [("2025-05-01", 50_000.0), ("2026-01-15", 10_000.0), ("2026-08-04", -4_000.0)]

N_TRADES = 40
FEE = 4.95                 # flat commission per trade, in the trade's currency
OVERSIZE_PROBABILITY = 0.15


def download_closes() -> pd.DataFrame:
    """Daily closes for every ticker plus FX, on days both TSX and NYSE traded."""
    tickers = CAD_TICKERS + USD_TICKERS + [FX_TICKER]
    data = yf.download(tickers, start=START, end=END, auto_adjust=False, progress=False)
    closes = data["Close"].dropna(how="any")  # keep only days with a price for everything
    closes.index = closes.index.tz_localize(None)
    return closes


def cad_per_unit(ticker: str, fx: float) -> float:
    """CAD value of one unit of the ticker's currency."""
    return 1.0 if ticker in CAD_TICKERS else 1.0 / fx


def equity_cad(cash: float, holdings: dict, prices: pd.Series) -> float:
    """Cash plus holdings marked to that day's close, in CAD."""
    fx = prices[FX_TICKER]
    value = sum(qty * prices[t] * cad_per_unit(t, fx) for t, qty in holdings.items())
    return cash + value


def make_buy(rng, date, prices, cash, holdings):
    """Size a new position so the loss to the stop is ~1% of equity (sometimes more)."""
    candidates = [t for t in CAD_TICKERS + USD_TICKERS if t not in holdings]
    ticker = str(rng.choice(candidates))
    fx_rate = cad_per_unit(ticker, prices[FX_TICKER])

    price = round(float(prices[ticker]), 2)
    stop = round(price * (1 - rng.uniform(0.06, 0.12)), 2)
    oversized = rng.random() < OVERSIZE_PROBABILITY
    risk_pct = rng.uniform(0.014, 0.022) if oversized else rng.uniform(0.0085, 0.0100)

    risk_budget = equity_cad(cash, holdings, prices) * risk_pct
    qty = math.floor(risk_budget / ((price - stop) * fx_rate))
    affordable = math.floor((cash - FEE * fx_rate) / (price * fx_rate))
    qty = min(qty, affordable)
    if qty < 1:
        return None

    holdings[ticker] = qty
    cash -= (qty * price + FEE) * fx_rate
    trade = dict(date=date, ticker=ticker, action="buy", quantity=qty, price=price,
                 currency="CAD" if ticker in CAD_TICKERS else "USD", fees=FEE, stop_price=stop)
    return trade, cash


def make_sell(rng, date, prices, cash, holdings):
    """Exit a random position fully, or trim half of it."""
    ticker = str(rng.choice(sorted(holdings)))
    fx_rate = cad_per_unit(ticker, prices[FX_TICKER])
    held = holdings[ticker]
    qty = held if (rng.random() < 0.65 or held < 2) else held // 2

    price = round(float(prices[ticker]), 2)
    holdings[ticker] = held - qty
    if holdings[ticker] == 0:
        del holdings[ticker]
    cash += (qty * price - FEE) * fx_rate
    trade = dict(date=date, ticker=ticker, action="sell", quantity=qty, price=price,
                 currency="CAD" if ticker in CAD_TICKERS else "USD", fees=FEE, stop_price=None)
    return trade, cash


def simulate(closes: pd.DataFrame, rng) -> list[dict]:
    """Walk forward through randomly chosen trade dates, buying or selling."""
    days = closes.index
    # Force three early trades so the portfolio is invested from the start.
    later = rng.choice(days[5:], size=N_TRADES - 3, replace=False)
    trade_days = sorted(set(days[[0, 2, 4]]) | set(later))

    flows = [(pd.Timestamp(d), amt) for d, amt in CASH_FLOWS]
    cash, holdings, trades = 0.0, {}, []

    for day in trade_days:
        cash += sum(amt for d, amt in flows if d <= day)
        flows = [(d, amt) for d, amt in flows if d > day]
        prices = closes.loc[day]

        wants_buy = len(holdings) < 5 or (len(holdings) < 9 and rng.random() < 0.5)
        result = make_buy(rng, day, prices, cash, holdings) if wants_buy else None
        if result is None and holdings:
            result = make_sell(rng, day, prices, cash, holdings)
        if result is None:
            continue
        trade, cash = result
        trades.append(trade)

    final_equity = equity_cad(cash, holdings, closes.iloc[-1])
    print(f"Final holdings: {holdings}")
    print(f"Final cash: {cash:,.2f} CAD | equity: {final_equity:,.2f} CAD")
    return trades


def main() -> None:
    rng = np.random.default_rng(SEED)
    closes = download_closes()
    trades = pd.DataFrame(simulate(closes, rng))
    trades["date"] = trades["date"].dt.strftime("%Y-%m-%d")
    trades.to_csv(DATA_DIR / "sample_trades.csv", index=False)

    flows = pd.DataFrame(CASH_FLOWS, columns=["date", "amount"])
    flows["currency"] = "CAD"
    flows.to_csv(DATA_DIR / "sample_cash_flows.csv", index=False)

    print(f"Wrote {len(trades)} trades "
          f"({(trades.action == 'buy').sum()} buys, {(trades.action == 'sell').sum()} sells)")


if __name__ == "__main__":
    main()
