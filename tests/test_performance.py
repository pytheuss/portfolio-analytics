"""Hand-checkable tests for valuation, time-weighted returns and the benchmark."""

import numpy as np
import pandas as pd
import pytest

from src.ingest import LEDGER_COLUMNS, TRADE_COLUMNS, PortfolioData
from src.performance import (
    annualize,
    compound,
    compute_performance,
    daily_twr,
    external_flows_cad,
    monthly_returns,
    option_cost_changes,
    rebalanced_index,
    stock_prices_cad,
)
from src.prices import unsplit

D = pd.Timestamp


# --- time-weighted returns -------------------------------------------------

def test_deposit_is_not_a_gain():
    # Day 1: deposit 100. Day 2: grows to 110 (+10%). Day 3: deposit 100, ends at 210 (0%).
    values = pd.Series([0.0, 100.0, 110.0, 210.0])
    flows = pd.Series([0.0, 100.0, 0.0, 100.0])
    r = daily_twr(values, flows)
    assert np.isnan(r[0])                      # nothing invested yet: undefined, not 0
    assert r[1] == pytest.approx(0.0)
    assert r[2] == pytest.approx(0.10)
    assert r[3] == pytest.approx(0.0)          # 210 / (110 + 100) - 1
    assert compound(r) == pytest.approx(0.10)


def test_twr_ignores_timing_of_deposits():
    # Same +10% then -10% market, but a big deposit before the bad day.
    # The dollar loss is larger, yet the time-weighted return is the same: 1.1 x 0.9 - 1.
    values = pd.Series([100.0, 110.0, 99.0 + 900.0 * 0.9])
    flows = pd.Series([100.0, 0.0, 900.0])
    assert compound(daily_twr(values, flows)) == pytest.approx(1.1 * 0.9 - 1)


def test_monthly_returns_compound_daily_returns():
    days = pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-03"])
    daily = pd.Series([0.10, 0.10, -0.05], index=days)
    monthly = monthly_returns(daily)
    assert monthly[pd.Period("2026-01")] == pytest.approx(0.10)
    assert monthly[pd.Period("2026-02")] == pytest.approx(1.10 * 0.95 - 1)


def test_annualize_only_for_a_year_or_more():
    assert annualize(0.05, 200) is None
    assert annualize(0.21, int(2 * 365.25)) == pytest.approx(0.10, abs=1e-4)


# --- benchmark ---------------------------------------------------------------

def test_benchmark_rebalances_monthly():
    # Month 1: A +10%, B flat -> 50/50 blend +5%.
    # Month 2: A flat, B +10% -> rebalanced blend +5% again, total 1.05^2 - 1 = 10.25%.
    # (Without rebalancing, A would be 55% going into month 2 and the total would be 10.0%.)
    days = pd.to_datetime(["2026-01-30", "2026-02-27", "2026-03-31"])
    levels = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 110.0]}, index=days)
    index = rebalanced_index(levels, {"A": 0.5, "B": 0.5})
    assert index.iloc[1] == pytest.approx(1.05)
    assert index.iloc[2] == pytest.approx(1.05 ** 2)


# --- prices --------------------------------------------------------------------

def test_unsplit_restores_pre_split_prices():
    # Real prices: 800, 800, then an 8-for-1 split, then 100, 100.
    # yfinance shows every day as 100 and records the split on day 3.
    days = pd.bdate_range("2026-04-17", periods=4)
    yahoo_close = pd.Series(100.0, index=days)
    splits = pd.Series([0.0, 0.0, 8.0, 0.0], index=days)
    assert unsplit(yahoo_close, splits).tolist() == [800.0, 800.0, 100.0, 100.0]


def test_missing_market_price_falls_back_to_trade_price():
    calendar = pd.bdate_range("2026-02-02", periods=4)
    closes = pd.DataFrame({"NEW": [np.nan, np.nan, 12.0, 13.0]}, index=calendar)  # lists on day 3
    trades = _trades([("2026-02-03", "NEW", "buy", 10, 11.0, "USD")])
    usdcad = pd.Series(1.0, index=calendar)
    prices, has_market = stock_prices_cad(closes, ["NEW"], trades, usdcad, calendar)
    # Day 1 is before any price: held flat at the first known one (the trade at 11).
    assert prices["NEW"].tolist() == [11.0, 11.0, 12.0, 13.0]
    assert has_market["NEW"].tolist() == [False, False, True, True]


def test_weekend_flow_lands_on_next_business_day():
    calendar = pd.bdate_range("2026-03-05", "2026-03-10")          # Thu .. Tue
    ledger = _ledger([("2026-03-07", 500.0, "external")])           # Saturday
    flows = external_flows_cad(ledger, calendar, pd.Series(1.0, index=calendar))
    assert flows[D("2026-03-09")] == 500.0                          # Monday
    assert flows.sum() == 500.0


# --- options at cost -----------------------------------------------------------------

def test_option_bought_then_expired_loses_its_cost():
    trades = _trades([
        ("2026-02-02", "ABC260220C00010000", "buy", 2, 30.0, "USD", "option"),
        ("2026-02-20", "ABC260220C00010000", "adjust", -2, np.nan, None, "option"),
    ])
    usdcad = pd.Series(1.4, index=pd.bdate_range("2026-01-30", "2026-02-27"))
    changes = option_cost_changes(trades, usdcad)
    assert changes.tolist() == pytest.approx([84.0, -84.0])        # 2 x 30 USD x 1.4


def test_written_option_carries_negative_value_until_bought_back():
    trades = _trades([
        ("2026-08-03", "XYZ260821C00017500", "sell", 50, 10.0, "USD", "option"),
        ("2026-08-10", "XYZ260821C00017500", "buy", 50, 15.0, "USD", "option"),
    ])
    usdcad = pd.Series(1.0, index=pd.bdate_range("2026-07-31", "2026-08-31"))
    changes = option_cost_changes(trades, usdcad)
    assert changes.tolist() == pytest.approx([-500.0, 500.0])       # liability, then closed


# --- end to end --------------------------------------------------------------------

def test_compute_performance_small_portfolio():
    """Deposit 1,000 CAD, buy 10 shares at 100, price rises to 110, then a 2-for-1 split.

    Value: 1,000 -> 1,000 (bought at market) -> 1,100 -> 1,100 (split changes nothing).
    Return: +10%. Benchmark flat, so excess return is +10%.
    """
    calendar = pd.bdate_range("2026-03-02", periods=5)              # Mon .. Fri
    trades = _trades([
        ("2026-03-03", "ABC.TO", "buy", 10, 100.0, "CAD"),
        ("2026-03-06", "ABC.TO", "adjust", 10, np.nan, None),       # split: 10 -> 20 shares
    ])
    ledger = _ledger([("2026-03-03", 1000.0, "external"), ("2026-03-03", -1000.0, "internal")])
    data = PortfolioData(trades=trades, cash_ledger=ledger,
                         opening_positions=pd.DataFrame(columns=["account", "ticker", "quantity"]),
                         opening_cash=pd.DataFrame(columns=["account", "currency", "amount"]),
                         start_date=D("2026-03-03"))

    closes = pd.DataFrame({"ABC.TO": [100.0, 100.0, 105.0, 110.0, 55.0]}, index=calendar)
    fx = pd.Series(0.72, index=calendar)
    index_closes = pd.DataFrame({"^IDX": 1000.0}, index=calendar)
    components = [{"ticker": "^IDX", "currency": "CAD", "weight": 1.0}]

    result = compute_performance(data, closes, fx, index_closes, components)
    assert result.values["total"].tolist() == pytest.approx([0.0, 1000.0, 1050.0, 1100.0, 1100.0])
    assert result.summary["total_return"] == pytest.approx(0.10)
    assert result.summary["benchmark_total_return"] == pytest.approx(0.0)
    assert result.summary["excess_return"] == pytest.approx(0.10)
    assert result.summary["investment_gain_cad"] == pytest.approx(100.0)
    assert result.summary["annualized_return"] is None


# --- helpers ---------------------------------------------------------------------

def _trades(rows) -> pd.DataFrame:
    records = []
    for row in rows:
        date, ticker, action, qty, price, currency, *asset = row
        records.append(dict(date=D(date), timestamp=D(date), account="Main", ticker=ticker,
                            asset_type=asset[0] if asset else "stock", action=action,
                            quantity=float(qty), price=price, currency=currency,
                            fees=0.0, stop_price=np.nan))
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def _ledger(rows) -> pd.DataFrame:
    records = [dict(date=D(d), timestamp=D(d), account="Main", currency="CAD",
                    amount=amount, kind=kind, activity=kind) for d, amount, kind in rows]
    return pd.DataFrame(records, columns=LEDGER_COLUMNS)
