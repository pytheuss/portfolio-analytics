"""Hand-checkable tests for volatility, drawdown, Sharpe, beta, round trips and sizing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ingest import LEDGER_COLUMNS, TRADE_COLUMNS, PortfolioData
from src.risk import (
    WEEKDAYS_PER_YEAR,
    annualized_volatility,
    average_true_range,
    beta_and_correlation,
    max_drawdown,
    position_sizing,
    round_trips,
    sharpe_ratio,
    sizing_summary,
    trade_statistics,
)

D = pd.Timestamp
CONFIG = {"risk": {"target_risk_per_trade": 0.01, "atr_period": 3, "atr_multiple": 2.0}}


# --- portfolio-level risk -----------------------------------------------------------

def test_annualized_volatility():
    daily = pd.Series([0.01, -0.01, 0.01, -0.01])
    # sample sd of +/-1% alternating = 0.011547; x sqrt(260.9)
    assert annualized_volatility(daily) == pytest.approx(0.011547 * np.sqrt(WEEKDAYS_PER_YEAR), rel=1e-4)


def test_max_drawdown_dates():
    days = pd.bdate_range("2026-01-05", periods=5)
    growth = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3], index=days)
    dd = max_drawdown(growth)
    assert dd["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1)      # -25%
    assert (dd["peak_date"], dd["trough_date"], dd["recovery_date"]) == (
        "2026-01-06", "2026-01-07", "2026-01-09")


def test_drawdown_not_recovered():
    growth = pd.Series([1.0, 0.8, 0.9], index=pd.bdate_range("2026-01-05", periods=3))
    assert max_drawdown(growth)["recovery_date"] is None


def test_sharpe_ratio_with_zero_risk_free_rate():
    daily = pd.Series([0.02, 0.0])
    # mean 1%, sample sd 1.4142%: Sharpe = 0.7071 x sqrt(260.9)
    assert sharpe_ratio(daily, 0.0) == pytest.approx(np.sqrt(0.5) * np.sqrt(WEEKDAYS_PER_YEAR))


def test_risk_free_rate_lowers_sharpe():
    daily = pd.Series([0.02, 0.0, 0.01, 0.005])
    assert sharpe_ratio(daily, 0.05) < sharpe_ratio(daily, 0.0)


def test_flat_returns_give_no_sharpe_or_beta():
    flat = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(flat, 0.0))
    assert all(np.isnan(x) for x in beta_and_correlation(flat, pd.Series([0.01, -0.01, 0.0])))


def test_beta_of_a_two_times_leveraged_copy():
    benchmark = pd.Series([0.01, -0.02, 0.015, 0.003])
    beta, correlation = beta_and_correlation(2 * benchmark, benchmark)
    assert beta == pytest.approx(2.0)
    assert correlation == pytest.approx(1.0)


# --- round trips ---------------------------------------------------------------------

def test_round_trip_uses_average_cost():
    # Buy 10 @ 10 and 10 @ 20 (average 15), sell 20 @ 18: profit 20 x 3 = 60, less 1 fee = 59.
    # Then buy 5 @ 10, sell 5 @ 8: loss -10.
    data = _data([
        ("2026-02-02", "ABC.TO", "buy", 10, 10.0),
        ("2026-02-03", "ABC.TO", "buy", 10, 20.0),
        ("2026-02-04", "ABC.TO", "sell", 20, 18.0, 1.0),
        ("2026-02-05", "ABC.TO", "buy", 5, 10.0),
        ("2026-02-06", "ABC.TO", "sell", 5, 8.0),
    ])
    trips = round_trips(data, _perf())
    assert trips["pnl_cad"].tolist() == pytest.approx([59.0, -10.0])

    stats = trade_statistics(trips)
    assert stats["win_rate"] == 0.5
    assert stats["profit_factor"] == pytest.approx(5.9)              # 59 / 10
    assert stats["payoff_ratio"] == pytest.approx(5.9)


def test_split_changes_shares_not_cost():
    # Buy 10 @ 100 (cost 1,000). 2-for-1 split -> 20 shares, still cost 1,000. Sell 20 @ 55: +100.
    data = _data([
        ("2026-02-02", "ABC.TO", "buy", 10, 100.0),
        ("2026-02-03", "ABC.TO", "adjust", 10, np.nan),
        ("2026-02-04", "ABC.TO", "sell", 20, 55.0),
    ])
    assert round_trips(data, _perf())["pnl_cad"].tolist() == pytest.approx([100.0])


def test_expired_option_loses_premium():
    data = _data([
        ("2026-02-02", "ABC260220C00010000", "buy", 2, 30.0, 0.0, "option"),
        ("2026-02-20", "ABC260220C00010000", "adjust", -2, np.nan, 0.0, "option"),
    ])
    trips = round_trips(data, _perf())
    assert trips["pnl_cad"].tolist() == pytest.approx([-60.0])
    assert trips["asset_type"].tolist() == ["option"]


# --- position sizing ----------------------------------------------------------------

def test_atr_of_steady_range_equals_the_range():
    days = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"high": 11.0, "low": 9.0, "close": 10.0}, index=days)
    assert average_true_range(prices, 3).dropna().tolist() == pytest.approx([2.0] * 4)


def test_position_sizing_flags_trades_above_one_percent():
    # Account equity 10,000. ATR = 0.50, stop = 2 x ATR = 1.00 below entry.
    # 100 shares risk 100 = 1.0% (within); 300 shares risk 300 = 3.0% (over).
    days = pd.bdate_range("2026-01-05", periods=10)
    history = {"ABC.TO": pd.DataFrame({"high": 10.25, "low": 9.75, "close": 10.0}, index=days)}
    data = _data([
        ("2026-01-15", "ABC.TO", "buy", 100, 10.0),
        ("2026-01-16", "ABC.TO", "buy", 300, 10.0),
    ])
    perf = _perf(account_values=pd.DataFrame({"Main": 10_000.0}, index=days))

    sizing = position_sizing(data, perf, CONFIG, history=history)
    assert sizing["risk_cad"].tolist() == pytest.approx([100.0, 300.0])
    assert sizing["risk_pct"].tolist() == pytest.approx([0.01, 0.03])
    assert sizing["over_target"].tolist() == [False, True]
    assert sizing["stop_price"].tolist() == pytest.approx([9.0, 9.0])

    summary = sizing_summary(sizing, 0.01)
    assert (summary["within_target"], summary["over_target"]) == (1, 1)


def test_your_stop_overrides_atr_and_long_option_risks_premium():
    days = pd.bdate_range("2026-01-05", periods=10)
    history = {"ABC.TO": pd.DataFrame({"high": 10.25, "low": 9.75, "close": 10.0}, index=days)}
    data = _data([
        ("2026-01-15", "ABC.TO", "buy", 100, 10.0, 0.0, "stock", 9.5),   # your stop: risk 50
        ("2026-01-15", "ABC260220C00010000", "buy", 2, 30.0, 0.0, "option"),  # premium: 60
    ])
    perf = _perf(account_values=pd.DataFrame({"Main": 10_000.0}, index=days))
    sizing = position_sizing(data, perf, CONFIG, history=history)
    assert sizing["method"].tolist() == ["your stop", "premium"]
    assert sizing["risk_cad"].tolist() == pytest.approx([50.0, 60.0])


# --- helpers ---------------------------------------------------------------------

def _data(rows) -> PortfolioData:
    records = []
    for row in rows:
        date, ticker, action, qty, price, *rest = row
        fee = rest[0] if len(rest) > 0 else 0.0
        asset = rest[1] if len(rest) > 1 else "stock"
        stop = rest[2] if len(rest) > 2 else np.nan
        records.append(dict(date=D(date), timestamp=D(date), account="Main", ticker=ticker,
                            asset_type=asset, action=action, quantity=float(qty), price=price,
                            currency="CAD", fees=fee, stop_price=stop))
    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    return PortfolioData(trades=trades, cash_ledger=pd.DataFrame(columns=LEDGER_COLUMNS),
                         opening_positions=pd.DataFrame(columns=["account", "ticker", "quantity"]),
                         opening_cash=pd.DataFrame(columns=["account", "currency", "amount"]),
                         start_date=trades["date"].min())


def _perf(account_values=None):
    """The few PerformanceResult fields the risk functions read."""
    days = pd.bdate_range("2026-01-01", "2026-03-31")
    empty = pd.DataFrame(index=days[:1], columns=pd.MultiIndex.from_arrays([[], []],
                                                                            names=["account", "ticker"]))
    return SimpleNamespace(
        usdcad=pd.Series(1.0, index=days),
        values=pd.DataFrame({"total": 0.0}, index=days[:1]),
        shares=empty, position_values=empty,
        account_values=account_values if account_values is not None else pd.DataFrame(index=days),
    )
