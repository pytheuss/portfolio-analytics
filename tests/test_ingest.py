"""Tests for loading and normalizing broker CSVs."""

import io

import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.ingest import (
    BROKER_MAPPINGS,
    STANDARD_COLUMNS,
    BrokerMapping,
    apply_stop_overrides,
    detect_broker,
    find_position_problems,
    load_standard_portfolio,
    load_trades,
    normalize_trades,
    read_csv,
    trade_cash,
)

# A made-up broker with awkward conventions, to prove the mapping layer works.
FAKE_BROKER = BrokerMapping(
    name="Fake Broker",
    columns={"date": "Trade Date", "ticker": "Symbol", "action": "Side",
             "quantity": "Qty", "price": "Fill Price", "fees": "Commission"},
    actions={"BOT": "buy", "SLD": "sell"},
    row_filter={"Type": ["Trade"]},
    default_currency="USD",
    date_format="%m/%d/%Y",
)

FAKE_CSV = """Trade Date,Type,Symbol,Side,Qty,Fill Price,Commission
01/15/2026,Trade, aapl ,BOT,10,"1,200.50",-1.00
01/20/2026,Dividend,AAPL,,,,
02/01/2026,Trade,AAPL,SLD,-4,1250.00,-1.00
"""


def fake_trades() -> pd.DataFrame:
    return normalize_trades(read_csv(io.StringIO(FAKE_CSV)), FAKE_BROKER)


def test_sample_file_loads_in_standard_format():
    trades = load_trades(PROJECT_ROOT / "data" / "sample_trades.csv")
    assert list(trades.columns) == STANDARD_COLUMNS
    assert len(trades) == 40
    assert trades["date"].is_monotonic_increasing
    assert set(trades["currency"]) == {"CAD", "USD"}


def test_sample_file_is_detected_as_standard_format():
    raw = read_csv(PROJECT_ROOT / "data" / "sample_trades.csv")
    assert detect_broker(raw.columns) is BROKER_MAPPINGS["standard"]


def test_mapping_renames_filters_and_cleans():
    trades = fake_trades()
    assert len(trades) == 2                                   # dividend row dropped
    assert list(trades["action"]) == ["buy", "sell"]
    assert list(trades["ticker"]) == ["AAPL", "AAPL"]         # trimmed, upper-cased
    assert trades.loc[0, "price"] == 1200.50                  # thousands separator handled
    assert trades.loc[1, "quantity"] == 4                     # negative sell qty made positive
    assert trades.loc[0, "fees"] == 1.00                      # fees stored as positive cost
    assert trades.loc[0, "date"] == pd.Timestamp("2026-01-15")
    assert set(trades["currency"]) == {"USD"}                 # default currency applied


def test_unknown_action_is_reported_with_line_number():
    bad = FAKE_CSV.replace("SLD", "SHORT")
    with pytest.raises(ValueError, match=r"unknown action values \['short'\] on lines 4"):
        normalize_trades(read_csv(io.StringIO(bad)), FAKE_BROKER)


def test_missing_required_column_is_reported():
    bad = FAKE_CSV.replace("Fill Price", "Price")
    with pytest.raises(ValueError, match="missing columns"):
        normalize_trades(read_csv(io.StringIO(bad)), FAKE_BROKER)


def test_oversold_position_is_flagged():
    oversold = FAKE_CSV.replace("SLD,-4", "SLD,-15")
    trades = normalize_trades(read_csv(io.StringIO(oversold)), FAKE_BROKER)
    assert find_position_problems(trades) == ["Main AAPL: position falls to -5 on 2026-02-01"]


def test_sample_portfolio_has_balanced_cash_ledger():
    data = load_standard_portfolio(PROJECT_ROOT / "data" / "sample_trades.csv",
                                   PROJECT_ROOT / "data" / "sample_cash_flows.csv")
    assert data.warnings == []
    assert data.accounts == ["Main"]
    external = data.cash_ledger[data.cash_ledger["kind"] == "external"]
    assert external["amount"].sum() == 56_000      # 50k + 10k deposits - 4k withdrawal


def test_trade_cash_includes_fees():
    trades = fake_trades()
    # buy 10 @ 1200.50 + 1.00 fee; sell 4 @ 1250 - 1.00 fee
    assert list(trade_cash(trades)) == [-12_006.0, 4_999.0]


def test_stop_overrides_fill_buys_only():
    trades = fake_trades()
    stops = io.StringIO("date,ticker,stop_price\n2026-01-15,AAPL,1100\n2026-02-01,AAPL,999\n")
    filled = apply_stop_overrides(trades, stops)
    assert filled.loc[0, "stop_price"] == 1100          # the buy gets your stop
    assert pd.isna(filled.loc[1, "stop_price"])          # a sell is not an entry
