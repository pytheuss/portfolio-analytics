"""Adapter for Wealthsimple's Activities export.

Get it from Wealthsimple: Documents > Request documents > Activities export (CSV).

The export mixes trades, options, corporate actions, dividends, interest and
transfers in one file, and several rows only make sense with context, so it
gets a small adapter instead of a one-line BrokerMapping. The output is the
same standard trades + cash ledger every other broker produces.

Wealthsimple conventions handled here:
- The trade date is the "executed at" date in the description; effective_date
  can be the day the order was placed (e.g. a weekend).
- There's no exchange column. A trade in USD, or with an "FX Rate" in its
  description, is a US listing (registered accounts settle US trades in CAD).
  A CAD trade with no FX rate is a Canadian listing, whose exchange suffix
  (.TO, .V, ...) comes from the holdings report or a yfinance lookup.
- Options use OCC symbols with a space: "INFQ  260515C00015000". Price and
  quantity are per contract.
- Corporate actions, option expiries and the option leg of an exercise change
  quantities with no cash: they become "adjust" rows.
- Every row that moves cash carries net_cash_amount, which becomes the cash
  ledger. MoneyMovement rows (deposits, withdrawals, transfers) are external.
"""

import re

import numpy as np
import pandas as pd

from src.ingest import (
    LEDGER_COLUMNS,
    TRADE_COLUMNS,
    HoldingsSnapshot,
    PortfolioData,
    find_position_problems,
    infer_opening_cash,
    infer_opening_positions,
    read_csv,
    sort_trades,
    to_date,
    to_number,
    validate_trades,
)

# Wealthsimple Cash is a bank account, not part of the investment portfolio.
EXCLUDED_ACCOUNT_TYPES = {"Chequing"}

# Activity types that change how many shares or contracts you hold.
QUANTITY_ACTIVITIES = {"Trade", "OptionExercise", "OptionExpiry", "CorporateAction"}
# Money moving in or out of the portfolio (excluded from returns by time-weighting).
EXTERNAL_ACTIVITIES = {"MoneyMovement"}

OPTION_SYMBOL = re.compile(r"^\s*([A-Z0-9.\-]+)\s+(\d{6}[CP]\d{8})\s*$")
EXECUTED_AT = re.compile(r"executed at (\d{4}-\d{2}-\d{2})")
CANADIAN_SUFFIXES = (".TO", ".V", ".NE", ".CN")


def read_activities(source) -> pd.DataFrame:
    """Read the export, drop non-investment accounts and add a clean trade date."""
    raw = read_csv(source)
    raw = raw[~raw["account_type"].isin(EXCLUDED_ACCOUNT_TYPES)].copy()

    executed = raw["description"].astype(str).str.extract(EXECUTED_AT)[0]
    raw["date"] = to_date(executed.fillna(raw["effective_date"]))
    # When Wealthsimple recorded the row; used to line up with a mid-day holdings report.
    raw["timestamp"] = pd.to_datetime(raw["effective_date"] + " " + raw["effective_time"].fillna("00:00:00"))
    raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper().replace("NAN", np.nan)
    for col in ["quantity", "unit_price", "commission", "net_cash_amount"]:
        raw[col] = to_number(raw[col])
    return raw


def is_option(symbols: pd.Series) -> pd.Series:
    return symbols.fillna("").str.match(OPTION_SYMBOL)


def option_ticker(symbol: str) -> str:
    """'INFQ  260515C00015000' -> 'INFQ260515C00015000' (the format yfinance uses)."""
    underlying, contract = OPTION_SYMBOL.match(symbol).groups()
    return underlying + contract


def us_listed_symbols(raw: pd.DataFrame) -> set[str]:
    """Symbols traded in USD or with an FX conversion: US listings."""
    priced = raw[raw["activity_type"].isin(QUANTITY_ACTIVITIES) & raw["currency"].notna()]
    foreign = (priced["currency"] == "USD") | priced["description"].str.contains("FX Rate", na=False)
    return set(priced.loc[foreign, "symbol"])


def canadian_symbols_needing_lookup(raw: pd.DataFrame, snapshot: HoldingsSnapshot | None) -> set[str]:
    """Canadian stock symbols whose exchange suffix isn't known from the file or snapshot."""
    stocks = raw[raw["activity_type"].isin(QUANTITY_ACTIVITIES) & ~is_option(raw["symbol"])]
    symbols = set(stocks["symbol"].dropna())
    known = set(snapshot.positions["symbol"].str.upper()) if snapshot else set()
    return {s for s in symbols - us_listed_symbols(raw) - known
            if not s.endswith(CANADIAN_SUFFIXES)}


def build_ticker_map(raw: pd.DataFrame, snapshot: HoldingsSnapshot | None,
                     canadian_lookup: dict[str, str]) -> dict[str, str]:
    """Wealthsimple symbol -> yfinance ticker for every stock symbol in the file.

    Priority: a suffix already in the symbol (HIVE.TO), then the holdings
    report's exchange code, then US listing, then the Canadian lookup table.
    Symbols that can't be resolved are left out (and reported by the caller).
    """
    from_snapshot = {}
    if snapshot is not None:
        from_snapshot = dict(zip(snapshot.positions["symbol"].str.upper(),
                                 snapshot.positions["ticker"], strict=True))
    us = us_listed_symbols(raw)

    tickers = {}
    for symbol in set(raw["symbol"].dropna()) - set(raw.loc[is_option(raw["symbol"]), "symbol"]):
        if symbol.endswith(CANADIAN_SUFFIXES):
            base, suffix = symbol.rsplit(".", 1)
            tickers[symbol] = f"{base.replace('.', '-')}.{suffix}"
        elif symbol in from_snapshot:
            tickers[symbol] = from_snapshot[symbol]
        elif symbol in us:
            tickers[symbol] = symbol.replace(".", "-")
        elif symbol in canadian_lookup:
            tickers[symbol] = canadian_lookup[symbol]
    return tickers


def _actions(rows: pd.DataFrame) -> np.ndarray:
    """buy / sell / adjust for each quantity-changing row."""
    sub = rows["activity_sub_type"].astype(str).str.upper()
    cash = rows["net_cash_amount"]
    return np.select(
        [
            (rows["activity_type"] == "Trade") & (sub == "BUY"),
            (rows["activity_type"] == "Trade") & (sub == "SELL"),
            cash.isna(),     # no money changed hands: split, expiry, option leg of exercise
            cash < 0,        # e.g. shares bought by exercising a call
        ],
        ["buy", "sell", "adjust", "buy"],
        default="sell",
    )


def build_trades(raw: pd.DataFrame, tickers: dict[str, str]) -> pd.DataFrame:
    """Standard trades from every row that changes a share or contract count."""
    rows = raw[raw["activity_type"].isin(QUANTITY_ACTIVITIES) & raw["quantity"].notna()]
    options = is_option(rows["symbol"])
    action = _actions(rows)
    is_adjust = action == "adjust"

    trades = pd.DataFrame(index=rows.index)
    trades["date"] = rows["date"]
    trades["timestamp"] = rows["timestamp"]
    trades["account"] = rows["account_type"].astype(str).str.strip()
    trades["ticker"] = [option_ticker(s) if opt else tickers.get(s, s)
                        for s, opt in zip(rows["symbol"], options, strict=True)]
    trades["asset_type"] = np.where(options, "option", "stock")
    trades["action"] = action
    # Buys and sells store size as a positive number; adjustments keep their sign.
    trades["quantity"] = np.where(is_adjust, rows["quantity"], rows["quantity"].abs())
    trades["price"] = np.where(is_adjust, np.nan, rows["unit_price"].abs())
    trades["currency"] = rows["currency"]
    trades["fees"] = rows["commission"].abs().fillna(0.0)
    trades["stop_price"] = np.nan

    validate_trades(trades)
    # Within a day, keep Wealthsimple's time order (a buy and sell of the same stock).
    trades = trades.loc[rows.sort_values(["date", "effective_time"], kind="mergesort").index]
    return sort_trades(trades)


def build_cash_ledger(raw: pd.DataFrame) -> pd.DataFrame:
    """Every row with a net cash amount, tagged external (transfers) or internal."""
    rows = raw[raw["net_cash_amount"].notna()]
    ledger = pd.DataFrame({
        "date": rows["date"],
        "timestamp": rows["timestamp"],
        "account": rows["account_type"].astype(str).str.strip(),
        "currency": rows["currency"].astype(str).str.strip().str.upper(),
        "amount": rows["net_cash_amount"],
        "kind": np.where(rows["activity_type"].isin(EXTERNAL_ACTIVITIES), "external", "internal"),
        "activity": rows["activity_type"],
    })
    return ledger[LEDGER_COLUMNS].sort_values("date", kind="mergesort").reset_index(drop=True)


def load_wealthsimple(activities_source, snapshot: HoldingsSnapshot | None = None,
                      canadian_lookup: dict[str, str] | None = None,
                      snapshot_cutoff: pd.Timestamp | None = None) -> PortfolioData:
    """Turn a Wealthsimple Activities export into standardized portfolio data.

    snapshot: a holdings report. If the history starts after the accounts were
        opened, it is used to back out positions and cash held at the start.
    canadian_lookup: symbol -> yfinance ticker for Canadian symbols whose
        exchange isn't known (see prices.resolve_canadian_tickers).
    snapshot_cutoff: the moment the holdings report actually reflects, if it
        differs from its footer (see config.yaml).
    """
    raw = read_activities(activities_source)
    tickers = build_ticker_map(raw, snapshot, canadian_lookup or {})
    trades = build_trades(raw, tickers)
    ledger = build_cash_ledger(raw)

    warnings = []
    stock_symbols = set(raw.loc[raw["activity_type"].isin(QUANTITY_ACTIVITIES)
                                & ~is_option(raw["symbol"]), "symbol"].dropna())
    unresolved = sorted(stock_symbols - set(tickers))
    if unresolved:
        warnings.append(f"No exchange found for {unresolved}; prices will be missing. "
                        "Add them to data/private/ticker_map.csv.")

    if snapshot is not None:
        opening_positions = infer_opening_positions(trades, snapshot, snapshot_cutoff)
        opening_cash = infer_opening_cash(ledger, snapshot, snapshot_cutoff)
    else:
        opening_positions = pd.DataFrame(columns=["account", "ticker", "quantity"])
        opening_cash = pd.DataFrame(columns=["account", "currency", "amount"])
    warnings += find_position_problems(trades, opening_positions)

    return PortfolioData(
        trades=trades[TRADE_COLUMNS],
        cash_ledger=ledger,
        opening_positions=opening_positions,
        opening_cash=opening_cash,
        start_date=min(trades["date"].min(), ledger["date"].min()),
        warnings=warnings,
        snapshot=snapshot,
    )
