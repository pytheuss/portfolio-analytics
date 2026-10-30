"""Load portfolio data from the sample files, your files in data/private/, or uploads."""

import io

import pandas as pd

from src.config import PROJECT_ROOT
from src.ingest import (
    PortfolioData,
    apply_stop_overrides,
    load_holdings,
    load_standard_portfolio,
    read_csv,
)
from src.prices import resolve_canadian_tickers
from src.wealthsimple import canadian_symbols_needing_lookup, load_wealthsimple, read_activities

# Optional: stop-loss levels you record yourself (date, ticker, stop_price[, account]).
STOPS_PATH = PROJECT_ROOT / "data" / "private" / "stops.csv"
# Columns that identify a Wealthsimple activities export.
WEALTHSIMPLE_COLUMNS = {"activity_type", "activity_sub_type", "net_cash_amount"}


def load_sample(config: dict) -> PortfolioData:
    """The fake sample portfolio that ships with the project."""
    files = config["data"]
    return load_standard_portfolio(PROJECT_ROOT / files["sample_trades"],
                                   PROJECT_ROOT / files["sample_cash_flows"])


def private_files_available(config: dict) -> bool:
    """True if the Wealthsimple files named in config.yaml exist in data/private/."""
    files = config.get("wealthsimple", {})
    return all((PROJECT_ROOT / files.get(key, "missing")).exists()
               for key in ["activities_export", "holdings_report"])


def _wealthsimple(open_activities, snapshot, cutoff) -> PortfolioData:
    """Shared Wealthsimple path. open_activities() returns a fresh file each call."""
    # Look up exchanges for Canadian symbols the files don't identify (cached after the first run).
    needed = canadian_symbols_needing_lookup(read_activities(open_activities()), snapshot)
    lookup = resolve_canadian_tickers(needed)
    data = load_wealthsimple(open_activities(), snapshot, lookup, snapshot_cutoff=cutoff)
    if STOPS_PATH.exists():
        data.trades = apply_stop_overrides(data.trades, STOPS_PATH)
    return data


def load_configured_wealthsimple(config: dict) -> PortfolioData:
    """Your Wealthsimple activities export plus holdings report, from data/private/."""
    files = config["wealthsimple"]
    activities = PROJECT_ROOT / files["activities_export"]
    snapshot = load_holdings(PROJECT_ROOT / files["holdings_report"])
    cutoff = pd.Timestamp(files["holdings_snapshot_cutoff"])
    return _wealthsimple(lambda: activities, snapshot, cutoff)


def load_uploaded(trades: bytes, holdings: bytes | None = None, cash_flows: bytes | None = None,
                  cutoff: str | None = None) -> PortfolioData:
    """Files uploaded in the app, as raw bytes.

    A Wealthsimple activities export is recognised by its columns; anything else
    is read as a simple trade file (standard format or a registered BrokerMapping).
    """
    columns = set(read_csv(io.BytesIO(trades)).columns)
    if WEALTHSIMPLE_COLUMNS <= columns:
        snapshot = load_holdings(io.BytesIO(holdings)) if holdings else None
        return _wealthsimple(lambda: io.BytesIO(trades), snapshot,
                             pd.Timestamp(cutoff) if cutoff else None)
    flows = io.BytesIO(cash_flows) if cash_flows else None
    return load_standard_portfolio(io.BytesIO(trades), flows)
