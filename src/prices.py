"""Fetch and cache price and FX data from yfinance.

Which price to use matters:
- yfinance "Adj Close" folds dividends into the price. Our cash ledger already
  records dividends as cash, so using it would count every dividend twice.
- yfinance "Close" is still adjusted for stock splits: before VGT's 8-for-1
  split, it shows about 1/8 of the real price. Our trade history records the
  real share counts (5 shares, then 40 after the split), so we undo the split
  adjustment to get the price you actually saw that day.
"""

import logging
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PROJECT_ROOT

# yfinance logs an error for every ticker that doesn't exist; that's expected here.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# yfinance still uses pandas features deprecated in pandas 3; harmless for us.
warnings.filterwarnings("ignore", module="yfinance")

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
TICKER_MAP_PATH = PROJECT_ROOT / "data" / "private" / "ticker_map.csv"
# Order matters: primary listing exchanges (TSX, TSX Venture, CSE) before Cboe
# Canada, which also quotes many CSE stocks as a secondary market.
CANADIAN_SUFFIXES = [".TO", ".V", ".CN", ".NE"]


# ---------------------------------------------------------------------------
# Daily closes
# ---------------------------------------------------------------------------

def ticker_currency(ticker: str) -> str:
    """Trading currency of a stock ticker: Canadian exchange suffix -> CAD, else USD."""
    return "CAD" if ticker.endswith(tuple(CANADIAN_SUFFIXES)) else "USD"


def unsplit(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo yfinance's split adjustment so prices match the shares actually held.

    A split of ratio r on day s means yfinance divided every close before s by r.
    So each close is multiplied by the product of all split ratios after its date.
    """
    ratios = splits.reindex(close.index).fillna(0.0).replace(0.0, 1.0)
    later_splits = ratios[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * later_splits


def _cache_path(ticker: str, start: pd.Timestamp) -> Path:
    """One file per ticker and start date, e.g. data/cache/AAPL_20251215.csv."""
    safe = ticker.replace("^", "IDX_").replace("=", "_")
    return CACHE_DIR / f"{safe}_{start:%Y%m%d}.csv"


def _is_fresh(path: Path) -> bool:
    """Cache files are reused for the rest of the day they were downloaded."""
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()


def download_history(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Raw yfinance download: columns (field, ticker), e.g. ("Close", "AAPL")."""
    return yf.download(tickers, start=start, end=end + pd.Timedelta(days=1), auto_adjust=False,
                       actions=True, progress=False, threads=True)


def _field(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    try:
        series = data[field][ticker].dropna()
    except KeyError:
        return pd.Series(dtype=float)
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    return series


def _price_table(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """high, low, close (split-unadjusted) and adj_close (total return) for one ticker."""
    splits = _field(data, "Stock Splits", ticker)
    table = pd.DataFrame({
        name: unsplit(_field(data, field, ticker), splits)
        for name, field in [("high", "High"), ("low", "Low"), ("close", "Close")]
    })
    table["adj_close"] = _field(data, "Adj Close", ticker)
    return table


def fetch_prices(tickers, start, end=None) -> dict[str, pd.DataFrame]:
    """Daily high/low/close/adj_close per ticker, downloaded once and cached for the day.

    Tickers with no data (delisted, or unknown to yfinance) come back as an
    empty table, so callers can fall back to trade prices.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end or date.today())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tables, to_download = {}, []
    for ticker in sorted(set(tickers)):
        path = _cache_path(ticker, start)
        if _is_fresh(path):
            tables[ticker] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            to_download.append(ticker)

    if to_download:
        data = download_history(to_download, start, end)
        for ticker in to_download:
            table = _price_table(data, ticker)
            table.to_csv(_cache_path(ticker, start), index_label="date")
            tables[ticker] = table

    return {t: table.loc[(table.index >= start) & (table.index <= end)] for t, table in tables.items()}


def fetch_closes(tickers, start, end=None, total_return: bool = False) -> pd.DataFrame:
    """Daily closes, one column per ticker.

    total_return=False: the price you'd have seen that day (split-unadjusted, no
        dividends). Used to value holdings, whose dividends are in the cash ledger.
    total_return=True: "Adj Close", which reinvests dividends. Used for the
        benchmark ETFs, which have no cash ledger of their own.
    """
    column = "adj_close" if total_return else "close"
    tables = fetch_prices(tickers, start, end)
    return pd.DataFrame({t: table[column] for t, table in tables.items()}).sort_index()


# ---------------------------------------------------------------------------
# Sector and country (for exposure)
# ---------------------------------------------------------------------------

INFO_PATH = CACHE_DIR / "security_info.csv"


def _lookup_info(ticker: str) -> dict:
    """Sector, country and security type from yfinance; blanks if unavailable."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {"ticker": ticker, "name": info.get("shortName") or info.get("longName"),
            "quote_type": info.get("quoteType"), "sector": info.get("sector"),
            "country": info.get("country")}


def fetch_security_info(tickers) -> pd.DataFrame:
    """Per ticker: name, quote_type, sector, country. Cached permanently (sectors rarely change).

    Delete data/cache/security_info.csv to refresh.
    """
    cached = pd.read_csv(INFO_PATH) if INFO_PATH.exists() else pd.DataFrame(columns=["ticker"])
    missing = sorted(set(tickers) - set(cached["ticker"]))
    if missing:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = pd.concat([cached, pd.DataFrame([_lookup_info(t) for t in missing])],
                           ignore_index=True)
        cached.to_csv(INFO_PATH, index=False)
    return cached.set_index("ticker").reindex(sorted(set(tickers)))


# ---------------------------------------------------------------------------
# Canadian exchange lookup
# ---------------------------------------------------------------------------
# The Wealthsimple activities export doesn't say which Canadian exchange a bare
# symbol like "FLT" trades on, so we ask yfinance.

def read_ticker_map(path: Path = TICKER_MAP_PATH) -> dict[str, str]:
    """Symbol -> yfinance ticker, from an editable two-column CSV."""
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return dict(zip(table["symbol"].str.upper(), table["ticker"], strict=True))


def write_ticker_map(mapping: dict[str, str], path: Path = TICKER_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(sorted(mapping.items()), columns=["symbol", "ticker"])
    table.to_csv(path, index=False)


def has_price_history(ticker: str) -> bool:
    """True if yfinance returns any daily prices for this ticker in the last two years."""
    try:
        return not yf.Ticker(ticker).history(period="2y", auto_adjust=False).empty
    except Exception:
        return False


def resolve_canadian_tickers(symbols, path: Path = TICKER_MAP_PATH) -> dict[str, str]:
    """Find each symbol's exchange by trying .TO, .V, .CN, .NE until one has prices.

    Results are cached in data/private/ticker_map.csv, so each symbol is looked
    up once. Edit that file to fix a wrong match; edits are never overwritten.
    """
    mapping = read_ticker_map(path)
    for symbol in sorted(set(symbols) - set(mapping)):
        base = symbol.replace(".", "-")
        for suffix in CANADIAN_SUFFIXES:
            if has_price_history(base + suffix):
                mapping[symbol] = base + suffix
                break
    write_ticker_map(mapping, path)
    return mapping
