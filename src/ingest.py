"""Load broker exports and normalize them into the app's standard formats.

Every other module only ever sees these formats, whichever broker the data came from:

trades
    date | timestamp | account | ticker | asset_type | action | quantity | price | currency | fees | stop_price

    action is "buy", "sell" or "adjust". An adjustment changes the share count
    with no cash changing hands (stock split, consolidation, option expiry), so
    its quantity is signed and it has no price.
    asset_type is "stock" (stocks and ETFs) or "option".
    stop_price is optional; fill it in to drive the position-sizing check.

cash ledger
    date | timestamp | account | currency | amount | kind | activity

    date is the trade date used for pricing. timestamp is when the broker
    recorded the row (date + time where the export has a time, else midnight);
    it is only used to line rows up against a holdings snapshot taken mid-day.

    One row per cash movement. kind is "external" for money you deposit or
    withdraw (time-weighted returns strip these out) or "internal" for
    everything that is part of the return: trades, dividends, interest, tax.

Simple brokers are described by a BrokerMapping ("your column X is our column
Y"). Brokers whose exports need more interpretation get their own adapter
module (see src/wealthsimple.py) that outputs the same formats.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

TRADE_COLUMNS = ["date", "timestamp", "account", "ticker", "asset_type", "action",
                 "quantity", "price", "currency", "fees", "stop_price"]
LEDGER_COLUMNS = ["date", "timestamp", "account", "currency", "amount", "kind", "activity"]
REQUIRED_COLUMNS = ["date", "ticker", "action", "quantity", "price"]
SUPPORTED_CURRENCIES = {"CAD", "USD"}
DEFAULT_ACCOUNT = "Main"

# Kept for readability elsewhere: the standard trade format.
STANDARD_COLUMNS = TRADE_COLUMNS


# ---------------------------------------------------------------------------
# Generic trade files (declarative column mapping)
# ---------------------------------------------------------------------------

@dataclass
class BrokerMapping:
    """How one broker's trade export lines up with the standard trade format.

    columns:     standard column name -> broker column name (matched case-insensitively).
                 Only REQUIRED_COLUMNS must be present; the rest are optional.
    actions:     broker action value -> "buy" or "sell" (matched case-insensitively).
    row_filter:  keep only rows where broker column's value is in the list,
                 e.g. {"activity_type": ["Trade"]} drops dividends and deposits.
    default_currency: used when the export has no currency column.
    date_format: strftime pattern if dates aren't ISO (e.g. "%m/%d/%Y").
    """

    name: str
    columns: dict[str, str]
    actions: dict[str, str]
    row_filter: dict[str, list[str]] = field(default_factory=dict)
    default_currency: str | None = None
    date_format: str | None = None


# Registry of brokers with simple trade files. To add one, add an entry here.
BROKER_MAPPINGS: dict[str, BrokerMapping] = {
    # The app's own format, used by data/sample_trades.csv.
    "standard": BrokerMapping(
        name="Standard format",
        columns={col: col for col in TRADE_COLUMNS},
        actions={"buy": "buy", "sell": "sell"},
    ),
}


def read_csv(source) -> pd.DataFrame:
    """Read a CSV (path or uploaded file) with tidy, lower-case headers."""
    # utf-8-sig strips the invisible byte-order mark some exports start with.
    raw = pd.read_csv(source, encoding="utf-8-sig")
    raw.columns = [str(col).strip().lower() for col in raw.columns]
    return raw


def detect_broker(columns) -> BrokerMapping:
    """Return the first mapping whose required columns all appear in the file."""
    present = {str(col).strip().lower() for col in columns}
    for mapping in BROKER_MAPPINGS.values():
        needed = {mapping.columns[col].lower() for col in REQUIRED_COLUMNS}
        needed |= {col.lower() for col in mapping.row_filter}
        if needed <= present:
            return mapping
    known = ", ".join(m.name for m in BROKER_MAPPINGS.values())
    raise ValueError(
        f"Couldn't recognise this file's columns {sorted(present)}. "
        f"Known formats: {known}. Add a BrokerMapping in src/ingest.py."
    )


def _filter_rows(raw: pd.DataFrame, mapping: BrokerMapping) -> pd.DataFrame:
    """Keep only trade rows (drops dividends, deposits, fees, etc.)."""
    keep = pd.Series(True, index=raw.index)
    for column, allowed in mapping.row_filter.items():
        values = raw[column.lower()].astype(str).str.strip().str.lower()
        keep &= values.isin([a.lower() for a in allowed])
    return raw[keep]


def to_number(series: pd.Series) -> pd.Series:
    """Parse numbers like '1,234.50' or '$12.00'; unreadable values become NaN.

    Always returns floats, since quantities can be fractional shares.
    """
    cleaned = series.astype(str).str.replace(r"[,$\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def to_date(series: pd.Series, date_format: str | None = None) -> pd.Series:
    """Parse dates and drop any time-of-day or timezone, keeping the calendar day."""
    dates = pd.to_datetime(series, format=date_format, errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _column(raw: pd.DataFrame, mapping: BrokerMapping, standard_name: str):
    """Return the broker's column for a standard field, or None if it isn't mapped."""
    broker_name = mapping.columns.get(standard_name)
    if broker_name is None or broker_name.lower() not in raw.columns:
        return None
    return raw[broker_name.lower()]


def _text_or_default(raw, mapping, standard_name, default):
    """A cleaned text column, or a constant if the broker doesn't provide it."""
    column = _column(raw, mapping, standard_name)
    return default if column is None else column.astype(str).str.strip()


def normalize_trades(raw: pd.DataFrame, mapping: BrokerMapping) -> pd.DataFrame:
    """Convert a broker's raw rows into the standard trade format, validated and sorted."""
    missing = [c for c in REQUIRED_COLUMNS if _column(raw, mapping, c) is None]
    if missing:
        wanted = [mapping.columns.get(c, c) for c in missing]
        raise ValueError(f"{mapping.name} file is missing columns: {wanted}")

    raw = _filter_rows(raw, mapping)
    trades = pd.DataFrame(index=raw.index)

    trades["date"] = to_date(_column(raw, mapping, "date"), mapping.date_format)
    trades["timestamp"] = trades["date"]
    trades["account"] = _text_or_default(raw, mapping, "account", DEFAULT_ACCOUNT)
    trades["ticker"] = _column(raw, mapping, "ticker").astype(str).str.strip().str.upper()
    trades["asset_type"] = _text_or_default(raw, mapping, "asset_type", "stock")

    action_lookup = {k.lower(): v for k, v in mapping.actions.items()}
    raw_action = _column(raw, mapping, "action").astype(str).str.strip().str.lower()
    trades["action"] = raw_action.map(action_lookup)

    # Some brokers show sells as negative quantities; direction lives in `action`.
    trades["quantity"] = to_number(_column(raw, mapping, "quantity")).abs()
    trades["price"] = to_number(_column(raw, mapping, "price")).abs()

    currency = _text_or_default(raw, mapping, "currency", mapping.default_currency)
    trades["currency"] = currency.str.upper() if isinstance(currency, pd.Series) else currency

    fees = _column(raw, mapping, "fees")
    trades["fees"] = 0.0 if fees is None else to_number(fees).abs().fillna(0.0)

    stop = _column(raw, mapping, "stop_price")
    trades["stop_price"] = float("nan") if stop is None else to_number(stop)

    validate_trades(trades, raw_action)
    return sort_trades(trades)


def sort_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Order by date; the stable sort keeps same-day trades in file order."""
    return trades[TRADE_COLUMNS].sort_values("date", kind="mergesort").reset_index(drop=True)


def _line_numbers(mask: pd.Series) -> str:
    """CSV line numbers for flagged rows (+2: header line and 1-based counting)."""
    lines = [str(i + 2) for i in mask[mask].index[:10]]
    more = " ..." if mask.sum() > 10 else ""
    return ", ".join(lines) + more


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().isin(["", "nan", "NAN", "None"])


def validate_trades(trades: pd.DataFrame, raw_action: pd.Series | None = None) -> None:
    """Raise one ValueError listing every problem, so the file can be fixed in one pass."""
    problems = []
    is_trade = trades["action"].isin(["buy", "sell"])
    is_adjust = trades["action"] == "adjust"

    unknown = trades["action"].isna()
    if unknown.any():
        values = sorted(set(raw_action[unknown])) if raw_action is not None else []
        problems.append(f"unknown action values {values} on lines {_line_numbers(unknown)}")

    for col in ["date", "ticker", "quantity"]:
        missing = _is_blank(trades[col])
        if missing.any():
            problems.append(f"missing or unreadable {col} on lines {_line_numbers(missing)}")

    # Buys and sells move cash, so they need a price and currency; adjustments don't.
    for col in ["price", "currency"]:
        missing = is_trade & _is_blank(trades[col])
        if missing.any():
            problems.append(f"missing or unreadable {col} on lines {_line_numbers(missing)}")

    for col in ["quantity", "price"]:
        not_positive = is_trade & (trades[col] <= 0)
        if not_positive.any():
            problems.append(f"{col} must be positive on lines {_line_numbers(not_positive)}")

    zero_adjust = is_adjust & (trades["quantity"] == 0)
    if zero_adjust.any():
        problems.append(f"adjustment of zero shares on lines {_line_numbers(zero_adjust)}")

    bad_ccy = is_trade & trades["currency"].notna() & ~trades["currency"].isin(SUPPORTED_CURRENCIES)
    if bad_ccy.any():
        problems.append(
            f"unsupported currency {sorted(set(trades['currency'][bad_ccy]))} "
            f"on lines {_line_numbers(bad_ccy)} (supported: {sorted(SUPPORTED_CURRENCIES)})"
        )

    if problems:
        raise ValueError("Trade file failed validation:\n- " + "\n- ".join(problems))


def load_trades(source, broker: str | None = None) -> pd.DataFrame:
    """Read a simple broker trade CSV and return standardized trades.

    broker: a key in BROKER_MAPPINGS, or None to detect it from the headers.
    """
    raw = read_csv(source)
    mapping = BROKER_MAPPINGS[broker] if broker else detect_broker(raw.columns)
    return normalize_trades(raw, mapping)


# ---------------------------------------------------------------------------
# Cash: deposits/withdrawals file and the cash ledger
# ---------------------------------------------------------------------------

def load_cash_flows(source) -> pd.DataFrame:
    """Read deposits (+) and withdrawals (-): columns date, amount, currency[, account]."""
    raw = read_csv(source)
    dates = to_date(raw["date"])
    return pd.DataFrame({
        "date": dates,
        "timestamp": dates,
        "account": raw["account"].astype(str).str.strip() if "account" in raw else DEFAULT_ACCOUNT,
        "currency": raw["currency"].astype(str).str.strip().str.upper(),
        "amount": to_number(raw["amount"]),
        "kind": "external",
        "activity": "Deposit/Withdrawal",
    })


def trade_cash(trades: pd.DataFrame) -> pd.Series:
    """Cash effect of each trade: buys cost price x qty + fees, sells bring in price x qty - fees."""
    gross = trades["quantity"] * trades["price"]
    cash = (-gross - trades["fees"]).where(trades["action"] == "buy", gross - trades["fees"])
    return cash.where(trades["action"] != "adjust", 0.0)


def build_cash_ledger(trades: pd.DataFrame, cash_flows: pd.DataFrame) -> pd.DataFrame:
    """Ledger for brokers that only export trades: trade cash plus the deposits file."""
    moved_cash = trades[trades["action"] != "adjust"]
    from_trades = pd.DataFrame({
        "date": moved_cash["date"],
        "timestamp": moved_cash["timestamp"],
        "account": moved_cash["account"],
        "currency": moved_cash["currency"],
        "amount": trade_cash(moved_cash),
        "kind": "internal",
        "activity": "Trade",
    })
    ledger = pd.concat([cash_flows[LEDGER_COLUMNS], from_trades], ignore_index=True)
    return ledger.sort_values("date", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Holdings snapshots
# ---------------------------------------------------------------------------
# A holdings report is a point-in-time snapshot of what you own, as opposed to
# the trade history above. It is the source of truth for current positions, and
# lets us work out what was held before the trade history starts.

HOLDINGS_COLUMNS = ["account", "symbol", "ticker", "name", "security_type",
                    "quantity", "price", "currency", "book_value_cad", "market_value"]

# Exchange MIC code -> yfinance ticker suffix. US exchanges need no suffix.
MIC_TO_YFINANCE_SUFFIX = {
    "XTSE": ".TO",  # Toronto Stock Exchange
    "XTSX": ".V",   # TSX Venture Exchange
    "NEOE": ".NE",  # Cboe Canada (formerly NEO)
    "XCNQ": ".CN",  # Canadian Securities Exchange
    "XNYS": "",     # NYSE
    "XNAS": "",     # Nasdaq
    "ARCX": "",     # NYSE Arca
    "BATS": "",     # Cboe BZX
    "XASE": "",     # NYSE American
}


def to_yfinance_ticker(symbol: str, mic: str) -> str:
    """Broker symbol + exchange -> yfinance ticker, e.g. VEQT on XTSE -> VEQT.TO.

    yfinance writes share classes with a dash, so BRK.B becomes BRK-B.
    """
    symbol = str(symbol).strip().upper().replace(".", "-")
    suffix = MIC_TO_YFINANCE_SUFFIX.get(str(mic).strip().upper())
    if suffix is None:
        raise ValueError(
            f"Unknown exchange code {mic!r} for {symbol}. Add it to MIC_TO_YFINANCE_SUFFIX."
        )
    return symbol + suffix


@dataclass
class HoldingsMapping:
    """How one broker's holdings report lines up with the standard holdings format.

    columns: standard name -> broker column. Must include "symbol" and "mic"
             (used to build the yfinance ticker) and "direction" (LONG/SHORT).
    cash_security_type: the security-type value that marks a cash balance row.
    """

    name: str
    columns: dict[str, str]
    cash_security_type: str


HOLDINGS_MAPPINGS: dict[str, HoldingsMapping] = {
    "wealthsimple": HoldingsMapping(
        name="Wealthsimple holdings report",
        columns={
            # Account type (TFSA, RRSP, ...) is the label shared with the activities export.
            "account": "account type",
            "symbol": "symbol",
            "mic": "mic",
            "name": "name",
            "security_type": "security type",
            "quantity": "quantity",
            "direction": "position direction",
            "price": "market price",
            "currency": "market price currency",
            "book_value_cad": "book value (cad)",
            "market_value": "market value",
        },
        cash_security_type="CURRENCY",
    ),
}


@dataclass
class HoldingsSnapshot:
    """Positions and cash balances as of one date."""

    as_of: pd.Timestamp | None
    positions: pd.DataFrame  # HOLDINGS_COLUMNS, one row per account x security
    cash: pd.DataFrame       # account, currency, amount (negative = margin borrowing)


def _split_footer(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Remove a trailing 'As of 2026-06-02 16:00 ...' line and return its date and time."""
    first_col = raw.iloc[:, 0].astype(str)
    is_footer = first_col.str.startswith("As of") & raw.iloc[:, 1:].isna().all(axis=1)
    as_of = None
    if is_footer.any():
        match = re.search(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?", first_col[is_footer].iloc[0])
        as_of = pd.Timestamp(match.group()) if match else None
    return raw[~is_footer], as_of


def load_holdings(source, broker: str = "wealthsimple") -> HoldingsSnapshot:
    """Read a holdings report and split it into positions and cash balances."""
    mapping = HOLDINGS_MAPPINGS[broker]
    raw, as_of = _split_footer(read_csv(source))

    missing = [col for col in mapping.columns.values() if col not in raw.columns]
    if missing:
        raise ValueError(f"{mapping.name} is missing columns: {missing}")

    df = pd.DataFrame({std: raw[col] for std, col in mapping.columns.items()})
    for col in ["account", "symbol", "mic", "name", "security_type", "direction", "currency"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["quantity", "price", "book_value_cad", "market_value"]:
        df[col] = to_number(df[col])

    # Short positions are stored as negative quantities.
    df["quantity"] = df["quantity"].where(df["direction"].str.upper() != "SHORT", -df["quantity"])

    is_cash = df["security_type"].str.upper() == mapping.cash_security_type
    cash = pd.DataFrame({
        "account": df.loc[is_cash, "account"],
        "currency": df.loc[is_cash, "symbol"],
        "amount": df.loc[is_cash, "quantity"],
    }).reset_index(drop=True)

    positions = df[~is_cash].copy()
    positions["ticker"] = [to_yfinance_ticker(s, m)
                           for s, m in zip(positions["symbol"], positions["mic"], strict=True)]
    positions = positions[HOLDINGS_COLUMNS].reset_index(drop=True)

    return HoldingsSnapshot(as_of=as_of, positions=positions, cash=cash)


# ---------------------------------------------------------------------------
# Positions held before the trade history starts
# ---------------------------------------------------------------------------
# If the trade history starts after an account was opened, some positions and
# cash already existed on day one. A holdings snapshot lets us back them out:
#
#     held at start = held on snapshot date - net change between start and snapshot
#
# The same subtraction works for cash, using the cash ledger. "Between" means
# rows recorded up to the moment the report was taken. That can be mid-day, so
# it is compared on timestamp, not just date.

def signed_quantity(trades: pd.DataFrame) -> pd.Series:
    """+qty for buys, -qty for sells, adjustments as recorded (already signed)."""
    qty = trades["quantity"]
    return qty.where(trades["action"] != "sell", -qty)


def infer_opening_positions(trades: pd.DataFrame, snapshot: HoldingsSnapshot,
                            cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Stock/ETF positions held before the first trade: account, ticker, quantity.

    cutoff: the moment the snapshot reflects (defaults to its "as of" time).
    Options aren't in the holdings report, so they're assumed to start at zero.
    """
    cutoff = cutoff or snapshot.as_of
    stocks = trades[(trades["asset_type"] == "stock") & (trades["timestamp"] <= cutoff)]
    net_change = signed_quantity(stocks).groupby([stocks["account"], stocks["ticker"]]).sum()
    held = snapshot.positions.groupby(["account", "ticker"])["quantity"].sum()

    opening = held.sub(net_change, fill_value=0.0)
    opening = opening[opening.abs() > 1e-6]
    return opening.rename("quantity").reset_index()


def infer_opening_cash(ledger: pd.DataFrame, snapshot: HoldingsSnapshot,
                       cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Cash balance per account and currency before the first ledger entry."""
    cutoff = cutoff or snapshot.as_of
    before = ledger[ledger["timestamp"] <= cutoff]
    net_change = before.groupby(["account", "currency"])["amount"].sum()
    held = snapshot.cash.groupby(["account", "currency"])["amount"].sum()

    opening = held.sub(net_change, fill_value=0.0)
    accounts = set(snapshot.cash["account"])
    # Only accounts in the snapshot can be backed out; others are assumed to start empty.
    opening = opening[opening.index.get_level_values("account").isin(accounts)]
    opening = opening[opening.abs() > 0.005]
    return opening.rename("amount").reset_index()


def find_position_problems(trades: pd.DataFrame, opening: pd.DataFrame | None = None) -> list[str]:
    """Warn where a stock position goes negative at a day's close.

    A negative position means a sell with no matching buy, so the history (or the
    opening positions) is incomplete. Options are skipped: selling an option you
    don't own (writing it) is a legitimate short.
    """
    stocks = trades[trades["asset_type"] == "stock"]
    daily = (signed_quantity(stocks)
             .groupby([stocks["account"], stocks["ticker"], stocks["date"]]).sum())
    held = daily.groupby(level=["account", "ticker"]).cumsum()

    if opening is not None and not opening.empty:
        start = opening.set_index(["account", "ticker"])["quantity"]
        held = held.add(start.reindex(held.droplevel("date").index).fillna(0.0).values)
        problems = [f"{acct} {tkr}: inferred opening position is negative ({qty:g})"
                    for (acct, tkr), qty in start[start < -1e-6].items()]
    else:
        problems = []

    negative = held[held < -1e-6]
    first_negative = negative.reset_index().groupby(["account", "ticker"]).first()
    for (acct, tkr), row in first_negative.iterrows():
        problems.append(f"{acct} {tkr}: position falls to {row.iloc[-1]:g} on {row['date']:%Y-%m-%d}")
    return problems


# ---------------------------------------------------------------------------
# Everything the analytics need, in one object
# ---------------------------------------------------------------------------

@dataclass
class PortfolioData:
    """Standardized inputs for the performance and risk modules."""

    trades: pd.DataFrame             # TRADE_COLUMNS
    cash_ledger: pd.DataFrame        # LEDGER_COLUMNS
    opening_positions: pd.DataFrame  # account, ticker, quantity held at start_date
    opening_cash: pd.DataFrame       # account, currency, amount held at start_date
    start_date: pd.Timestamp
    warnings: list[str] = field(default_factory=list)
    snapshot: HoldingsSnapshot | None = None
    # If set, the account holds cash only in this currency: a USD trade is paid
    # for by converting at that day's rate. None means the ledger already has
    # real per-currency balances (e.g. Wealthsimple).
    cash_currency: str | None = None

    @property
    def accounts(self) -> list[str]:
        return sorted(set(self.trades["account"]) | set(self.cash_ledger["account"]))


def deposits_matching_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Assumed cash flows when no deposits file is given: every buy is funded by new
    money and every sale's proceeds are withdrawn, so cash stays at zero and the
    returns measure the holdings alone."""
    moved_cash = trades[trades["action"] != "adjust"]
    return pd.DataFrame({
        "date": moved_cash["date"],
        "timestamp": moved_cash["timestamp"],
        "account": moved_cash["account"],
        "currency": moved_cash["currency"],
        "amount": -trade_cash(moved_cash),
        "kind": "external",
        "activity": "Assumed deposit/withdrawal",
    })


def load_standard_portfolio(trades_source, cash_flows_source=None,
                            broker: str | None = None) -> PortfolioData:
    """Load a simple trade file plus an optional deposits/withdrawals file (e.g. the sample data)."""
    trades = load_trades(trades_source, broker)
    warnings = find_position_problems(trades)
    if cash_flows_source is not None:
        flows = load_cash_flows(cash_flows_source)
    else:
        flows = deposits_matching_trades(trades)
        warnings.append("No deposits file: each purchase is treated as new money and each sale "
                        "as a withdrawal, so returns reflect your holdings only, not cash.")
    ledger = build_cash_ledger(trades, flows)
    empty_positions = pd.DataFrame(columns=["account", "ticker", "quantity"])
    empty_cash = pd.DataFrame(columns=["account", "currency", "amount"])
    return PortfolioData(
        trades=trades,
        cash_ledger=ledger,
        opening_positions=empty_positions,
        opening_cash=empty_cash,
        start_date=ledger["date"].min(),
        warnings=warnings,
        cash_currency="CAD",
    )


def select_accounts(data: PortfolioData, accounts: list[str]) -> PortfolioData:
    """The same portfolio data restricted to some accounts.

    Transfers between two selected accounts cancel out; a transfer from a
    selected account to an unselected one counts as a withdrawal.
    """
    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["account"].isin(accounts)].reset_index(drop=True)

    return PortfolioData(
        trades=keep(data.trades),
        cash_ledger=keep(data.cash_ledger),
        opening_positions=keep(data.opening_positions),
        opening_cash=keep(data.opening_cash),
        start_date=data.start_date,
        warnings=[w for w in data.warnings if any(w.startswith(a) for a in accounts)],
        snapshot=data.snapshot,
        cash_currency=data.cash_currency,
    )


def apply_stop_overrides(trades: pd.DataFrame, source) -> pd.DataFrame:
    """Fill in stop_price from a file you keep yourself: date, ticker, stop_price[, account].

    For exports with no stop-loss column (like Wealthsimple's). Each row sets the
    stop for buys of that ticker on that date; the position-sizing check then uses
    your stop instead of the ATR estimate.
    """
    stops = read_csv(source)
    stops["date"] = to_date(stops["date"])
    stops["ticker"] = stops["ticker"].astype(str).str.strip().str.upper()
    keys = ["date", "ticker"] + (["account"] if "account" in stops else [])

    merged = trades.merge(stops[keys + ["stop_price"]].rename(columns={"stop_price": "your_stop"}),
                          on=keys, how="left")
    is_buy = merged["action"] == "buy"
    merged["stop_price"] = merged["stop_price"].where(~is_buy | merged["your_stop"].isna(),
                                                      merged["your_stop"])
    return merged[TRADE_COLUMNS]
