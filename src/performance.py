"""Portfolio value, time-weighted returns, and the blended benchmark.

How it fits together:
    1. Share counts: opening positions + every buy, sell and adjustment, at each day's close.
    2. Stock value: shares x closing price x FX, in CAD.
    3. Option value: carried at cost until sold, expired or exercised.
    4. Cash: opening cash + every cash ledger row, per currency, converted to CAD.
    5. Portfolio value = stocks + options + cash.
    6. Time-weighted return: each day's growth with deposits and withdrawals removed,
       chained together.
    7. Benchmark: weighted mix of index returns, reset to the target weights each month.

compute_performance() does the maths on data it is given (easy to test);
analyze_performance() fetches the prices it needs first.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ingest import PortfolioData, signed_quantity
from src.prices import fetch_closes, ticker_currency

DAYS_PER_YEAR = 365.25
# Price history fetched before the start date: covers weekends and holidays for
# the opening valuation, and the warm-up for the ATR used in the risk module.
HISTORY_LOOKBACK_DAYS = 60


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

def rate_on(dates, usdcad: pd.Series) -> np.ndarray:
    """USD->CAD rate on each date; weekends and holidays use the previous quote."""
    return usdcad.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy()


def to_cad(amounts: pd.Series, currencies: pd.Series, dates: pd.Series,
           usdcad: pd.Series) -> np.ndarray:
    """Convert amounts to CAD at each row's own date. CAD amounts are unchanged."""
    rate = np.where(currencies.to_numpy() == "USD", rate_on(dates, usdcad), 1.0)
    return amounts.to_numpy(dtype=float) * rate


# ---------------------------------------------------------------------------
# Building daily balances from a list of changes
# ---------------------------------------------------------------------------

def cumulative_by_day(changes: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Running total of changes at each calendar day's close.

    changes: values indexed by (date, key...), e.g. (date, account, ticker).
    Returns one row per calendar day and one column per key. A change dated on a
    weekend shows up on the next business day.
    """
    keys = list(changes.index.names[1:])
    if changes.empty:
        return pd.DataFrame(index=calendar, columns=pd.MultiIndex.from_arrays([[]] * len(keys), names=keys))
    daily = changes.groupby(level=[0, *keys]).sum().unstack(keys).sort_index()
    return daily.fillna(0.0).cumsum().reindex(calendar, method="ffill").fillna(0.0)


def add_opening(held: pd.DataFrame, opening: pd.Series) -> pd.DataFrame:
    """Add balances that existed before the history starts to every day."""
    columns = held.columns.union(opening.index)
    return held.reindex(columns=columns, fill_value=0.0) + opening.reindex(columns, fill_value=0.0)


def _keyed(values, frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Values indexed by (date, *keys) taken from the matching rows of frame."""
    index = pd.MultiIndex.from_arrays([frame["date"], *[frame[k] for k in keys]],
                                      names=["date", *keys])
    return pd.Series(np.asarray(values, dtype=float), index=index)


# ---------------------------------------------------------------------------
# Stocks and ETFs
# ---------------------------------------------------------------------------

def daily_shares(trades: pd.DataFrame, opening_positions: pd.DataFrame,
                 calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares held per (account, ticker) at each day's close."""
    stocks = trades[trades["asset_type"] == "stock"]
    held = cumulative_by_day(_keyed(signed_quantity(stocks), stocks, ["account", "ticker"]), calendar)
    opening = opening_positions.set_index(["account", "ticker"])["quantity"].astype(float)
    return add_opening(held, opening)


def stock_prices_cad(closes: pd.DataFrame, tickers: list[str], trades: pd.DataFrame,
                     usdcad: pd.Series, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CAD price per share for each ticker and day.

    Returns (prices, has_market_price). Where yfinance has no close, the price
    falls back to the last price you traded at (already in CAD), held flat.
    Before the first known price of any kind (e.g. a SPAC before its merger
    ticker existed), the position is held flat at that first price rather than
    counted as worthless, which would show up as a fake gain later.
    """
    market = (closes.reindex(columns=tickers).sort_index().ffill()
              .reindex(calendar, method="ffill"))
    rate = usdcad.reindex(calendar, method="ffill").to_numpy()[:, None]
    is_usd = np.array([ticker_currency(t) == "USD" for t in tickers])
    market = market * np.where(is_usd, rate, 1.0)

    traded = trades[(trades["asset_type"] == "stock") & trades["action"].isin(["buy", "sell"])]
    price_cad = to_cad(traded["price"], traded["currency"], traded["date"], usdcad)
    last_trade = _keyed(price_cad, traded, ["ticker"]).groupby(level=[0, 1]).last().unstack("ticker")
    fallback = (last_trade.reindex(columns=tickers).sort_index().ffill()
                .reindex(calendar, method="ffill"))

    return market.fillna(fallback).bfill(), market.notna()


def stock_values(shares: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Shares x CAD price for each (account, ticker). Zero where nothing is held."""
    per_share = prices.reindex(columns=shares.columns.get_level_values("ticker")).to_numpy()
    held = shares.to_numpy()
    values = np.where(held == 0, 0.0, held * per_share)
    return pd.DataFrame(values, index=shares.index, columns=shares.columns)


# ---------------------------------------------------------------------------
# Options, carried at cost
# ---------------------------------------------------------------------------

def option_cost_changes(trades: pd.DataFrame, usdcad: pd.Series) -> pd.Series:
    """Change in each option position's at-cost value, per trade, in CAD.

    Average-cost method: opening or adding to a position adds what you paid;
    closing, expiring or exercising removes the average cost of the contracts
    closed. A written (short) option carries a negative value: the premium
    received, which you'd have to pay to buy it back.
    """
    options = trades[trades["asset_type"] == "option"]
    prices = np.nan_to_num(to_cad(options["price"], options["currency"], options["date"], usdcad))
    deltas = signed_quantity(options).to_numpy()

    position = {}  # (account, ticker) -> (contracts, cost in CAD)
    changes = []
    for account, ticker, delta, price in zip(options["account"], options["ticker"], deltas, prices,
                                             strict=True):
        contracts, cost = position.get((account, ticker), (0.0, 0.0))
        if contracts == 0 or np.sign(delta) == np.sign(contracts):
            new_cost = cost + delta * price                     # opening or adding
        else:
            closed = min(abs(delta), abs(contracts))            # closing some or all
            new_cost = cost * (1 - closed / abs(contracts))
            flipped = abs(delta) - closed                       # went from long to short (or back)
            if flipped > 0:
                new_cost = np.sign(delta) * flipped * price
        position[(account, ticker)] = (contracts + delta, new_cost)
        changes.append(new_cost - cost)

    return _keyed(changes, options, ["account", "ticker"])


# ---------------------------------------------------------------------------
# Cash and external flows
# ---------------------------------------------------------------------------

def daily_cash(ledger: pd.DataFrame, opening_cash: pd.DataFrame,
               calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Cash per (account, currency) at each day's close, in its own currency."""
    held = cumulative_by_day(_keyed(ledger["amount"], ledger, ["account", "currency"]), calendar)
    opening = opening_cash.set_index(["account", "currency"])["amount"].astype(float)
    return add_opening(held, opening)


def convert_ledger_to_cad(ledger: pd.DataFrame, usdcad: pd.Series) -> pd.DataFrame:
    """For CAD-only accounts: book every cash movement in CAD at its own date's rate."""
    converted = ledger.copy()
    converted["amount"] = to_cad(ledger["amount"], ledger["currency"], ledger["date"], usdcad)
    converted["currency"] = "CAD"
    return converted


def cash_by_account_cad(cash: pd.DataFrame, usdcad: pd.Series) -> pd.DataFrame:
    """Cash per account in CAD: CAD balances plus USD balances x USD->CAD rate."""
    rate = usdcad.reindex(cash.index, method="ffill").to_numpy()[:, None]
    is_usd = cash.columns.get_level_values("currency") == "USD"
    in_cad = cash * np.where(is_usd, rate, 1.0)
    return in_cad.T.groupby(level="account").sum().T


def cash_in_cad(cash: pd.DataFrame, usdcad: pd.Series) -> pd.Series:
    """Total cash in CAD across all accounts."""
    return cash_by_account_cad(cash, usdcad).sum(axis=1)


def external_flows_cad(ledger: pd.DataFrame, calendar: pd.DatetimeIndex,
                       usdcad: pd.Series) -> pd.Series:
    """Deposits (+) and withdrawals (-) in CAD per calendar day.

    A flow dated on a weekend lands on the next business day, matching how
    balances are rolled forward in cumulative_by_day.
    """
    external = ledger[ledger["kind"] == "external"]
    amounts = to_cad(external["amount"], external["currency"], external["date"], usdcad)
    position = calendar.searchsorted(external["date"])
    in_range = position < len(calendar)
    flows = pd.Series(amounts[in_range], index=calendar[position[in_range]])
    return flows.groupby(level=0).sum().reindex(calendar, fill_value=0.0)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def daily_twr(values: pd.Series, flows: pd.Series) -> pd.Series:
    """Each day's time-weighted return.

    Flows are treated as arriving at the start of the day, so the day's return
    is measured on what was invested that day:

        r = value_today / (value_yesterday + flow_today) - 1

    A deposit therefore can't look like a gain. Days with nothing invested
    return NaN (undefined), not 0.
    """
    invested = values.shift(1).fillna(0.0) + flows
    return (values / invested - 1).where(invested > 0)


def compound(returns: pd.Series) -> float:
    """Chain returns together: (1 + r1) x (1 + r2) x ... - 1."""
    return float((1 + returns.dropna()).prod() - 1)


def monthly_returns(daily: pd.Series) -> pd.Series:
    """Compound daily returns within each calendar month."""
    daily = daily.dropna()
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1


def annualize(total_return: float, days: int) -> float | None:
    """Annual rate that compounds to total_return over `days` calendar days.

    Returns None for periods under a year: scaling up a few months' return
    would mostly be extrapolating luck (the GIPS standard says the same).
    """
    if days < 365:
        return None
    return (1 + total_return) ** (DAYS_PER_YEAR / days) - 1


# ---------------------------------------------------------------------------
# Blended benchmark
# ---------------------------------------------------------------------------

def benchmark_levels_cad(index_closes: pd.DataFrame, components: list[dict],
                         calendar: pd.DatetimeIndex, usdcad: pd.Series) -> pd.DataFrame:
    """Index levels in CAD, one column per component, on every calendar day."""
    levels = index_closes.sort_index().ffill().reindex(calendar, method="ffill")
    rate = usdcad.reindex(calendar, method="ffill")
    for component in components:
        if component["currency"] == "USD":
            levels[component["ticker"]] = levels[component["ticker"]] * rate
    return levels[[c["ticker"] for c in components]]


def rebalanced_index(levels: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Growth of 1 invested in the blend, reset to the target weights at each month-end.

    Within a month each component drifts with its own return. At month-end the
    mix is rebalanced, so each month's benchmark return is exactly
    sum(weight x component's monthly return).
    """
    month = levels.index.to_period("M")
    month_start = levels.groupby(month).last().shift(1)   # previous month-end level
    month_start.iloc[0] = levels.iloc[0]
    growth_this_month = levels / month_start.loc[month].to_numpy()

    w = pd.Series(weights)[levels.columns]
    blend_this_month = growth_this_month.mul(w, axis=1).sum(axis=1)
    carried_in = blend_this_month.groupby(month).last().cumprod().shift(1).fillna(1.0)
    return blend_this_month * carried_in.loc[month].to_numpy()


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

@dataclass
class PerformanceResult:
    values: pd.DataFrame           # stocks, options, cash, total (CAD) per day
    account_values: pd.DataFrame   # total value (CAD) per account per day
    position_values: pd.DataFrame  # CAD value per (account, ticker) per day
    shares: pd.DataFrame           # shares per (account, ticker) per day
    cash: pd.DataFrame             # cash per (account, currency) per day, own currency
    flows: pd.Series               # external flows in CAD per day
    daily_returns: pd.Series       # portfolio TWR per day
    benchmark_returns: pd.Series   # blended benchmark per day, same days as portfolio
    growth: pd.DataFrame           # portfolio and benchmark growth of 1
    monthly: pd.DataFrame          # portfolio, benchmark, excess, partial
    summary: dict
    usdcad: pd.Series              # CAD per USD, every quoted day
    warnings: list[str] = field(default_factory=list)


def _month_is_partial(month: pd.Period, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if the period doesn't cover the whole month (only possible for the first or last).

    A month is fully covered if we have a value at the previous month's last
    business day and at this month's last business day.
    """
    last_business_day = pd.offsets.BMonthEnd().rollback
    previous_month_end = last_business_day(month.start_time - pd.Timedelta(days=1))
    month_end = last_business_day(month.end_time.normalize())
    return start > previous_month_end or end < month_end


def monthly_table(portfolio: pd.Series, benchmark: pd.Series,
                  start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Monthly portfolio vs benchmark returns, with partial months flagged."""
    table = pd.DataFrame({"portfolio": monthly_returns(portfolio),
                          "benchmark": monthly_returns(benchmark)})
    table["excess"] = table["portfolio"] - table["benchmark"]
    table["partial"] = [_month_is_partial(m, start, end) for m in table.index]
    return table


def _month_record(table: pd.DataFrame, month: pd.Period) -> dict:
    row = table.loc[month]
    return {"month": str(month), "return": float(row["portfolio"]),
            "benchmark": float(row["benchmark"]), "partial": bool(row["partial"])}


def summarize(values: pd.Series, flows: pd.Series, returns: pd.Series,
              benchmark: pd.Series, monthly: pd.DataFrame, period_start: pd.Timestamp) -> dict:
    """Headline numbers. Every figure here is computed, never estimated."""
    end = values.index[-1]
    days = (end - period_start).days
    total = compound(returns)
    bench_total = compound(benchmark)
    net_flows = float(flows.loc[flows.index > period_start].sum())
    start_value, end_value = float(values.loc[period_start]), float(values.iloc[-1])

    return {
        "start_date": period_start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "days": days,
        "starting_value_cad": start_value,
        "ending_value_cad": end_value,
        "net_deposits_cad": net_flows,
        "investment_gain_cad": end_value - start_value - net_flows,
        "total_return": total,
        "benchmark_total_return": bench_total,
        "excess_return": total - bench_total,
        "annualized_return": annualize(total, days),
        "benchmark_annualized_return": annualize(bench_total, days),
        "best_month": _month_record(monthly, monthly["portfolio"].idxmax()),
        "worst_month": _month_record(monthly, monthly["portfolio"].idxmin()),
    }


def compute_performance(data: PortfolioData, closes: pd.DataFrame, fx_cadusd: pd.Series,
                        index_closes: pd.DataFrame, components: list[dict]) -> PerformanceResult:
    """Rebuild daily portfolio value and returns from trades, cash and prices."""
    usdcad = (1.0 / fx_cadusd.dropna()).sort_index()
    calendar = pd.bdate_range(data.start_date - pd.offsets.BDay(1), usdcad.index[-1])
    warnings = list(data.warnings)
    ledger = data.cash_ledger
    if data.cash_currency == "CAD":
        ledger = convert_ledger_to_cad(ledger, usdcad)

    # Stocks and ETFs
    shares = daily_shares(data.trades, data.opening_positions, calendar)
    tickers = sorted(set(shares.columns.get_level_values("ticker")))
    prices, has_market = stock_prices_cad(closes, tickers, data.trades, usdcad, calendar)
    stocks = stock_values(shares, prices)

    held = shares.ne(0)
    held_tickers = held.T.groupby(level="ticker").any().T
    no_market = (held_tickers & ~has_market.reindex(columns=held_tickers.columns)).any()
    if no_market.any():
        warnings.append("No market price for some days while held; valued at the nearest price "
                        "you traded at: " + ", ".join(sorted(no_market[no_market].index)))
    unpriced = stocks.isna().any()
    if unpriced.any():
        names = sorted({t for _, t in unpriced[unpriced].index})
        warnings.append(f"No price at all, counted as 0: {', '.join(names)}")
        stocks = stocks.fillna(0.0)

    # Options at cost, and cash
    options = cumulative_by_day(option_cost_changes(data.trades, usdcad), calendar)
    cash = daily_cash(ledger, data.opening_cash, calendar)

    values = pd.DataFrame({
        "stocks": stocks.sum(axis=1),
        "options": options.sum(axis=1),
        "cash": cash_in_cad(cash, usdcad),
    })
    values["total"] = values.sum(axis=1)

    # Returns
    flows = external_flows_cad(ledger, calendar, usdcad)
    returns = daily_twr(values["total"], flows)
    if returns.dropna().empty:
        raise ValueError("No day had money invested, so no return can be calculated.")
    first_return = returns.first_valid_index()
    period_start = calendar[max(calendar.get_loc(first_return) - 1, 0)]
    gaps = returns.loc[first_return:].isna().sum()
    if gaps:
        warnings.append(f"{gaps} day(s) with zero or negative invested value were skipped.")

    levels = benchmark_levels_cad(index_closes, components, calendar, usdcad)
    bench_index = rebalanced_index(levels, {c["ticker"]: c["weight"] for c in components})
    bench_returns = bench_index.pct_change().where(returns.notna())

    growth = pd.DataFrame({
        "portfolio": (1 + returns.fillna(0.0)).cumprod(),
        "benchmark": (1 + bench_returns.fillna(0.0)).cumprod(),
    }).loc[period_start:]

    monthly = monthly_table(returns, bench_returns, period_start, calendar[-1])
    position_values = pd.concat([stocks, options], axis=1)
    account_values = (position_values.T.groupby(level="account").sum().T
                      .add(cash_by_account_cad(cash, usdcad), fill_value=0.0))

    return PerformanceResult(
        values=values.loc[period_start:],
        account_values=account_values.loc[period_start:],
        position_values=position_values.loc[period_start:],
        shares=shares.loc[period_start:],
        cash=cash.loc[period_start:],
        flows=flows.loc[period_start:],
        daily_returns=returns.loc[first_return:],
        benchmark_returns=bench_returns.loc[first_return:],
        growth=growth,
        monthly=monthly,
        summary=summarize(values["total"], flows, returns, bench_returns, monthly, period_start),
        usdcad=usdcad,
        warnings=warnings,
    )


def analyze_performance(data: PortfolioData, config: dict, end=None) -> PerformanceResult:
    """Fetch the prices, FX and index levels needed, then compute performance."""
    fetch_from = data.start_date - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)
    stock_tickers = sorted(set(data.trades.loc[data.trades["asset_type"] == "stock", "ticker"])
                           | set(data.opening_positions["ticker"]))
    components = config["benchmark"]["components"]

    closes = fetch_closes(stock_tickers, fetch_from, end)
    fx = fetch_closes([config["fx_ticker"]], fetch_from, end)[config["fx_ticker"]]
    index_closes = fetch_closes([c["ticker"] for c in components], fetch_from, end,
                                total_return=config["benchmark"].get("total_return", False))
    return compute_performance(data, closes, fx, index_closes, components)
