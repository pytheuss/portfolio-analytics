"""Risk: volatility, drawdown, Sharpe, beta, trade statistics, position sizing, exposure.

Everything here is calculated from the Phase 2 results (daily returns, values
and positions) plus market data. Nothing is estimated by the AI.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ingest import PortfolioData, signed_quantity
from src.performance import HISTORY_LOOKBACK_DAYS, PerformanceResult, to_cad
from src.prices import fetch_prices, fetch_security_info, ticker_currency

# Our daily series has one row per weekday, holidays included. A holiday's
# return is ~0 and adds no variance, so annualizing by weekdays per year
# (rather than the usual 252 trading days) gives the exact annual variance.
WEEKDAYS_PER_YEAR = 365.25 * 5 / 7   # ~260.9


# ---------------------------------------------------------------------------
# Portfolio-level risk
# ---------------------------------------------------------------------------

def annualized_volatility(daily: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to a year: sd x sqrt(days per year)."""
    return float(daily.dropna().std() * np.sqrt(WEEKDAYS_PER_YEAR))


def drawdown_series(growth: pd.Series) -> pd.Series:
    """How far below its previous high the growth-of-1 line is on each day (0 = at a high)."""
    return growth / growth.cummax() - 1


def max_drawdown(growth: pd.Series) -> dict:
    """Largest peak-to-trough fall, with the dates it started, bottomed and recovered.

    Measured on time-weighted growth, not account value, so a withdrawal isn't
    mistaken for a loss and a deposit isn't mistaken for a recovery.
    """
    drawdown = drawdown_series(growth)
    trough = drawdown.idxmin()
    peak = growth.loc[:trough].idxmax()
    after = growth.loc[trough:]
    recovered = after[after >= growth[peak]]
    return {
        "max_drawdown": float(drawdown.min()),
        "peak_date": peak.date().isoformat(),
        "trough_date": trough.date().isoformat(),
        "recovery_date": recovered.index[0].date().isoformat() if len(recovered) else None,
    }


def sharpe_ratio(daily: pd.Series, risk_free_rate: float) -> float:
    """Annualized Sharpe ratio: average return above the risk-free rate per unit of volatility.

        daily excess return = r - rf_daily,  where rf_daily = (1 + rf)^(1/days per year) - 1
        Sharpe = mean(excess) / sd(excess) x sqrt(days per year)
    """
    rf_daily = (1 + risk_free_rate) ** (1 / WEEKDAYS_PER_YEAR) - 1
    excess = daily.dropna() - rf_daily
    if not excess.std() > 0:
        return float("nan")   # no variation (or too little data): Sharpe is undefined
    return float(excess.mean() / excess.std() * np.sqrt(WEEKDAYS_PER_YEAR))


def beta_and_correlation(portfolio: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Beta = cov(portfolio, benchmark) / var(benchmark), from daily returns.

    A beta of 1.5 means the portfolio moved 1.5% on average for each 1% benchmark move.
    """
    both = pd.concat([portfolio, benchmark], axis=1).dropna()
    if not (both.std() > 0).all():
        return float("nan"), float("nan")   # a series that never moves has no beta
    covariance = both.cov().iloc[0, 1]
    return float(covariance / both.iloc[:, 1].var()), float(both.corr().iloc[0, 1])


def portfolio_risk(perf: PerformanceResult, risk_free_rate: float) -> dict:
    """Headline risk numbers for the portfolio and, for comparison, the benchmark."""
    benchmark_growth = perf.growth["benchmark"]
    beta, correlation = beta_and_correlation(perf.daily_returns, perf.benchmark_returns)
    return {
        "risk_free_rate": risk_free_rate,
        "volatility": annualized_volatility(perf.daily_returns),
        "benchmark_volatility": annualized_volatility(perf.benchmark_returns),
        "sharpe_ratio": sharpe_ratio(perf.daily_returns, risk_free_rate),
        "benchmark_sharpe_ratio": sharpe_ratio(perf.benchmark_returns, risk_free_rate),
        "beta": beta,
        "correlation": correlation,
        "drawdown": max_drawdown(perf.growth["portfolio"]),
        "benchmark_drawdown": max_drawdown(benchmark_growth),
    }


# ---------------------------------------------------------------------------
# Round trips: win rate, average win/loss, profit factor
# ---------------------------------------------------------------------------
# A round trip runs from the trade that opens a position (from zero shares)
# to the trade that brings it back to zero. Its profit is realized with the
# average-cost method (adjusted cost base, as the CRA uses), in CAD.

def _opening_prices(perf: PerformanceResult) -> pd.Series:
    """CAD price per share of each opening position, from the first day's valuation."""
    first_shares = perf.shares.iloc[0]
    first_values = perf.position_values.iloc[0].reindex(first_shares.index)
    held = first_shares[first_shares != 0]
    return (first_values[held.index] / held).rename("price")


def round_trips(data: PortfolioData, perf: PerformanceResult) -> pd.DataFrame:
    """One row per closed round trip: account, ticker, asset_type, opened, closed, pnl_cad.

    Positions held on the start date enter at that day's market price, so their
    profit is measured over the analysis period only (from_opening = True).
    Stock adjustments (splits, consolidations) change the share count but not
    the cost; an option expiring closes at a price of zero.
    """
    trades = data.trades
    prices = np.nan_to_num(to_cad(trades["price"], trades["currency"], trades["date"], perf.usdcad))
    fees = np.nan_to_num(to_cad(trades["fees"], trades["currency"], trades["date"], perf.usdcad))
    deltas = signed_quantity(trades).to_numpy()
    start = perf.values.index[0]

    opening_prices = _opening_prices(perf)
    state = {}  # (account, ticker) -> dict(qty, cost, pnl, opened, from_opening)
    for (account, ticker), qty in data.opening_positions.set_index(["account", "ticker"])["quantity"].items():
        price = opening_prices.get((account, ticker), np.nan)
        state[(account, ticker)] = dict(qty=qty, cost=qty * price, pnl=0.0, opened=start,
                                        from_opening=True)

    trips = []
    rows = zip(trades["date"], trades["account"], trades["ticker"], trades["asset_type"],
               trades["action"], deltas, prices, fees, strict=True)
    for day, account, ticker, asset, action, delta, price, fee in rows:
        pos = state.get((account, ticker)) or dict(qty=0.0, cost=0.0, pnl=0.0, opened=day,
                                                   from_opening=False)
        if pos["qty"] == 0:
            pos.update(cost=0.0, pnl=0.0, opened=day, from_opening=False)

        if action == "adjust" and asset == "stock":
            pos["qty"] += delta                              # split: same cost, more shares
        elif pos["qty"] == 0 or np.sign(delta) == np.sign(pos["qty"]):
            pos["cost"] += delta * price + fee               # opening or adding
            pos["qty"] += delta
        else:
            closing = np.sign(delta) * min(abs(delta), abs(pos["qty"]))
            average_cost = pos["cost"] / pos["qty"]
            pos["pnl"] += -closing * (price - average_cost) - fee
            pos["cost"] += closing * average_cost
            pos["qty"] += closing
            leftover = delta - closing                       # flipped from long to short or back
            if abs(leftover) > 1e-9:
                trips.append(_trip(account, ticker, asset, pos, day))
                pos = dict(qty=leftover, cost=leftover * price, pnl=0.0, opened=day,
                           from_opening=False)

        if abs(pos["qty"]) < 1e-9 and (action != "adjust" or asset == "option"):
            trips.append(_trip(account, ticker, asset, pos, day))
            pos = dict(qty=0.0, cost=0.0, pnl=0.0, opened=day, from_opening=False)
        state[(account, ticker)] = pos

    return pd.DataFrame(trips, columns=["account", "ticker", "asset_type", "opened", "closed",
                                        "pnl_cad", "from_opening"])


def _trip(account, ticker, asset, pos, day) -> dict:
    return dict(account=account, ticker=ticker, asset_type=asset, opened=pos["opened"],
                closed=day, pnl_cad=pos["pnl"], from_opening=pos["from_opening"])


def trade_statistics(trips: pd.DataFrame) -> dict:
    """Win rate, average win vs average loss, and profit factor for closed round trips.

    profit factor = total profit on winners / total loss on losers (above 1 = net profitable)
    payoff ratio  = average win / average loss
    """
    pnl = trips["pnl_cad"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = -losses.sum()
    return {
        "round_trips": int(len(pnl)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else None,
        "average_win_cad": float(wins.mean()) if len(wins) else None,
        "average_loss_cad": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "net_realized_pnl_cad": float(pnl.sum()),
        "largest_win_cad": float(pnl.max()) if len(pnl) else None,
        "largest_loss_cad": float(pnl.min()) if len(pnl) else None,
    }


# ---------------------------------------------------------------------------
# Position sizing: risk at entry as a % of account equity
# ---------------------------------------------------------------------------

def average_true_range(prices: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's Average True Range: the typical daily price range, including overnight gaps.

        true range = max(high - low, |high - previous close|, |low - previous close|)
        ATR        = exponential average of true range with weight 1/period
    """
    previous_close = prices["close"].shift(1)
    true_range = pd.concat([
        prices["high"] - prices["low"],
        (prices["high"] - previous_close).abs(),
        (prices["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_before(atr: pd.Series, day: pd.Timestamp) -> float:
    """The ATR as of the last close before `day`: only information known at entry."""
    known = atr.loc[atr.index < day].dropna()
    return float(known.iloc[-1]) if len(known) else np.nan


def _equity_at_entry(perf: PerformanceResult, ledger: pd.DataFrame) -> callable:
    """Function (account, day) -> account value at the previous close plus deposits since.

    Adding same-day deposits means a first trade made right after funding an
    account is measured against the money actually in it.
    """
    values = perf.account_values
    external = ledger[ledger["kind"] == "external"]
    flows_cad = pd.Series(to_cad(external["amount"], external["currency"], external["date"],
                                 perf.usdcad), index=external.index)

    def equity(account: str, day: pd.Timestamp) -> float:
        before = values.loc[values.index < day, account] if account in values else pd.Series()
        last_close = before.index[-1] if len(before) else pd.Timestamp.min
        base = float(before.iloc[-1]) if len(before) else 0.0
        recent = (external["account"] == account) & (external["date"] > last_close) & \
                 (external["date"] <= day)
        return base + float(flows_cad[recent].sum())

    return equity


def position_sizing(data: PortfolioData, perf: PerformanceResult, config: dict,
                    history: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Risk taken on every entry (each buy into a long, each option opened), as % of equity.

    history: daily high/low/close per ticker; fetched from yfinance if not given.

    Risk per share, by method:
      "your stop"     entry price - the stop_price you entered
      "ATR"           atr_multiple x ATR before the trade (no hard stop recorded)
      "premium"       long option: the most you can lose is what you paid
      "uncapped"      written (sold) option: loss isn't capped, so it can't be within 1%
    """
    settings = config["risk"]
    target, period, multiple = (settings["target_risk_per_trade"], settings["atr_period"],
                                settings["atr_multiple"])
    trades = data.trades
    before = signed_quantity(trades).groupby([trades["account"], trades["ticker"]]).cumsum() \
        - signed_quantity(trades)
    opening = data.opening_positions.set_index(["account", "ticker"])["quantity"]
    held_before = before + [opening.get((a, t), 0.0) for a, t in zip(trades["account"], trades["ticker"], strict=True)]

    opens_long = (trades["action"] == "buy") & (held_before >= 0)
    opens_short_option = (trades["asset_type"] == "option") & (trades["action"] == "sell") & (held_before <= 0)
    entries = trades[opens_long | opens_short_option]

    stock_tickers = sorted(set(entries.loc[entries["asset_type"] == "stock", "ticker"]))
    if history is None:
        history = fetch_prices(stock_tickers, data.start_date - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS))
    atrs = {t: average_true_range(history[t], period) for t in stock_tickers
            if t in history and not history[t].empty}
    equity_at = _equity_at_entry(perf, data.cash_ledger)

    rows = []
    for _, trade in entries.iterrows():
        trade_rate = 1.0 if trade["currency"] == "CAD" else float(perf.usdcad.asof(trade["date"]))
        listing_rate = 1.0 if ticker_currency(trade["ticker"]) == "CAD" \
            else float(perf.usdcad.asof(trade["date"]))
        risk_cad, stop, method = np.nan, np.nan, None

        if trade["asset_type"] == "option":
            if trade["action"] == "buy":
                method, risk_cad = "premium", trade["quantity"] * trade["price"] * trade_rate + trade["fees"] * trade_rate
            else:
                method = "uncapped"
        elif pd.notna(trade["stop_price"]):
            method, stop = "your stop", trade["stop_price"]
            risk_cad = trade["quantity"] * (trade["price"] - stop) * trade_rate
        else:
            atr = atr_before(atrs[trade["ticker"]], trade["date"]) if trade["ticker"] in atrs else np.nan
            method = "ATR" if pd.notna(atr) else "no data"
            stop = trade["price"] - multiple * atr * listing_rate / trade_rate
            risk_cad = trade["quantity"] * multiple * atr * listing_rate

        equity = equity_at(trade["account"], trade["date"])
        risk_pct = risk_cad / equity if equity > 0 and pd.notna(risk_cad) else np.nan
        rows.append(dict(
            date=trade["date"], account=trade["account"], ticker=trade["ticker"],
            asset_type=trade["asset_type"], quantity=trade["quantity"], price=trade["price"],
            currency=trade["currency"], method=method, stop_price=stop, risk_cad=risk_cad,
            equity_cad=equity, risk_pct=risk_pct,
            over_target=True if method == "uncapped" else bool(risk_pct > target) if pd.notna(risk_pct) else None,
        ))
    return pd.DataFrame(rows)


def sizing_summary(sizing: pd.DataFrame, target: float) -> dict:
    """How many entries stayed within the risk-per-trade target."""
    assessed = sizing[sizing["over_target"].notna()]
    over = assessed[assessed["over_target"] == True]  # noqa: E712 (column holds True/False/None)
    measured = assessed["risk_pct"].dropna()
    return {
        "target_risk_per_trade": target,
        "entries": int(len(sizing)),
        "assessed": int(len(assessed)),
        "within_target": int(len(assessed) - len(over)),
        "over_target": int(len(over)),
        "share_within_target": float((len(assessed) - len(over)) / len(assessed)) if len(assessed) else None,
        "uncapped_written_options": int((sizing["method"] == "uncapped").sum()),
        "not_assessed_no_data": int((sizing["method"] == "no data").sum()),
        "median_risk_pct": float(measured.median()) if len(measured) else None,
        "largest_risk_pct": float(measured.max()) if len(measured) else None,
    }


# ---------------------------------------------------------------------------
# Exposure: sector, country, currency, leverage
# ---------------------------------------------------------------------------

def _classify(ticker: str, asset_type: str, info: pd.DataFrame) -> tuple[str, str]:
    """(sector, country) for one holding, with fallbacks when yfinance has no data."""
    listing_country = "Canada" if ticker_currency(ticker) == "CAD" else "United States"
    if asset_type == "option":
        return "Options", listing_country
    row = info.loc[ticker] if ticker in info.index else pd.Series(dtype=object)
    if row.get("quote_type") == "ETF":
        return "ETF / fund", f"{listing_country} (fund listing)"
    sector = row.get("sector") if isinstance(row.get("sector"), str) else "Unknown"
    country = row.get("country") if isinstance(row.get("country"), str) else f"{listing_country} (listing)"
    return sector, country


def exposure(perf: PerformanceResult, info: pd.DataFrame, day: pd.Timestamp | None = None) -> dict:
    """Holdings on `day` (default: latest) grouped by sector, country and currency.

    Weights are shares of gross holdings value (stocks + options, ignoring cash).
    Leverage = gross holdings / account equity: above 1 means borrowing on margin.
    """
    day = day or perf.position_values.index[-1]
    values = perf.position_values.loc[day]
    values = values[values.abs() > 0.005]
    option_tickers = set(values.index.get_level_values("ticker")) - set(
        perf.shares.columns.get_level_values("ticker"))

    positions = values.rename("value_cad").reset_index()
    positions["asset_type"] = ["option" if t in option_tickers else "stock" for t in positions["ticker"]]
    classified = [_classify(t, a, info) for t, a in zip(positions["ticker"], positions["asset_type"], strict=True)]
    positions["sector"] = [c[0] for c in classified]
    positions["country"] = [c[1] for c in classified]
    positions["currency"] = [ticker_currency(t) for t in positions["ticker"]]
    gross = positions["value_cad"].abs().sum()
    positions["weight"] = positions["value_cad"] / gross if gross else np.nan

    cash = perf.cash.loc[day]
    rate = float(perf.usdcad.asof(day))
    cash_cad = cash * np.where(cash.index.get_level_values("currency") == "USD", rate, 1.0)
    equity = float(perf.values.loc[day, "total"])

    def by(column: str) -> pd.DataFrame:
        table = positions.groupby(column)["value_cad"].sum().sort_values(ascending=False)
        return pd.DataFrame({"value_cad": table, "weight": table / gross if gross else np.nan})

    currency = by("currency")
    currency["cash_cad"] = cash_cad.groupby(level="currency").sum().reindex(currency.index).fillna(0.0)
    currency["net_cad"] = currency["value_cad"] + currency["cash_cad"]

    return {
        "date": day.date().isoformat(),
        "positions": positions.sort_values("value_cad", ascending=False).reset_index(drop=True),
        "sector": by("sector"),
        "country": by("country"),
        "currency": currency,
        "gross_holdings_cad": float(gross),
        "cash_cad": float(cash_cad.sum()),
        "equity_cad": equity,
        "leverage": float(gross / equity) if equity > 0 else None,
    }


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    summary: dict                 # volatility, Sharpe, beta, drawdowns
    drawdown: pd.Series           # daily drawdown of the portfolio
    round_trips: pd.DataFrame
    trade_stats: dict
    trade_stats_by_type: dict     # "stock" / "option" -> trade_statistics
    sizing: pd.DataFrame
    sizing_summary: dict
    exposure: dict
    warnings: list[str] = field(default_factory=list)


def analyze_risk(data: PortfolioData, perf: PerformanceResult, config: dict) -> RiskResult:
    """All Phase 3 risk measures for the portfolio (or the selected accounts)."""
    settings = config["risk"]
    trips = round_trips(data, perf)
    by_type = {asset: trade_statistics(group) for asset, group in trips.groupby("asset_type")}
    sizing = position_sizing(data, perf, config)

    held = perf.position_values.iloc[-1]
    tickers = sorted(set(held[held.abs() > 0.005].index.get_level_values("ticker"))
                     & set(perf.shares.columns.get_level_values("ticker")))
    info = fetch_security_info(tickers)

    summary = portfolio_risk(perf, settings["risk_free_rate"])
    warnings = []
    if summary["sharpe_ratio"] > 0 and perf.summary["total_return"] < 0:
        warnings.append(
            "Sharpe ratio is positive but the compounded return is negative. Sharpe uses the "
            "average daily return; at this volatility, swings cost more than the average gains "
            "(volatility drag, roughly volatility^2 / 2 a year), so read Sharpe with care.")
    no_data = sizing.loc[sizing["method"] == "no data", "ticker"].unique()
    if len(no_data):
        warnings.append("No price history to estimate ATR, so these entries weren't assessed: "
                        + ", ".join(sorted(no_data)))

    return RiskResult(
        summary=summary,
        drawdown=drawdown_series(perf.growth["portfolio"]),
        round_trips=trips,
        trade_stats=trade_statistics(trips),
        trade_stats_by_type=by_type,
        sizing=sizing,
        sizing_summary=sizing_summary(sizing, settings["target_risk_per_trade"]),
        exposure=exposure(perf, info),
        warnings=warnings,
    )
