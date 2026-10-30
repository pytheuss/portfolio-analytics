"""AI commentary: build a JSON summary of calculated results, then have Claude write about it.

The core rule of this project lives here: the AI never computes figures.
    1. build_monthly_summary() / build_portfolio_context() collect numbers that
       Python has already calculated, rounded the way they should be quoted
       (percentages as percent values, CAD as whole dollars).
    2. The model receives only that JSON and is told to use only its figures.
    3. check_numbers() scans the model's output and flags any number that does
       not appear in the JSON, so a made-up figure can't slip through unnoticed.
"""

import json
import os
import re
from collections.abc import Iterator

import anthropic
import numpy as np
import pandas as pd

from src.ingest import PortfolioData, trade_cash
from src.performance import PerformanceResult, compound, to_cad
from src.risk import (
    RiskResult,
    annualized_volatility,
    beta_and_correlation,
    exposure,
    max_drawdown,
    sharpe_ratio,
)

# Server-side refusal fallback: if the model declines, the API retries on a fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DISCLAIMER = "This letter describes a personal portfolio and is not investment advice."


# ---------------------------------------------------------------------------
# Rounding: numbers are stored exactly as they should be quoted
# ---------------------------------------------------------------------------

def pct(x) -> float | None:
    """0.24917 -> 24.92 (a percent value, 2 decimals)."""
    return None if x is None or pd.isna(x) else round(float(x) * 100, 2)


def cad(x) -> int | None:
    """Whole Canadian dollars."""
    return None if x is None or pd.isna(x) else int(round(float(x)))


def num(x, digits: int = 2) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), digits)


OPTION_TICKER = re.compile(r"^([A-Z.\-]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def readable_ticker(ticker: str) -> str:
    """'INFQ261009C00013000' -> 'INFQ 2026-10-09 13 call'; stock tickers are unchanged."""
    match = OPTION_TICKER.match(ticker)
    if not match:
        return ticker
    underlying, yy, mm, dd, kind, strike = match.groups()
    return f"{underlying} 20{yy}-{mm}-{dd} {int(strike) / 1000:g} {'call' if kind == 'C' else 'put'}"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def month_window(perf: PerformanceResult, month: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end): the last valuation before the month and the last one in it."""
    index = perf.values.index
    in_month = index[(index >= month.start_time) & (index <= month.end_time)]
    before = index[index < month.start_time]
    start = before[-1] if len(before) else index[0]
    return start, in_month[-1]


def position_pnl(data: PortfolioData, perf: PerformanceResult,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Profit or loss per ticker between two dates, in CAD (all accounts combined).

        P&L = value at end - value at start + cash from sales - cash spent on purchases

    Dividends and interest are not attributed to positions (they sit in cash).
    """
    values = perf.position_values
    change = values.loc[end] - values.loc[start]
    trades = data.trades[(data.trades["date"] > start) & (data.trades["date"] <= end)]
    cash = to_cad(trade_cash(trades), trades["currency"], trades["date"], perf.usdcad)
    cash_by_position = pd.Series(np.nan_to_num(cash), index=pd.MultiIndex.from_arrays(
        [trades["account"], trades["ticker"]])).groupby(level=[0, 1]).sum()
    pnl = change.add(cash_by_position, fill_value=0.0)
    return pnl.groupby(level=-1).sum().sort_values()


def _ranked(pnl: pd.Series, start_value: float, n: int, best: bool) -> list[dict]:
    chosen = pnl[pnl > 0].tail(n)[::-1] if best else pnl[pnl < 0].head(n)
    return [{"position": readable_ticker(t), "pnl_cad": cad(v),
             "pnl_pct_of_start_value": pct(v / start_value) if start_value > 0 else None}
            for t, v in chosen.items()]


def _exposure_snapshot(perf: PerformanceResult, info: pd.DataFrame, day: pd.Timestamp) -> dict:
    """Weights by sector and currency, leverage and top holdings on one day."""
    e = exposure(perf, info, day)
    if e["gross_holdings_cad"] == 0:
        return {"date": e["date"], "holdings_cad": 0, "note": "no positions held"}
    top = e["positions"].groupby("ticker")["value_cad"].sum().sort_values(ascending=False).head(5)
    return {
        "date": e["date"],
        "equity_cad": cad(e["equity_cad"]),
        "holdings_cad": cad(e["gross_holdings_cad"]),
        "cash_cad": cad(e["cash_cad"]),
        "leverage_x": num(e["leverage"]),
        "sector_weights_pct": {k: pct(v) for k, v in e["sector"]["weight"].items()},
        "currency_weights_pct": {k: pct(v) for k, v in e["currency"]["weight"].items()},
        "top_holdings": [{"position": readable_ticker(t), "weight_pct": pct(v / e["gross_holdings_cad"]),
                          "value_cad": cad(v)} for t, v in top.items()],
    }


def _weight_changes(before: dict, after: dict, key: str) -> dict:
    """Change in percentage points for each group present on either date."""
    old, new = before.get(key, {}), after.get(key, {})
    return {k: num((new.get(k) or 0) - (old.get(k) or 0)) for k in sorted(set(old) | set(new))
            if abs((new.get(k) or 0) - (old.get(k) or 0)) >= 0.01}


def _risk_to(perf: PerformanceResult, end: pd.Timestamp, risk_free_rate: float) -> dict:
    """Risk statistics from the start of the history up to `end`."""
    returns = perf.daily_returns.loc[:end]
    bench = perf.benchmark_returns.loc[:end]
    beta, correlation = beta_and_correlation(returns, bench)
    drawdown = max_drawdown(perf.growth["portfolio"].loc[:end])
    return {
        "volatility_annualized_pct": pct(annualized_volatility(returns)),
        "benchmark_volatility_annualized_pct": pct(annualized_volatility(bench)),
        "sharpe_ratio": num(sharpe_ratio(returns, risk_free_rate)),
        "benchmark_sharpe_ratio": num(sharpe_ratio(bench, risk_free_rate)),
        "beta": num(beta),
        "correlation": num(correlation),
        "max_drawdown_pct": pct(drawdown["max_drawdown"]),
        "max_drawdown_peak_date": drawdown["peak_date"],
        "max_drawdown_trough_date": drawdown["trough_date"],
        "max_drawdown_recovered_on": drawdown["recovery_date"],
        "risk_free_rate_pct": pct(risk_free_rate),
    }


def _trades_in(data: PortfolioData, perf: PerformanceResult, start, end) -> dict:
    trades = data.trades[(data.trades["date"] > start) & (data.trades["date"] <= end)]
    moved = trades[trades["action"].isin(["buy", "sell"])].copy()
    moved["notional_cad"] = np.abs(to_cad(moved["quantity"] * moved["price"], moved["currency"],
                                          moved["date"], perf.usdcad))
    largest = moved.nlargest(5, "notional_cad")
    return {
        "buys": int((moved["action"] == "buy").sum()),
        "sells": int((moved["action"] == "sell").sum()),
        "option_trades": int((moved["asset_type"] == "option").sum()),
        "bought_cad": cad(moved.loc[moved["action"] == "buy", "notional_cad"].sum()),
        "sold_cad": cad(moved.loc[moved["action"] == "sell", "notional_cad"].sum()),
        "largest_trades": [{"date": f"{r.date:%Y-%m-%d}", "position": readable_ticker(r.ticker),
                            "action": r.action, "quantity": num(r.quantity, 4),
                            "notional_cad": cad(r.notional_cad)} for r in largest.itertuples()],
    }


def _round_trips_in(risk: RiskResult, start, end) -> dict:
    trips = risk.round_trips[(risk.round_trips["closed"] > start) & (risk.round_trips["closed"] <= end)]
    pnl = trips["pnl_cad"]
    return {
        "closed": int(len(trips)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": pct((pnl > 0).mean()) if len(trips) else None,
        "realized_pnl_cad": cad(pnl.sum()),
    }


def _sizing_in(risk: RiskResult, start, end, target: float) -> dict:
    entries = risk.sizing[(risk.sizing["date"] > start) & (risk.sizing["date"] <= end)]
    assessed = entries[entries["over_target"].notna()]
    over = assessed[assessed["over_target"] == True]  # noqa: E712 (column holds True/False/None)
    largest = assessed.dropna(subset=["risk_pct"]).nlargest(3, "risk_pct")
    return {
        "target_risk_per_trade_pct": pct(target),
        "entries_assessed": int(len(assessed)),
        "within_target": int(len(assessed) - len(over)),
        "over_target": int(len(over)),
        "median_risk_pct": pct(assessed["risk_pct"].median()) if len(assessed) else None,
        "largest_entries": [{"date": f"{r.date:%Y-%m-%d}", "position": readable_ticker(r.ticker),
                             "risk_pct_of_equity": pct(r.risk_pct), "method": r.method}
                            for r in largest.itertuples()],
        "method_note": "Without a hard stop, risk = shares x 2 x ATR(14) before entry; "
                       "long options risk their premium.",
    }


def _clean(value):
    """JSON-safe: numpy numbers to Python numbers, NaN to null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def _portfolio_description(data: PortfolioData, config: dict) -> dict:
    components = config["benchmark"]["components"]
    return {
        "type": "Personal self-directed portfolio",
        "accounts": data.accounts,
        "base_currency": config["base_currency"],
        "benchmark": " + ".join(f"{pct(c['weight']):g}% {c['name']}" for c in components)
                     + ", total return, rebalanced monthly",
        "return_method": "time-weighted (deposits and withdrawals excluded)",
    }


# ---------------------------------------------------------------------------
# The two summaries sent to the model
# ---------------------------------------------------------------------------

def build_monthly_summary(data: PortfolioData, perf: PerformanceResult, risk: RiskResult,
                          config: dict, month: pd.Period, info: pd.DataFrame,
                          macro_notes: str = "") -> dict:
    """Everything the monthly letter may mention, already calculated and rounded."""
    start, end = month_window(perf, month)
    row = perf.monthly.loc[month]
    values = perf.values["total"]
    flows = perf.flows.loc[(perf.flows.index > start) & (perf.flows.index <= end)].sum()
    start_value, end_value = float(values.loc[start]), float(values.loc[end])
    gain = end_value - start_value - flows
    pnl = position_pnl(data, perf, start, end)

    before, after = _exposure_snapshot(perf, info, start), _exposure_snapshot(perf, info, end)
    month_returns = perf.daily_returns.loc[start:end].iloc[1:]
    month_growth = perf.growth["portfolio"].loc[start:end]

    return _clean({
        "portfolio": _portfolio_description(data, config),
        "month": str(month),
        "month_is_partial": bool(row["partial"]),
        "period": {"from_close_of": f"{start:%Y-%m-%d}", "to_close_of": f"{end:%Y-%m-%d}"},
        "returns": {
            "portfolio_month_pct": pct(row["portfolio"]),
            "benchmark_month_pct": pct(row["benchmark"]),
            "excess_month_pct_points": pct(row["excess"]),
            "portfolio_since_start_pct": pct(compound(perf.daily_returns.loc[:end])),
            "benchmark_since_start_pct": pct(compound(perf.benchmark_returns.loc[:end])),
            "history_starts": perf.summary["start_date"],
            "best_day_pct": pct(month_returns.max()),
            "worst_day_pct": pct(month_returns.min()),
        },
        "values_cad": {
            "start_value": cad(start_value),
            "end_value": cad(end_value),
            "net_deposits": cad(flows),
            "investment_gain": cad(gain),
            "gain_from_positions": cad(pnl.sum()),
            "gain_from_dividends_interest_fx_and_other": cad(gain - pnl.sum()),
        },
        "top_contributors": _ranked(pnl, start_value, 3, best=True),
        "top_detractors": _ranked(pnl, start_value, 3, best=False),
        "trading_activity": _trades_in(data, perf, start, end),
        "closed_round_trips": _round_trips_in(risk, start, end),
        "risk": {
            "this_month": {
                "volatility_annualized_pct": pct(annualized_volatility(month_returns))
                if len(month_returns) > 1 else None,
                "max_drawdown_within_month_pct": pct(max_drawdown(month_growth)["max_drawdown"]),
            },
            "since_start_to_month_end": _risk_to(perf, end, config["risk"]["risk_free_rate"]),
            "position_sizing_this_month": _sizing_in(risk, start, end,
                                                     config["risk"]["target_risk_per_trade"]),
        },
        "exposure": {
            "start_of_month": before,
            "end_of_month": after,
            "sector_weight_change_pct_points": _weight_changes(before, after, "sector_weights_pct"),
            "currency_weight_change_pct_points": _weight_changes(before, after, "currency_weights_pct"),
        },
        "data_notes": perf.warnings + risk.warnings,
        "manager_macro_notes": macro_notes.strip() or None,
    })


# sizing_summary fields that are fractions (0.01 = 1%); the rest are counts.
FRACTION_FIELDS = {"target_risk_per_trade", "share_within_target", "median_risk_pct", "largest_risk_pct"}


def tickers_held(perf: PerformanceResult, days) -> list[str]:
    """Stock tickers with a position on any of the given days (for sector lookups)."""
    shares = perf.shares.loc[list(days)]
    held = shares.loc[:, (shares != 0).any()]
    return sorted(set(held.columns.get_level_values("ticker")))


def build_portfolio_context(data: PortfolioData, perf: PerformanceResult, risk: RiskResult,
                            config: dict, info: pd.DataFrame) -> dict:
    """The whole calculated picture, for answering questions on the Ask page."""
    s = perf.summary
    monthly = []
    for month, row in perf.monthly.iterrows():
        start, end = month_window(perf, month)
        pnl = position_pnl(data, perf, start, end)
        start_value = float(perf.values["total"].loc[start])
        monthly.append({
            "month": str(month), "partial": bool(row["partial"]),
            "portfolio_pct": pct(row["portfolio"]), "benchmark_pct": pct(row["benchmark"]),
            "excess_pct_points": pct(row["excess"]),
            "top_contributors": _ranked(pnl, start_value, 3, best=True),
            "top_detractors": _ranked(pnl, start_value, 3, best=False),
        })

    over = risk.sizing[risk.sizing["over_target"] == True]  # noqa: E712
    latest = perf.values.index[-1]
    return _clean({
        "portfolio": _portfolio_description(data, config),
        "period": {"start": s["start_date"], "end": s["end_date"], "days": s["days"]},
        "performance": {
            "total_return_pct": pct(s["total_return"]),
            "benchmark_total_return_pct": pct(s["benchmark_total_return"]),
            "excess_return_pct_points": pct(s["excess_return"]),
            "annualized_return_pct": pct(s["annualized_return"]),
            "annualized_note": "Not annualized for periods under a year." if s["annualized_return"] is None else None,
            "starting_value_cad": cad(s["starting_value_cad"]),
            "ending_value_cad": cad(s["ending_value_cad"]),
            "net_deposits_cad": cad(s["net_deposits_cad"]),
            "investment_gain_cad": cad(s["investment_gain_cad"]),
            "best_month": s["best_month"]["month"], "best_month_pct": pct(s["best_month"]["return"]),
            "worst_month": s["worst_month"]["month"], "worst_month_pct": pct(s["worst_month"]["return"]),
        },
        "monthly": monthly,
        "risk": _risk_to(perf, latest, config["risk"]["risk_free_rate"]),
        "trade_statistics": {
            name: {"round_trips": t["round_trips"], "win_rate_pct": pct(t["win_rate"]),
                   "average_win_cad": cad(t["average_win_cad"]), "average_loss_cad": cad(t["average_loss_cad"]),
                   "payoff_ratio": num(t["payoff_ratio"]), "profit_factor": num(t["profit_factor"]),
                   "net_realized_pnl_cad": cad(t["net_realized_pnl_cad"]),
                   "largest_win_cad": cad(t["largest_win_cad"]), "largest_loss_cad": cad(t["largest_loss_cad"])}
            for name, t in {"all": risk.trade_stats, **risk.trade_stats_by_type}.items()
        },
        "position_sizing": {
            **{k + "_pct" if k in FRACTION_FIELDS and not k.endswith("_pct") else k:
               pct(v) if k in FRACTION_FIELDS else v
               for k, v in risk.sizing_summary.items()},
            "entries_over_target_latest_25": [
                {"date": f"{e.date:%Y-%m-%d}", "account": e.account, "position": readable_ticker(e.ticker),
                 "risk_pct_of_equity": pct(e.risk_pct), "method": e.method}
                for e in over.sort_values("date", ascending=False).head(25).itertuples()],
        },
        "current_exposure": _exposure_snapshot(perf, info, latest),
        "account_values_cad": {a: cad(v) for a, v in perf.account_values.iloc[-1].items()},
        "data_notes": perf.warnings + risk.warnings,
    })


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LETTER_SYSTEM_PROMPT = f"""You are the manager of a personal investment portfolio, writing the monthly letter to its investors in the style of a fund manager's investor update. The reader is financially literate.

You receive one JSON summary. Every figure in it was calculated in code from the portfolio's trade history and market prices. Your job is to explain those figures clearly and honestly, not to calculate.

Rules for figures:
- Use only numbers that appear in the JSON. Do not compute new ones: no sums, differences, averages, ratios, annualizing or currency conversions. If a figure you want is not in the JSON, leave it out or say the data doesn't cover it.
- Quote numbers as they appear: percentages with the same decimals (a field ending in _pct holds a percent value, so 24.92 means 24.92%), CAD amounts as whole dollars with thousands separators.
- Do not invent prices, news, events, company developments or market commentary. You know nothing about markets beyond the JSON and the manager's notes.

Structure, in this order, with these Markdown headings:
## Performance summary
## What drove returns
## Positioning and macro view
## Risk discipline
## Outlook

Guidance by section:
- Performance summary: the month's return against the benchmark, the change in value, and context since the start of the history.
- What drove returns: the top contributors and detractors, trading activity, and closed round trips.
- Positioning and macro view: changes in exposure and leverage. The macro view comes only from manager_macro_notes. If that field is null, say in one sentence that no macro view was recorded this month, and do not supply one.
- Risk discipline: volatility, drawdown, and how many new positions stayed within the per-trade risk target. Be candid when the numbers show risk was high. If data_notes contains a caveat that bears on a figure you quote (for example about the Sharpe ratio), include it.
- Outlook: positioning going forward in qualitative terms, drawing on the manager's notes where given. No forecasts and no figures that are not in the JSON.

Style: about one page (450 to 650 words), plain and direct, no hype, written as the manager in the first person. If the month is partial, say so.

End with this line, in italics: {DISCLAIMER}"""

ASK_SYSTEM_PROMPT = """You answer questions about a personal investment portfolio, using only the calculated data in the JSON below. Every figure in it was computed in code from the portfolio's trades and market prices.

Rules:
- Answer only from the JSON. Quote figures exactly as they appear (fields ending in _pct are percent values). Do not compute new figures: no sums, differences, averages or conversions.
- If the question can't be answered from the JSON, say clearly: "I can't answer that from the calculated data", and say briefly what data would be needed. Don't guess.
- Don't give personalized investment advice or recommendations to buy or sell. You may describe what the data shows.
- Don't invent market news, prices or events.
- Keep answers short and direct, using a small table or list when it helps.

<portfolio_data>
{context}
</portfolio_data>"""


def letter_request(summary: dict) -> list[dict]:
    """The user turn for the letter: the JSON summary and the instruction."""
    payload = json.dumps(summary, indent=2)
    return [{"role": "user", "content": f"Here is the calculated data for {summary['month']}:\n\n"
                                        f"<summary>\n{payload}\n</summary>\n\n"
                                        "Write this month's letter."}]


def ask_system_prompt(context: dict) -> str:
    return ASK_SYSTEM_PROMPT.format(context=json.dumps(context, indent=2))


# ---------------------------------------------------------------------------
# Calling the model
# ---------------------------------------------------------------------------

class ModelRefused(Exception):
    """The model (and its fallback) declined to answer."""


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def stream_text(client: anthropic.Anthropic, config: dict, system: str,
                messages: list[dict]) -> Iterator[str]:
    """Stream Claude's reply as text chunks.

    Thinking stays on (the model decides how much to reason; effort is set in
    config.yaml). If the model declines, the API retries on a fallback model;
    if that also declines, ModelRefused is raised.
    """
    settings = config["ai"]
    with client.beta.messages.stream(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        system=system,
        messages=messages,
        output_config={"effort": settings["effort"]},
        cache_control={"type": "ephemeral"},   # reuse the large, unchanging prefix across questions
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        raise ModelRefused("The model declined to answer this request.")
    if final.stop_reason == "max_tokens":
        yield "\n\n*[Stopped at the length limit set in config.yaml.]*"


# ---------------------------------------------------------------------------
# Checking the output: every number must come from the JSON
# ---------------------------------------------------------------------------

# A number as written in prose: optional sign or $, digits with optional thousands
# separators and decimals, then an optional k/M suffix and/or %. Not part of a word
# (so the 3 in "Q3" is ignored).
NUMBER = re.compile(r"(?<![\w.])[-−+]?\$?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM](?![a-zA-Z]))?%?")


def parse_number(text: str) -> tuple[float, float]:
    """'$69.6k' -> (69600.0, 50.0): the value and the rounding tolerance it implies."""
    clean = text.replace("−", "-").replace("$", "").replace(",", "").replace("%", "").strip()
    multiplier = 1.0
    if clean[-1:].lower() in ("k", "m"):
        multiplier = 1_000.0 if clean[-1].lower() == "k" else 1_000_000.0
        clean = clean[:-1].strip()
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return float(clean) * multiplier, 0.5 * 10 ** -decimals * multiplier


def allowed_numbers(data) -> list[float]:
    """Every number in the JSON: values, numbers inside strings (dates, labels), and list lengths."""
    found = []
    if isinstance(data, dict):
        for v in data.values():
            found += allowed_numbers(v)
    elif isinstance(data, list):
        found.append(float(len(data)))
        for v in data:
            found += allowed_numbers(v)
    elif isinstance(data, bool) or data is None:
        pass
    elif isinstance(data, (int, float)):
        found.append(abs(float(data)))
    elif isinstance(data, str):
        found += [abs(parse_number(m)[0]) for m in NUMBER.findall(data)]
    return found


def check_numbers(text: str, source: dict) -> list[dict]:
    """Numbers in `text` that don't match any number in `source`.

    A number matches if it is a rounding of a source number: "24.9%" matches
    24.92, "69,563" matches 69563, "$69.6k" matches 69563. Signs are ignored, as
    prose often says "fell 5.2%" for -5.2.
    """
    allowed = allowed_numbers(source)
    unverified = []
    for match in NUMBER.finditer(text):
        value, tolerance = parse_number(match.group())
        if not any(abs(abs(value) - a) <= tolerance + 1e-9 for a in allowed):
            context = text[max(0, match.start() - 40): match.end() + 40].replace("\n", " ")
            unverified.append({"number": match.group().strip(), "context": f"…{context}…"})
    return unverified
