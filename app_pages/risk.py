"""Risk: volatility, Sharpe, beta, drawdown, trade statistics and the position-sizing check."""

import pandas as pd
import streamlit as st

from src import charts
from src.ui import current, money, number, pct, show_chart, show_warnings


def sizing_table(sizing: pd.DataFrame, only_flagged: bool):
    """Entries with a status column; rows over target get a red tint (status is also in words)."""
    table = sizing.copy()
    table["status"] = (table["over_target"].map({True: "⚠ Over target", False: "✓ Within"})
                       .fillna("– Not assessed"))
    if only_flagged:
        table = table[table["over_target"] == True]  # noqa: E712 (column holds True/False/None)
    table = table.sort_values("date", ascending=False)
    columns = ["status", "date", "account", "ticker", "asset_type", "quantity", "price", "currency",
               "method", "stop_price", "risk_cad", "equity_cad", "risk_pct"]
    tint = "background-color: rgba(208, 59, 59, 0.14)"
    return (table[columns].style
            .apply(lambda row: [tint if row["status"].startswith("⚠") else ""] * len(row), axis=1)
            .format({"date": "{:%Y-%m-%d}", "quantity": "{:,.4g}", "price": "{:,.2f}",
                     "stop_price": "{:,.2f}", "risk_cad": "{:,.0f}", "equity_cad": "{:,.0f}",
                     "risk_pct": "{:.2%}"}, na_rep="–"))


a = current()
r, settings = a.risk.summary, a.config["risk"]

st.title("Risk")
st.caption(f"Risk-free rate {pct(r['risk_free_rate'], signed=False)} "
           "(Government of Canada 3-month T-bill, set in config.yaml).")

# Benchmark figures go in a caption under each metric: a delta arrow would read as a gain.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Volatility", pct(r["volatility"], signed=False))
c1.caption(f"Benchmark {pct(r['benchmark_volatility'], signed=False)}")
c2.metric("Sharpe ratio", number(r["sharpe_ratio"]))
c2.caption(f"Benchmark {number(r['benchmark_sharpe_ratio'])}")
c3.metric("Beta", number(r["beta"]), help="How much the portfolio moved per 1% benchmark move.")
c3.caption(f"Correlation {number(r['correlation'])}")
c4.metric("Max drawdown", pct(r["drawdown"]["max_drawdown"], signed=False))
c4.caption(f"Benchmark {pct(r['benchmark_drawdown']['max_drawdown'], signed=False)}")
for warning in a.risk.warnings:
    if warning.startswith("Sharpe"):
        st.warning(warning, icon=":material/warning:")

st.subheader("Trade statistics")
st.caption("A round trip runs from opening a position to closing it. Profit uses average "
           "cost (adjusted cost base), in CAD.")
groups = {"All": a.risk.trade_stats,
          **{k.title() + "s": v for k, v in a.risk.trade_stats_by_type.items()}}
stats = pd.DataFrame({
    name: {"Round trips": t["round_trips"], "Win rate": pct(t["win_rate"], signed=False),
           "Average win": money(t["average_win_cad"]), "Average loss": money(t["average_loss_cad"]),
           "Payoff ratio": number(t["payoff_ratio"]), "Profit factor": number(t["profit_factor"]),
           "Net realized P&L": money(t["net_realized_pnl_cad"], signed=True)}
    for name, t in groups.items()
}).T
st.dataframe(stats, width="stretch")

st.subheader("Position sizing")
z = a.risk.sizing_summary
target = z["target_risk_per_trade"]
st.caption(f"Risk at entry as a share of account equity. Without a hard stop, the stop is "
           f"estimated at entry − {settings['atr_multiple']:g} × ATR({settings['atr_period']}). "
           "Long options risk their premium.")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Entries within {pct(target, signed=False)}", pct(z["share_within_target"], signed=False))
c1.caption(f"{z['within_target']} of {z['assessed']} assessed")
c2.metric("Over target", z["over_target"])
c3.metric("Median risk per entry", pct(z["median_risk_pct"], signed=False))
c4.metric("Largest", pct(z["largest_risk_pct"], signed=False))
show_chart(charts.risk_histogram(a.risk.sizing["risk_pct"].dropna(), target, a.mode))

only_flagged = st.toggle("Show only entries over target")
st.dataframe(sizing_table(a.risk.sizing, only_flagged), width="stretch", hide_index=True)

with st.expander("All round trips"):
    trips = a.risk.round_trips.sort_values("closed", ascending=False)
    st.dataframe(trips.style.format({"pnl_cad": "{:+,.2f}", "opened": "{:%Y-%m-%d}",
                                     "closed": "{:%Y-%m-%d}"}),
                 width="stretch", hide_index=True)
show_warnings([w for w in a.risk.warnings if not w.startswith("Sharpe")])
