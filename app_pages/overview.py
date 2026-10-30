"""Overview: headline numbers, growth vs the benchmark, drawdown and monthly returns."""

import streamlit as st

from src import charts
from src.ui import current, money, number, pct, show_chart, show_warnings

a = current()
s, r = a.perf.summary, a.risk.summary

st.title("Overview")
st.caption(f"{s['start_date']} to {s['end_date']} · {', '.join(a.data.accounts)} · "
           "all values in CAD · returns are time-weighted")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio value", money(s["ending_value_cad"]),
          f"{money(s['investment_gain_cad'], signed=True)} gain",
          help="Ending value minus starting value minus net deposits.")
c2.metric("Total return", pct(s["total_return"]), f"{pct(s['excess_return'])} vs bench.",
          help="Time-weighted: deposits and withdrawals don't count as gains or losses.")
c3.metric("Benchmark", pct(s["benchmark_total_return"]),
          help="50% S&P/TSX (XIC) + 50% S&P 500 in CAD (VFV), dividends reinvested, "
               "rebalanced monthly.")
c4.metric("Annualized return", pct(s["annualized_return"]),
          help="Shown only for periods of a year or more.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max drawdown", pct(r["drawdown"]["max_drawdown"], signed=False))
c2.metric("Sharpe ratio", number(r["sharpe_ratio"]),
          help=f"Risk-free rate {pct(r['risk_free_rate'], signed=False)}.")
c3.metric("Volatility (annualized)", pct(r["volatility"], signed=False))
c4.metric("Net deposits", money(s["net_deposits_cad"]))

st.subheader("Growth: portfolio vs benchmark")
show_chart(charts.growth_chart(a.perf.growth, a.mode))

st.subheader("Drawdown")
dd = r["drawdown"]
recovered = f"recovered {dd['recovery_date']}." if dd["recovery_date"] else "not yet recovered."
st.caption(f"Largest fall {pct(dd['max_drawdown'], signed=False)}: peak {dd['peak_date']}, "
           f"trough {dd['trough_date']}, {recovered}")
benchmark_growth = a.perf.growth["benchmark"]
show_chart(charts.drawdown_chart(a.risk.drawdown, benchmark_growth / benchmark_growth.cummax() - 1,
                                 a.mode))

st.subheader("Monthly returns")
best, worst = s["best_month"], s["worst_month"]
st.caption(f"Best month {best['month']} ({pct(best['return'])}), "
           f"worst month {worst['month']} ({pct(worst['return'])}).")
show_chart(charts.monthly_chart(a.perf.monthly, a.mode))
with st.expander("Monthly returns table"):
    table = a.perf.monthly.copy()
    table.index = table.index.astype(str)
    st.dataframe(table.style.format({c: "{:+.2%}" for c in ["portfolio", "benchmark", "excess"]}),
                 width="stretch")

show_warnings(a.perf.warnings)
