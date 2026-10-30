"""Holdings: current positions, exposure by sector, country and currency, and account values."""

import streamlit as st

from src import charts
from src.ui import current, money, number, show_chart

a = current()
e = a.risk.exposure

st.title("Holdings")
st.caption(f"Positions on {e['date']}, in CAD. Options are shown at cost.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", money(e["equity_cad"]))
c2.metric("Holdings", money(e["gross_holdings_cad"]))
c3.metric("Cash" if e["cash_cad"] >= 0 else "Margin borrowed", money(e["cash_cad"]))
c4.metric("Leverage", f"{number(e['leverage'])}×", help="Holdings ÷ equity. Above 1× means borrowing.")

positions = e["positions"][["ticker", "account", "asset_type", "sector", "country", "currency",
                            "value_cad", "weight"]]
st.dataframe(positions.style.format({"value_cad": "{:,.0f}", "weight": "{:.1%}"}),
             width="stretch", hide_index=True)

st.subheader("Exposure")
# One chart at a time: Plotly charts inside hidden tabs can render at zero width.
view = st.segmented_control("Group by", ["Sector", "Country", "Currency"], default="Sector",
                            label_visibility="collapsed") or "Sector"
show_chart(charts.exposure_chart(e[view.lower()], a.mode))
st.caption("Weights are shares of holdings, before cash. Net currency exposure "
           "(holdings + cash in that currency):")
currency = e["currency"].rename(columns={"value_cad": "holdings_cad", "net_cad": "net_exposure_cad"})
st.dataframe(currency.style.format("{:,.0f}", subset=["holdings_cad", "cash_cad", "net_exposure_cad"])
             .format("{:.1%}", subset=["weight"]), width="stretch")

st.subheader("Accounts")
latest = a.perf.account_values.iloc[-1].rename("value_cad").to_frame()
st.dataframe(latest.style.format("{:,.0f}"), width="stretch")
