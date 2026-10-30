"""Portfolio Analytics & Commentary: the Streamlit app.

Run with:  streamlit run app.py

This file picks the data (sidebar), runs the analysis and hands it to the pages
in app_pages/. Every number shown is calculated in src/ (plain Python); the AI
pages only write about numbers that have already been calculated.
"""

import streamlit as st
from dotenv import load_dotenv

from src.config import PROJECT_ROOT, load_config
from src.ingest import select_accounts
from src.performance import analyze_performance
from src.risk import analyze_risk
from src.sources import (
    load_configured_wealthsimple,
    load_sample,
    load_uploaded,
    private_files_available,
)
from src.ui import SIX_HOURS, Analysis, set_current, theme_mode

load_dotenv(PROJECT_ROOT / ".env")
st.set_page_config(page_title="Portfolio Analytics", page_icon=":material/monitoring:",
                   layout="wide")
CONFIG = load_config()


# Cached: files are read and prices fetched once per data selection.
@st.cache_data(show_spinner="Reading your files...", ttl=SIX_HOURS)
def load_portfolio(source: str, uploads: tuple):
    if source == "sample":
        return load_sample(CONFIG)
    if source == "private":
        return load_configured_wealthsimple(CONFIG)
    return load_uploaded(*uploads)


@st.cache_data(show_spinner="Fetching prices and calculating...", ttl=SIX_HOURS)
def analyze(source: str, uploads: tuple, accounts: tuple):
    data = select_accounts(load_portfolio(source, uploads), list(accounts))
    perf = analyze_performance(data, CONFIG)
    risk = analyze_risk(data, perf, CONFIG)
    return data, perf, risk


def choose_source() -> tuple[str, tuple]:
    """Sidebar: sample data, your files in data/private/, or uploaded files."""
    st.sidebar.header("Data")
    has_private = private_files_available(CONFIG)
    use_sample = st.sidebar.toggle("Use sample data", value=not has_private,
                                   help="Fake trades at real historical prices.")
    if use_sample:
        return "sample", ()

    options = (["My files in data/private"] if has_private else []) + ["Upload files"]
    choice = st.sidebar.radio("Source", options) if len(options) > 1 else options[0]
    if choice != "Upload files":
        return "private", ()

    trades = st.sidebar.file_uploader(
        "Trade history (CSV)", type="csv",
        help="Wealthsimple activities export, or a trade file in the standard format "
             "(date, ticker, action, quantity, price, currency, fees).")
    holdings = st.sidebar.file_uploader(
        "Holdings report (optional)", type="csv",
        help="Wealthsimple holdings report. Used to work out positions held before the "
             "trade history starts.")
    flows = st.sidebar.file_uploader(
        "Deposits and withdrawals (optional)", type="csv",
        help="For simple trade files: date, amount, currency. Wealthsimple exports "
             "already include these.")
    cutoff = st.sidebar.text_input("Holdings report taken at (optional)",
                                   placeholder="2026-06-02 12:00",
                                   help="Only needed if the report was taken during the day.")
    if trades is None:
        st.info("Upload a trade file in the sidebar, or switch on **Use sample data**.")
        st.stop()
    return "upload", (trades.getvalue(), holdings.getvalue() if holdings else None,
                      flows.getvalue() if flows else None, cutoff or None)


source, uploads = choose_source()
try:
    available = load_portfolio(source, uploads).accounts
    st.sidebar.header("Accounts")
    chosen = tuple(st.sidebar.multiselect("Include", available, default=available,
                                          label_visibility="collapsed"))
    if not chosen:
        st.warning("Pick at least one account in the sidebar.")
        st.stop()
    data, perf, risk = analyze(source, uploads, chosen)
except ValueError as error:
    st.error(f"Couldn't analyse this data.\n\n{error}")
    st.stop()

set_current(Analysis(data=data, perf=perf, risk=risk, config=CONFIG,
                     selection=f"{source}:{','.join(chosen)}", mode=theme_mode()))

st.navigation([
    st.Page("app_pages/overview.py", title="Overview", icon=":material/insights:", default=True),
    st.Page("app_pages/risk.py", title="Risk", icon=":material/shield:"),
    st.Page("app_pages/holdings.py", title="Holdings", icon=":material/pie_chart:"),
    st.Page("app_pages/commentary.py", title="Commentary", icon=":material/article:"),
    st.Page("app_pages/ask.py", title="Ask", icon=":material/chat:"),
]).run()
