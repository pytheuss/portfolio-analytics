"""Shared pieces for the Streamlit pages: the current analysis, formatting and display helpers.

app.py runs the analysis and stores it with set_current(); each page in
app_pages/ reads it back with current(). Pages only format and display results.
"""

from dataclasses import dataclass

import anthropic
import pandas as pd
import streamlit as st

from src.commentary import NUMBER, ModelRefused, check_numbers
from src.ingest import PortfolioData
from src.performance import PerformanceResult
from src.prices import fetch_security_info
from src.risk import RiskResult

SIX_HOURS = 6 * 60 * 60
NO_KEY_MESSAGE = ("AI features need an Anthropic API key. Copy `.env.example` to `.env`, add your "
                  "key after `ANTHROPIC_API_KEY=`, and restart the app. Everything else works without one.")


# ---------------------------------------------------------------------------
# The analysis shared by every page
# ---------------------------------------------------------------------------

@dataclass
class Analysis:
    data: PortfolioData
    perf: PerformanceResult
    risk: RiskResult
    config: dict
    selection: str     # identifies the data source and accounts, for per-selection state
    mode: str          # "light" or "dark", for chart colours


def set_current(analysis: Analysis) -> None:
    st.session_state["analysis"] = analysis


def current() -> Analysis:
    return st.session_state["analysis"]


def theme_mode() -> str:
    """'dark' or 'light', following the viewer's Streamlit theme."""
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) or "light"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(x, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def money(x, signed: bool = False) -> str:
    """Whole CAD amounts; the page captions say values are in CAD."""
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+,.0f}" if signed else f"{x:,.0f}"


def number(x, digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_chart(fig) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})


def show_warnings(warnings: list[str]) -> None:
    if warnings:
        with st.expander(f"Data notes ({len(warnings)})"):
            for warning in warnings:
                st.markdown(f"- {warning}")


def show_number_check(text: str, source: dict, compact: bool = False) -> None:
    """Flag any number in AI-written text that isn't in the data it was given."""
    unverified = check_numbers(text, source)
    total = len(NUMBER.findall(text))
    if not unverified:
        message = f"Number check passed: all {total} numbers match the calculated data."
        if compact:
            st.caption(f":material/verified: {message}")
        else:
            st.success(message, icon=":material/verified:")
        return
    st.warning(f"{len(unverified)} of {total} numbers don't appear in the calculated data. "
               "Check them before relying on this text.", icon=":material/warning:")
    st.dataframe(pd.DataFrame(unverified), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# AI plumbing
# ---------------------------------------------------------------------------

@st.cache_resource
def api_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY, loaded from .env


@st.cache_data(show_spinner="Looking up sectors...", ttl=SIX_HOURS)
def security_info(tickers: tuple):
    return fetch_security_info(list(tickers))


def ai_error(error: Exception) -> str:
    """A plain-language message for each kind of API failure; anything else is a bug and re-raised."""
    if isinstance(error, ModelRefused):
        return "The model declined this request. Try rephrasing it."
    if isinstance(error, anthropic.AuthenticationError):
        return "The API key in `.env` was rejected. Check it and restart the app."
    if isinstance(error, anthropic.RateLimitError):
        return "Rate limit reached. Wait a minute and try again."
    if isinstance(error, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your internet connection."
    if isinstance(error, anthropic.APIStatusError):
        return f"The Anthropic API returned an error ({error.status_code}): {error.message}"
    raise error
