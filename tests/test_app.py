"""Smoke test: the app runs on the sample data with no API key and no internet.

Market data is replaced by flat stand-in prices, so this checks that every page
renders without errors, not that the numbers are right (the other tests do that).
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT


def _flat_prices(tickers, start, end=None):
    days = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    return {t: pd.DataFrame({"high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0},
                            index=days) for t in tickers}


def _flat_closes(tickers, start, end=None, total_return=False):
    closes = pd.DataFrame({t: p["close"] for t, p in _flat_prices(tickers, start, end).items()})
    if "CADUSD=X" in closes:
        closes["CADUSD=X"] = 0.72
    return closes


def _no_info(tickers):
    return pd.DataFrame(index=sorted(set(tickers)),
                        columns=["name", "quote_type", "sector", "country"])


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("src.performance.fetch_closes", _flat_closes)
    monkeypatch.setattr("src.risk.fetch_prices", _flat_prices)
    monkeypatch.setattr("src.risk.fetch_security_info", _no_info)
    monkeypatch.setattr("src.ui.fetch_security_info", _no_info)
    # Behave like a fresh clone: no private files, no API key.
    monkeypatch.setattr("src.sources.private_files_available", lambda config: False)
    monkeypatch.setattr("src.commentary.has_api_key", lambda: False)
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)


@pytest.mark.parametrize("page, title", [
    (None, "Overview"),
    ("app_pages/risk.py", "Risk"),
    ("app_pages/holdings.py", "Holdings"),
    ("app_pages/commentary.py", "Monthly commentary"),
    ("app_pages/ask.py", "Ask about your portfolio"),
])
def test_every_page_renders_on_sample_data(app, page, title):
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, app.exception
    assert app.title[0].value == title
    assert app.sidebar.toggle[0].value is True  # "Use sample data" is on by default


def test_ai_pages_explain_the_missing_key(app):
    app.run()
    app.switch_page("app_pages/commentary.py").run()
    assert any("API key" in info.value for info in app.info)
    assert not app.button or all(b.label != "Generate letter" for b in app.button)
