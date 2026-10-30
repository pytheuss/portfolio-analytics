"""Tests for loading a Wealthsimple-style holdings report (all values are fake)."""

import io

import pandas as pd
import pytest

from src.ingest import load_holdings, to_yfinance_ticker

HEADER = (
    "Account Name,Account Type,Account Classification,Account Number,Symbol,Exchange,MIC,"
    "Name,Security Type,Quantity,Position Direction,Market Price,Market Price Currency,"
    "Book Value (CAD),Book Value Currency (CAD),Book Value (Market),"
    "Book Value Currency (Market),Market Value,Market Value Currency,"
    "Market Unrealized Returns,Market Unrealized Returns Currency"
)

FAKE_REPORT = HEADER + """
"TFSA","TFSA","Trade","TEST001","CAD","","","CAD","CURRENCY","10.5","LONG","1","CAD","10.5","CAD","10.5","CAD","10.5","CAD","0","CAD"
"TFSA","TFSA","Trade","TEST001","XEQT","TSX","XTSE","iShares Core Equity ETF Portfolio","EXCHANGE_TRADED_FUND","10.25","LONG","40","CAD","380","CAD","380","CAD","410","CAD","30","CAD"
"Margin ","Non-registered margin","Trade","TEST002","USD","","","USD","CURRENCY","-500","LONG","1","USD","-700","CAD","-500","USD","-500","USD","0","USD"
"Margin ","Non-registered margin","Trade","TEST002","BRK.B","NYSE","XNYS","Berkshire Hathaway","EQUITY","2","LONG","500","USD","1300","CAD","950","USD","1000","USD","50","USD"
"Margin ","Non-registered margin","Trade","TEST002","ABC","TSX-V","XTSX","Small Cap Co","EQUITY","1000","SHORT","0.5","CAD","600","CAD","600","CAD","500","CAD","100","CAD"

"As of 2026-06-02 16:00 GMT-04:00"
"""


def snapshot():
    return load_holdings(io.StringIO(FAKE_REPORT))


def test_footer_becomes_as_of_date():
    assert snapshot().as_of == pd.Timestamp("2026-06-02 16:00")


def test_cash_rows_are_split_out_including_margin_debt():
    cash = snapshot().cash
    assert list(cash["currency"]) == ["CAD", "USD"]
    assert list(cash["amount"]) == [10.5, -500.0]


def test_positions_get_yfinance_tickers_and_clean_values():
    pos = snapshot().positions
    assert list(pos["ticker"]) == ["XEQT.TO", "BRK-B", "ABC.V"]
    assert pos.loc[0, "quantity"] == 10.25            # fractional shares kept
    assert pos.loc[1, "account"] == "Non-registered margin"   # account type is the label
    assert pos.loc[2, "quantity"] == -1000            # SHORT stored as negative


@pytest.mark.parametrize("symbol, mic, expected", [
    ("VEQT", "XTSE", "VEQT.TO"),
    ("CH", "XTSX", "CH.V"),
    ("AAXJ", "XNAS", "AAXJ"),
    ("BBD.B", "XTSE", "BBD-B.TO"),
])
def test_to_yfinance_ticker(symbol, mic, expected):
    assert to_yfinance_ticker(symbol, mic) == expected


def test_unknown_exchange_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="Unknown exchange code"):
        to_yfinance_ticker("XYZ", "XLON")
