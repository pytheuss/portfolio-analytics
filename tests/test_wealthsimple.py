"""Tests for the Wealthsimple activities adapter.

Every row below is invented, but follows a real Wealthsimple convention.
"""

import io

import pandas as pd
import pytest

from src.ingest import HoldingsSnapshot
from src.wealthsimple import load_wealthsimple, option_ticker

HEADER = ("effective_date,effective_time,settlement_date,account_id,account_type,activity_type,"
          "activity_sub_type,description,direction,symbol,underlying symbol,name,currency,"
          "quantity,unit_price,commission,net_cash_amount")

ROWS = [
    # Deposit into the TFSA, and a transfer that only touches the excluded Chequing account.
    '2026-01-02,10:00:00,,T1,TFSA,MoneyMovement,EFT,Deposit,,,,,CAD,5000,,,5000',
    '2026-01-02,10:00:00,,C1,Chequing,MoneyMovement,TRANSFER,Transfer out,,,,,CAD,-5000,,,-5000',
    # Order placed on a Sunday, executed Monday: the trade date must be Monday.
    '2026-01-04,20:00:00,2026-01-06,T1,TFSA,Trade,BUY,'
    '"VEQT - Vanguard: Bought 10.0000 shares (executed at 2026-01-05)",LONG,VEQT,VEQT,Vanguard,CAD,10,40,0,-400',
    # US ETF bought in a CAD account: price is in CAD, "FX Rate" marks it as a US listing.
    '2026-01-05,11:00:00,2026-01-06,T1,TFSA,Trade,BUY,'
    '"VTI - Vanguard: Bought 2.0000 shares (executed at 2026-01-05), FX Rate: 1.40",LONG,VTI,VTI,Vanguard,CAD,2,450,0,-900',
    # Interlisted Canadian symbol Wealthsimple already suffixes.
    '2026-01-06,11:00:00,2026-01-07,M1,Non-registered margin,Trade,SELL,'
    '"HIVE.TO - Hive: Sold 50.0000 shares (executed at 2026-01-06)",LONG,HIVE.TO,HIVE.TO,Hive,CAD,-50,5,0,250',
    # Call option bought, then expired worthless.
    '2026-02-02,10:00:00,2026-02-03,M1,Non-registered margin,Trade,BUY,'
    '"ABC  260220C00010000: Bought 2 contract (executed at 2026-02-02)",LONG,ABC  260220C00010000,ABC,Abc,USD,2,30,0,-60',
    '2026-02-20,13:15:00,,M1,Non-registered margin,OptionExpiry,-,'
    '"ABC  260220C00010000: Expired 2 contract (executed at 2026-02-20)",LONG,ABC  260220C00010000,ABC,Abc,,-2,,,',
    # 2-for-1 split: quantity only, no cash.
    '2026-03-02,21:00:00,,T1,TFSA,CorporateAction,SUBDIVISION,'
    '"VTI - Vanguard: Corrected quantity of shares by 2.0000 (executed at 2026-03-03)",LONG,VTI,VTI,Vanguard,,2,,,',
    # Dividend: cash only, internal to the return.
    '2026-03-10,09:00:00,,T1,TFSA,Dividend,-,VEQT dividend,,VEQT,VEQT,Vanguard,CAD,,,,3.5',
    # Same-day trades either side of a mid-day snapshot.
    '2026-06-02,09:00:00,2026-06-03,M1,Non-registered margin,Trade,BUY,'
    '"XYZ - Xyz: Bought 10.0000 shares (executed at 2026-06-02)",LONG,XYZ,XYZ,Xyz,USD,10,20,0,-200',
    '2026-06-02,14:00:00,2026-06-03,M1,Non-registered margin,Trade,BUY,'
    '"XYZ - Xyz: Bought 5.0000 shares (executed at 2026-06-02)",LONG,XYZ,XYZ,Xyz,USD,5,20,0,-100',
]
FAKE_EXPORT = HEADER + "\n" + "\n".join(ROWS) + "\n"

# Holdings at noon on 2026-06-02. The margin account already held HIVE.TO on Jan 1.
SNAPSHOT = HoldingsSnapshot(
    as_of=pd.Timestamp("2026-06-02 16:00"),
    positions=pd.DataFrame({
        "account": ["TFSA", "TFSA", "Non-registered margin", "Non-registered margin"],
        "symbol": ["VEQT", "VTI", "HIVE.TO", "XYZ"],
        "ticker": ["VEQT.TO", "VTI", "HIVE.TO", "XYZ"],
        "quantity": [10.0, 4.0, 150.0, 10.0],
    }),
    cash=pd.DataFrame({
        "account": ["TFSA", "Non-registered margin", "Non-registered margin"],
        "currency": ["CAD", "CAD", "USD"],
        "amount": [3703.5, 1250.0, 740.0],
    }),
)
NOON = pd.Timestamp("2026-06-02 12:00")


@pytest.fixture
def data():
    return load_wealthsimple(io.StringIO(FAKE_EXPORT), SNAPSHOT, snapshot_cutoff=NOON)


def test_chequing_account_is_excluded(data):
    assert "Chequing" not in data.accounts


def test_trade_date_is_execution_date_not_order_date(data):
    veqt = data.trades[data.trades["ticker"] == "VEQT.TO"].iloc[0]
    assert veqt["date"] == pd.Timestamp("2026-01-05")


def test_listing_and_ticker_resolution(data):
    tickers = set(data.trades["ticker"])
    assert {"VEQT.TO", "VTI", "HIVE.TO", "XYZ", "ABC260220C00010000"} <= tickers


def test_option_trades_and_expiry(data):
    options = data.trades[data.trades["asset_type"] == "option"]
    assert list(options["action"]) == ["buy", "adjust"]
    assert list(options["quantity"]) == [2, -2]          # expiry closes the position


def test_split_is_a_cashless_adjustment(data):
    split = data.trades[(data.trades["ticker"] == "VTI") & (data.trades["action"] == "adjust")]
    assert split["quantity"].tolist() == [2]
    assert split["price"].isna().all()


def test_cash_ledger_kinds(data):
    ledger = data.cash_ledger
    assert ledger.loc[ledger["kind"] == "external", "amount"].tolist() == [5000]
    assert 3.5 in ledger.loc[ledger["activity"] == "Dividend", "amount"].tolist()


def test_opening_positions_respect_mid_day_snapshot(data):
    # HIVE.TO: 150 held at snapshot + 50 sold since Jan 1 = 200 at start.
    # XYZ: the 09:00 buy is in the snapshot, the 14:00 buy is not, so no opening position.
    opening = data.opening_positions.set_index("ticker")["quantity"].to_dict()
    assert opening == {"HIVE.TO": 200.0}
    assert data.warnings == []


def test_opening_cash_backs_out_ledger(data):
    cash = data.opening_cash.set_index(["account", "currency"])["amount"]
    # Margin CAD: 1250 now - 250 from the HIVE sale = 1000 at start.
    # Margin USD: 740 now + 60 option + 200 XYZ (the 14:00 buy is after the cutoff) = 1000.
    assert cash[("Non-registered margin", "CAD")] == pytest.approx(1000.0)
    assert cash[("Non-registered margin", "USD")] == pytest.approx(1000.0)
    assert "TFSA" not in cash.index.get_level_values("account")   # opened in 2026, starts at 0


def test_option_ticker_format():
    assert option_ticker("INFQ  260515C00015000") == "INFQ260515C00015000"
