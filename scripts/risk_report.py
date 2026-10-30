"""Print the Phase 3 risk summary in the terminal.

    python scripts/risk_report.py                        # sample data
    python scripts/risk_report.py --wealthsimple         # your files in data/private/
    python scripts/risk_report.py --wealthsimple --account "Non-registered margin"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.ingest import select_accounts  # noqa: E402
from src.performance import analyze_performance  # noqa: E402
from src.risk import analyze_risk  # noqa: E402
from src.sources import load_configured_wealthsimple, load_sample  # noqa: E402


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def money(x) -> str:
    return "n/a" if x is None else f"{x:,.0f} CAD"


def ratio(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wealthsimple", action="store_true", help="use your Wealthsimple files")
    parser.add_argument("--account", action="append", help="limit to an account (repeatable)")
    args = parser.parse_args()

    config = load_config()
    data = load_configured_wealthsimple(config) if args.wealthsimple else load_sample(config)
    if args.account:
        data = select_accounts(data, args.account)
    perf = analyze_performance(data, config)
    risk = analyze_risk(data, perf, config)
    s = risk.summary

    print(f"\nAccounts: {', '.join(data.accounts)}  |  risk-free rate {pct(s['risk_free_rate'])}\n")
    print(f"{'':24}{'Portfolio':>12}{'Benchmark':>12}")
    print(f"{'Annualized volatility':24}{pct(s['volatility']):>12}{pct(s['benchmark_volatility']):>12}")
    print(f"{'Sharpe ratio':24}{ratio(s['sharpe_ratio']):>12}{ratio(s['benchmark_sharpe_ratio']):>12}")
    print(f"{'Max drawdown':24}{pct(s['drawdown']['max_drawdown']):>12}"
          f"{pct(s['benchmark_drawdown']['max_drawdown']):>12}")
    d = s["drawdown"]
    print(f"  peak {d['peak_date']} -> trough {d['trough_date']}, "
          f"recovered {d['recovery_date'] or 'not yet'}")
    print(f"Beta {ratio(s['beta'])}, correlation {ratio(s['correlation'])}\n")

    print("Round trips (closed positions, average-cost P&L)")
    for label, stats in [("All", risk.trade_stats), *risk.trade_stats_by_type.items()]:
        print(f"  {label:<7} {stats['round_trips']:>4} trips  win rate {pct(stats['win_rate'])}  "
              f"avg win {money(stats['average_win_cad'])}  avg loss {money(stats['average_loss_cad'])}  "
              f"profit factor {ratio(stats['profit_factor'])}  net {money(stats['net_realized_pnl_cad'])}")

    z = risk.sizing_summary
    print(f"\nPosition sizing (target {pct(z['target_risk_per_trade'])} of account equity per entry)")
    print(f"  {z['within_target']} of {z['assessed']} assessed entries within target "
          f"({pct(z['share_within_target'])}); median risk {pct(z['median_risk_pct'])}, "
          f"largest {pct(z['largest_risk_pct'])}")
    print(f"  {z['uncapped_written_options']} written options (uncapped risk), "
          f"{z['not_assessed_no_data']} entries without price data")
    worst = risk.sizing.dropna(subset=["risk_pct"]).nlargest(5, "risk_pct")
    print(worst[["date", "account", "ticker", "method", "risk_cad", "equity_cad", "risk_pct"]]
          .to_string(index=False, formatters={"risk_pct": "{:.1%}".format,
                                              "risk_cad": "{:,.0f}".format,
                                              "equity_cad": "{:,.0f}".format}))

    e = risk.exposure
    print(f"\nExposure on {e['date']}: holdings {money(e['gross_holdings_cad'])}, "
          f"cash {money(e['cash_cad'])}, equity {money(e['equity_cad'])}, "
          f"leverage {ratio(e['leverage'])}x")
    for name in ["sector", "country", "currency"]:
        print(f"\nBy {name}")
        print(e[name].to_string(float_format=lambda v: f"{v:,.2f}"))

    for warning in perf.warnings + risk.warnings:
        print(f"- {warning}")


if __name__ == "__main__":
    main()
