"""Print the Phase 2 performance summary in the terminal.

    python scripts/performance_report.py                        # sample data
    python scripts/performance_report.py --wealthsimple         # your files in data/private/
    python scripts/performance_report.py --wealthsimple --account TFSA
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.ingest import select_accounts  # noqa: E402
from src.performance import analyze_performance  # noqa: E402
from src.sources import load_configured_wealthsimple, load_sample  # noqa: E402


def pct(x) -> str:
    return "n/a (<1 year)" if x is None else f"{x:+.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wealthsimple", action="store_true", help="use your Wealthsimple files")
    parser.add_argument("--account", action="append", help="limit to an account (repeatable)")
    args = parser.parse_args()

    config = load_config()
    data = load_configured_wealthsimple(config) if args.wealthsimple else load_sample(config)
    if args.account:
        data = select_accounts(data, args.account)
    result = analyze_performance(data, config)
    s = result.summary

    print(f"\nPeriod: {s['start_date']} to {s['end_date']} ({s['days']} days)")
    print(f"Accounts: {', '.join(data.accounts)}\n")
    print(f"Starting value     {s['starting_value_cad']:>14,.2f} CAD")
    print(f"Net deposits       {s['net_deposits_cad']:>14,.2f} CAD")
    print(f"Investment gain    {s['investment_gain_cad']:>14,.2f} CAD")
    print(f"Ending value       {s['ending_value_cad']:>14,.2f} CAD\n")
    print(f"Total return (TWR)      {pct(s['total_return'])}")
    print(f"Benchmark               {pct(s['benchmark_total_return'])}")
    print(f"Excess return           {pct(s['excess_return'])}")
    print(f"Annualized              {pct(s['annualized_return'])}")
    print(f"Benchmark annualized    {pct(s['benchmark_annualized_return'])}\n")
    for label in ["best_month", "worst_month"]:
        m = s[label]
        note = " (partial month)" if m["partial"] else ""
        print(f"{label.replace('_', ' ').title():<12} {m['month']}  {m['return']:+.2%} "
              f"vs benchmark {m['benchmark']:+.2%}{note}")

    print("\nMonthly returns")
    table = result.monthly.copy()
    for col in ["portfolio", "benchmark", "excess"]:
        table[col] = table[col].map(lambda x: f"{x:+.2%}")
    print(table.to_string())

    if result.warnings:
        print("\nWarnings")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
