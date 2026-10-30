"""Tests for the AI layer: the number check, prompts, and the API wrapper (no real API calls)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.commentary import (
    DISCLAIMER,
    LETTER_SYSTEM_PROMPT,
    ModelRefused,
    ask_system_prompt,
    check_numbers,
    letter_request,
    parse_number,
    readable_ticker,
    stream_text,
)

SUMMARY = {
    "month": "2026-04",
    "returns": {"portfolio_month_pct": 5.44, "benchmark_month_pct": 5.9},
    "values_cad": {"end_value": 68654, "investment_gain": -3543},
    "top_contributors": [{"position": "GOOGL", "pnl_cad": 1507}],
    "portfolio": {"benchmark": "50% S&P/TSX (XIC) + 50% S&P 500 in CAD (VFV)"},
}


# --- the number check -------------------------------------------------------------

def test_numbers_from_the_json_pass():
    text = ("In April 2026 the portfolio returned 5.44% against 5.9% for the benchmark. "
            "Value ended at $68,654. GOOGL added 1,507 CAD. The S&P 500 half...")
    assert check_numbers(text, SUMMARY) == []


def test_rounded_numbers_pass():
    # 5.4% and 5% are roundings of 5.44; $68.7k is a rounding of 68,654.
    assert check_numbers("Up 5.4%, about 5%, to $68.7k.", SUMMARY) == []


def test_sign_is_ignored():
    # The JSON has -3,543; prose says "lost 3,543".
    assert check_numbers("The portfolio lost 3,543 CAD.", SUMMARY) == []


def test_invented_number_is_flagged():
    flagged = check_numbers("The portfolio returned 5.44%, and oil rose 12.3%.", SUMMARY)
    assert [f["number"] for f in flagged] == ["12.3%"]
    assert "oil rose" in flagged[0]["context"]


def test_wrong_rounding_is_flagged():
    # 5.5% is not a rounding of 5.44 (5.44 rounds to 5.4).
    assert [f["number"] for f in check_numbers("Up 5.5%.", SUMMARY)] == ["5.5%"]


def test_digits_inside_words_are_not_numbers():
    assert check_numbers("Q3 and the H2 outlook.", SUMMARY) == []


def test_parse_number_tolerance():
    assert parse_number("$69.6k") == (69_600.0, 50.0)
    assert parse_number("24.92%") == (24.92, pytest.approx(0.005))
    assert parse_number("−1,234") == (-1234.0, 0.5)


# --- labels and prompts -------------------------------------------------------------

def test_option_ticker_is_readable():
    assert readable_ticker("INFQ261009C00013000") == "INFQ 2026-10-09 13 call"
    assert readable_ticker("QQQ260612P00734500") == "QQQ 2026-06-12 734.5 put"
    assert readable_ticker("RY.TO") == "RY.TO"


def test_letter_prompt_carries_the_rules():
    assert "Use only numbers that appear in the JSON" in LETTER_SYSTEM_PROMPT
    for heading in ["Performance summary", "What drove returns", "Positioning and macro view",
                    "Risk discipline", "Outlook"]:
        assert f"## {heading}" in LETTER_SYSTEM_PROMPT
    assert DISCLAIMER in LETTER_SYSTEM_PROMPT
    assert '"portfolio_month_pct": 5.44' in letter_request(SUMMARY)[0]["content"]


def test_ask_prompt_embeds_the_data_and_the_refusal_rule():
    prompt = ask_system_prompt(SUMMARY)
    assert "I can't answer that from the calculated data" in prompt
    assert '"end_value": 68654' in prompt


# --- the API wrapper, with a stand-in client ------------------------------------------

CONFIG = {"ai": {"model": "claude-opus-5-5", "max_tokens": 16000, "effort": "high"}}


def fake_client(chunks, stop_reason="end_turn"):
    """Mimics client.beta.messages.stream(...) closely enough for stream_text."""
    calls = []

    @contextmanager
    def stream(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(text_stream=iter(chunks),
                              get_final_message=lambda: SimpleNamespace(stop_reason=stop_reason))

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream))), calls


def test_stream_text_yields_chunks_and_sends_settings():
    client, calls = fake_client(["Hello ", "world"])
    assert "".join(stream_text(client, CONFIG, "system", [{"role": "user", "content": "hi"}])) == "Hello world"
    sent = calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "high"}
    assert sent["fallbacks"] == "default"
    assert "temperature" not in sent


def test_refusal_raises():
    client, _ = fake_client(["partial"], stop_reason="refusal")
    with pytest.raises(ModelRefused):
        list(stream_text(client, CONFIG, "system", []))


def test_length_limit_is_flagged_in_the_text():
    client, _ = fake_client(["cut"], stop_reason="max_tokens")
    assert "length limit" in "".join(stream_text(client, CONFIG, "system", []))
