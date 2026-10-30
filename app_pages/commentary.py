"""Commentary: an AI-written monthly letter based only on the calculated JSON summary."""

import streamlit as st

from src.commentary import (
    LETTER_SYSTEM_PROMPT,
    build_monthly_summary,
    has_api_key,
    letter_request,
    month_window,
    stream_text,
    tickers_held,
)
from src.ui import NO_KEY_MESSAGE, ai_error, api_client, current, security_info, show_number_check

a = current()

st.title("Monthly commentary")
st.caption("A fund-style investor letter written by AI from figures calculated in this app. "
           "The AI receives only the JSON below and never calculates numbers itself.")
month = st.selectbox("Month", list(a.perf.monthly.index[::-1]), format_func=str)
notes = st.text_area(
    "Your macro notes for the month (optional)",
    placeholder="e.g. Bank of Canada on hold. I'm cautious on energy and adding to quality tech on weakness.",
    help="Your view drives the 'positioning and macro view' section. Without notes, the "
         "letter says no macro view was recorded rather than inventing one.")

start, end = month_window(a.perf, month)
info = security_info(tuple(tickers_held(a.perf, [start, end])))
summary = build_monthly_summary(a.data, a.perf, a.risk, a.config, month, info, notes)
with st.expander("Data sent to the AI (JSON)"):
    st.json(summary, expanded=2)

if not has_api_key():
    st.info(NO_KEY_MESSAGE, icon=":material/key:")
    st.stop()

key = f"letter:{a.selection}:{month}"
if st.button("Generate letter", type="primary"):
    try:
        text = st.write_stream(stream_text(api_client(), a.config, LETTER_SYSTEM_PROMPT,
                                           letter_request(summary)))
    except Exception as error:  # shown to the user, see ai_error()
        st.error(ai_error(error))
        st.stop()
    st.session_state[key] = {"text": text, "summary": summary}
    st.rerun()

saved = st.session_state.get(key)
if saved:
    if saved["summary"] != summary:
        st.caption("Your notes or the data changed since this letter was written. "
                   "Generate again to update it.")
    with st.container(border=True):
        st.markdown(saved["text"])
    show_number_check(saved["text"], saved["summary"])
    st.download_button("Download letter (.md)", saved["text"], file_name=f"letter-{month}.md")
