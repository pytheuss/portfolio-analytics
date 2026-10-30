"""Ask: questions answered by AI only from the calculated data, with a number check on each answer."""

import streamlit as st

from src.commentary import ask_system_prompt, build_portfolio_context, has_api_key, stream_text, tickers_held
from src.ui import NO_KEY_MESSAGE, ai_error, api_client, current, security_info, show_number_check

PLACEHOLDER = "e.g. Which position hurt most in July?"

a = current()

st.title("Ask about your portfolio")
st.caption("Answers come only from the figures calculated in this app. If a question can't "
           "be answered from them, the assistant says so.")
latest = a.perf.values.index[-1]
context = build_portfolio_context(a.data, a.perf, a.risk, a.config,
                                  security_info(tuple(tickers_held(a.perf, [latest]))))
with st.expander("Data the AI can see (JSON)"):
    st.json(context, expanded=2)

if not has_api_key():
    st.info(NO_KEY_MESSAGE, icon=":material/key:")
    st.chat_input(PLACEHOLDER, disabled=True)
    st.stop()

history = st.session_state.setdefault(f"chat:{a.selection}", [])
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_number_check(message["content"], message["checked_against"], compact=True)

question = st.chat_input(PLACEHOLDER)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        conversation = [{"role": m["role"], "content": m["content"]} for m in history]
        conversation.append({"role": "user", "content": question})
        try:
            answer = st.write_stream(stream_text(api_client(), a.config, ask_system_prompt(context),
                                                 conversation))
        except Exception as error:  # shown to the user, see ai_error()
            st.error(ai_error(error))
            st.stop()
        # Numbers the user typed are allowed to appear in the answer too.
        source = {"data": context, "question": question}
        show_number_check(answer, source, compact=True)
    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": answer, "checked_against": source}]

if history and st.button("Clear conversation"):
    history.clear()
    st.rerun()
