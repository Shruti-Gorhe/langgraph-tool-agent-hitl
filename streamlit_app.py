"""Simple reviewer UI for the Week 4 agent and HITL approval flow."""

import json
import uuid

import streamlit as st
from langgraph.types import Command

from src.agent.tool_agent import get_agent
from src.agent.trace import extract_trace, final_text


st.set_page_config(page_title="ZenLabs Week 4 Agent", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Tool-Using Agent")
st.caption("Week 4: tool use + human-in-the-loop approval")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

agent = get_agent()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

with st.sidebar:
    st.subheader("Configuration")
    st.write("RAG strategy:", "`AGENT_RAG_STRATEGY` (default: hierarchical)")
    st.write("RAG method:", "`AGENT_RAG_METHOD` (default: hybrid)")
    st.write("Ollama model:", "`OLLAMA_MODEL` (default: llama3.2)")
    st.divider()
    st.markdown("**Demo queries**")
    st.code("What is the annual home-office stipend for remote workers?")
    st.code("If the home-office stipend is $500 per year, what is the monthly equivalent?")
    st.code("What is EMP001's leave balance?")
    st.code("Update EMP001's leave balance to 20 days.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pending = st.session_state.pending
if pending:
    st.warning("A sensitive tool call is waiting for human approval.")
    action_requests = pending.get("action_requests", [])
    for action in action_requests:
        st.markdown(f"**Tool:** `{action.get('name')}`")
        st.json(action.get("arguments", {}))
        st.caption(action.get("description", "Data write requires approval."))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"} for _ in action_requests]}),
                config=config,
                version="v2",
            )
            st.session_state.pending = None
            st.session_state.last_result = result
            answer = final_text(result)
            if answer:
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            result = agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {"type": "reject", "message": "Human reviewer rejected this data write."}
                            for _ in action_requests
                        ]
                    }
                ),
                config=config,
                version="v2",
            )
            st.session_state.pending = None
            st.session_state.last_result = result
            answer = final_text(result)
            if answer:
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

if prompt := st.chat_input("Ask the enterprise agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            version="v2",
        )
        st.session_state.last_result = result

        interrupts = getattr(result, "interrupts", None)
        if interrupts:
            st.session_state.pending = interrupts[0].value
            st.rerun()

        answer = final_text(result)
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    except Exception as exc:
        st.error(f"Agent error: {exc}")

if st.session_state.last_result:
    with st.expander("🔎 Observable execution trace", expanded=False):
        st.json(extract_trace(st.session_state.last_result))
