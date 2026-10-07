"""
STEP 5 - A chat page for the agent, built with Streamlit.

Nothing new about AI here: this file only puts a chat window on top of
the agent from step4_agent.py.

Run step2_build_index.py first.

Run:  streamlit run app.py
"""

import streamlit as st

from step4_agent import build_agent, tools_used

st.set_page_config(page_title="VoltKart Shopping Assistant", page_icon="🛒")
st.title("🛒 VoltKart Shopping Assistant")
st.caption("Demo project with a fictional store. Ask about products, store policies or an order.")


# Streamlit re-runs this whole file on every message.
# cache_resource makes sure the agent is built only once.
@st.cache_resource
def load_agent():
    return build_agent()


agent = load_agent()

# session_state survives the re-runs, so the conversation is kept here.
if "history" not in st.session_state:
    st.session_state.history = []    # everything the agent needs (incl. tool calls)
if "chat" not in st.session_state:
    st.session_state.chat = []       # only what we show on screen

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "- Wireless earbuds under Rs. 3000\n"
        "- Can I return a laptop?\n"
        "- Where is my order ORD1002?\n"
        "- A laptop for coding, and how long is its warranty?"
    )
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.chat = []
        st.rerun()

# Show the conversation so far.
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("tools"):
            st.caption("Tools used: " + ", ".join(turn["tools"]))

# Handle a new question.
question = st.chat_input("Type your question")
if question:
    st.session_state.chat.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            old_length = len(st.session_state.history)
            messages = st.session_state.history + [{"role": "user", "content": question}]
            try:
                result = agent.invoke({"messages": messages})
                st.session_state.history = result["messages"]
                answer = result["messages"][-1].text
                tools = tools_used(result["messages"][old_length:])
            except Exception as error:
                answer = f"Something went wrong while calling the model: {error}"
                tools = []

        st.markdown(answer)
        if tools:
            st.caption("Tools used: " + ", ".join(tools))

    st.session_state.chat.append({"role": "assistant", "text": answer, "tools": tools})
