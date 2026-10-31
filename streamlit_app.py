import os
import streamlit as st

# Use the key from Streamlit Secrets when deployed.
# This must run before graph.py is imported, because graph.py creates the LLM.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

from graph import graph

st.title("Old Product Exit Assistant")
st.write(
    "Ask about your product. Include the demand level "
    "(High, Medium, or Low) and the technology gap (Small, Medium, or Large)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Type your question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            out = graph.invoke({"question": question})
            text = out["answer"]
            if out.get("action"):
                text = f"**Action:** {out['action']}\n\n{text}"
        except Exception as e:
            text = f"Sorry, something went wrong: {e}"
        st.write(text)

    st.session_state.messages.append({"role": "assistant", "content": text})