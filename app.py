"""Streamlit App for LangGraph Agent (Groq)"""

import streamlit as st
from agent import build_graph
from langchain_core.messages import HumanMessage

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="LangGraph Agent", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph Agent (Groq)")

# -----------------------------
# Initialize Agent Graph (Groq only)
# -----------------------------
graph = build_graph()

# -----------------------------
# Initialize Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Chat Interface
# -----------------------------
user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input.strip():
        # Store user input
        st.session_state.chat_history.append(("User", user_input))

        # Run agent
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        answer = result["messages"][-1].content

        # Store answer
        st.session_state.chat_history.append(("Agent", answer))

# -----------------------------
# Display Chat History
# -----------------------------
for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**🧑 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")