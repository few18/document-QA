import numpy as np
import openai
import pandas as pd
import streamlit as st
from utils import load_data, ask_gpt

st.set_page_config(
    page_title="Chat with Your Document",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.header("Chat with Your Document")

client = openai.OpenAI()

# Initialize chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your document"}
    ]

# Either load embeddings from storage or create via openai api
faq_embeddings = load_data()

# Prompt for user input and save to chat history
if query := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": query})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


ending = """

May I help you with any other questions you have regarding your document?
"""

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_gpt(query, faq_embeddings, client, n=2)
            output = response + ending
            st.write(output)
            message = {"role": "assistant", "content": output}
            # Add response to message history
            st.session_state.messages.append(output)
