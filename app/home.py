import numpy as np
import openai

client = openai.OpenAI()
import pandas as pd
import streamlit as st

# from ..utils import load_data, ask_gpt

import os
import time

# Set embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def partition_text(file):
    """Partition file of Q and A's into list of Q&A pairs.

    Args:
        file - path to txt with Q&A pairs.

    Returns:
        q_and_a_list (list) - list of Q&A pairs with minimal cleaning.
    """
    with open(f"app/data/{file}.txt", "r") as f:
        text = f.read()

    q_and_a_list = ["Q: " + s.strip().replace("\n", " ") for s in text.split("Q:")]
    return q_and_a_list


def get_embedding(text, client, model=EMBEDDING_MODEL):
    time.sleep(20)
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def embed_qas(client, qas):
    qas_df = pd.DataFrame(qas, columns=["qa"])
    qas_df["ada_embedding"] = qas_df.qa.apply(
        lambda x: get_embedding(x, client, model=EMBEDDING_MODEL)
    )
    qas_df.to_csv("app/data/embeddings.csv", index=False)


def cosine_similarity(x, y):
    """Compute cosine similarity between two np.array vectors."""
    return x @ np.array(y).T


def get_n_faq(df, query, n, client):
    """Get n most similar faq pairs to the query.

    Args:
        df (pandas Dataframe) - Dataframe containing text-embedding
        pairs for the faqs.
        query (str) - User query
        n (int) - Number of most similar faq pairs to retrieve
        client - OpenAI client

    Returns:
        similar_faq (pd.Dataframe) - Dataframe of the n most similar pairs
        faqs.
    """
    query_embedding = get_embedding(query, client)
    df["similarity"] = df.ada_embedding.apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False).head(n)
    return results


def load_data(client):
    with st.spinner(
        text="Loading and embedding your document – hang tight! This should not take too long."
    ):
        if os.path.isfile("app/data/embeddings.csv"):
            df = pd.read_csv("app/data/embeddings.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
        else:
            qas = partition_text("faqs")
            embed_qas(client, qas)
            df = pd.read_csv("app/data/embeddings.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
    return df


def construct_prompt(query, similar_faqs):
    faqs = similar_faqs["qa"].tolist()
    newline = "\n\n"
    return f"""```{newline.join(faqs)}```

{query}
"""


def ask_gpt(query, embeddings, client, n):
    """Send query with top n similar matches to gpt 3.5 turbo"""

    similar_faqs = get_n_faq(embeddings, query, n, client)
    message = construct_prompt(query, similar_faqs)
    print(message)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant answering peoples questions. You will use text extracts from frequently asked questions documents to answer the users questions. The extracts are given within the triple single quotes and are separated by new lines. Answer the users question using only the information in these extracts. Do not use any other information you have. If the answer to the users question is not in the extracts you will not give an alternative answer instead state that you could not find the desired answer in the document.",
        },
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0
    )
    return response.choices[0].message.content


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
faq_embeddings = load_data(client)

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
            st.session_state.messages.append(message)
