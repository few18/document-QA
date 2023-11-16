import streamlit as st
import ast
import openai
import pandas as pd
import tiktoken
from scipy import spatial
import os
import numpy as np

# Set embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"


def partition_text(file):
    """Partition file of Q and A's into list of Q&A pairs.

    Args:
        file - path to txt with Q&A pairs.

    Returns:
        q_and_a_list (list) - list of Q&A pairs with minimal cleaning.
    """
    with open(f"../data/{file}.txt", "r") as f:
        text = f.read()

    q_and_a_list = ["Q: " + s.strip().replace("\n", " ") for s in text.split("Q:")]
    return q_and_a_list


def get_embedding(text, client, model=EMBEDDING_MODEL):
    return client.embeddings.create(input=[text], model=model)["data"][0]["embedding"]


def embed_qas(client, qas):
    qas_df = pd.DataFrame(qas, columns=["qa"])
    qas_df["ada_embedding"] = qas_df.combined.apply(
        lambda x: get_embedding(x, client, model=EMBEDDING_MODEL)
    )
    qas_df.to_csv("../data/embeddings.csv", index=False)


def cosine_similarity(x, y):
    """Compute cosine similarity between two np.array vectors."""
    return 1 - x @ y.T


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
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False).head(n)
    return results


@st.cache_resource(show_spinner=True)
def load_data(client):
    with st.spinner(
        text="Loading and embedding your document – hang tight! This should not take too long."
    ):
        if os.path.is_file("../data/embeddings.csv"):
            df = pd.read_csv("../data/embeddings.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
        else:
            qas = partition_text("faqs")
            embed_qas(client, qas)
            df = pd.read_csv("../data/embeddings.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
    return df
