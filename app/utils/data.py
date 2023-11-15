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


def embed_qas(qas):
    client = openai.OpenAI()

    qas_df = pd.DataFrame(qas, columns=["qa"])

    def get_embedding(text, model=EMBEDDING_MODEL):
        return client.embeddings.create(input=[text], model=model)["data"][0][
            "embedding"
        ]

    qas_df["ada_embedding"] = qas_df.combined.apply(
        lambda x: get_embedding(x, model=EMBEDDING_MODEL)
    )
    qas_df.to_csv("../data/embeddings.csv", index=False)


@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(
        text="Loading and embedding your document – hang tight! This should not take too long."
    ):
        if os.path.is_file("../data/embeddings.csv"):
            df = pd.read_csv("output/embedded_1k_reviews.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
        else:
            qas = partition_text("faqs")
            embeddings = embed_qas(qas)
            df = pd.read_csv("output/embedded_1k_reviews.csv")
            df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)
