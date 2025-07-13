import streamlit as st
import pandas as pd
import io

def quick_eda(df):
    """
    Performs a quick exploratory data analysis of a pandas DataFrame using Streamlit.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    st.subheader("📋 DataFrame Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("🔎 First 5 Rows")
    st.dataframe(df.head())

    st.subheader("🔍 Last 5 Rows")
    st.dataframe(df.tail())

    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe().T)

    st.subheader("❗ Missing Values Count")
    st.dataframe(df.isnull().sum().sort_values(ascending=False).to_frame("Missing Count"))

    st.subheader("🔢 Unique Values Count")
    st.dataframe(df.nunique().sort_values(ascending=False).to_frame("Unique Count"))
