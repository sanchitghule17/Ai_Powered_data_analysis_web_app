# ai_eda_app/eda.py
import pandas as pd
import plotly.express as px
import streamlit as st

def show_overview(df: pd.DataFrame) -> None:
    st.write(f'Dataset: {df.shape[0]} rows × {df.shape[1]} columns')
    st.dataframe(df.describe())

def value_counts(df: pd.DataFrame, col: str, top_k: int = 10) -> None:
    vc = df[col].value_counts().head(top_k).reset_index()
    st.dataframe(vc)
    st.plotly_chart(px.bar(vc, x='index', y=col, text=col, template='plotly_white'))
    st.plotly_chart(px.pie(vc, names='index', values=col))

def pairplot(df: pd.DataFrame, title: str) -> None:
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 1:
        st.subheader(title)
        st.plotly_chart(px.scatter_matrix(df, dimensions=num_cols))
