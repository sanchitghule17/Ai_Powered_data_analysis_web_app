import plotly.express as px
import streamlit as st

def value_counts(df, col: str, top_k: int = 20) -> None:
    """Show a bar chart of the `top_k` most frequent values in `col`."""
    vc = (
        df[col]
        .astype(str)                 # make sure everything is string-able
        .value_counts()
        .head(top_k)
        .reset_index(name="count")    # DataFrame with two columns
        .rename(columns={"index": col})
    )

    fig = px.bar(
        vc,
        x=col,
        y="count",
        text="count",
        template="plotly_white",
    )
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
