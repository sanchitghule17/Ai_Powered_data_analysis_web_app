import pandas as pd
import plotly.express as px
import streamlit as st


# ─────────────────────────────────────────────────────────────
# 1. Dataset overview
# ─────────────────────────────────────────────────────────────
def show_overview(df: pd.DataFrame, *, sample_rows: int = 5) -> None:
    """Display basic facts, a preview and summary statistics."""
    st.subheader("🗂️ Dataset overview")
    st.write(f"Rows × Columns: **{df.shape[0]} × {df.shape[1]}**")

    st.markdown("**Preview**")
    st.dataframe(df.head(sample_rows))

    st.markdown("**Summary statistics**")
    st.dataframe(df.describe(include="all").transpose())


# ─────────────────────────────────────────────────────────────
# 2. Top-K value counts for a categorical column
# ─────────────────────────────────────────────────────────────
def value_counts(df: pd.DataFrame, col: str, top_k: int = 10) -> None:
    """Show a table, bar chart and pie chart of the most frequent values."""
    if col not in df.columns:
        st.error(f"Column **{col}** not found.")
        return

    vc = (
        df[col]
        .astype(str)                 # guard against non-strings
        .value_counts()
        .head(top_k)
        .reset_index(name="count")   # → DataFrame with two real columns
        .rename(columns={"index": col})
    )

    st.dataframe(vc)

    bar = px.bar(
        vc,
        x=col,
        y="count",
        text="count",
        template="plotly_white",
    )
    bar.update_layout(xaxis_title=col, yaxis_title="Count")
    st.plotly_chart(bar, use_container_width=True)

    pie = px.pie(vc, names=col, values="count", template="plotly_white")
    st.plotly_chart(pie, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 3. Numeric pair plot
# ─────────────────────────────────────────────────────────────
def pairplot(df: pd.DataFrame, title: str = "Pair Plot") -> None:
    """Scatter-matrix for all numeric columns."""
    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for a pair plot.")
        return

    st.subheader(title)
    fig = px.scatter_matrix(df, dimensions=num_cols, title=title)
    st.plotly_chart(fig, use_container_width=True)
