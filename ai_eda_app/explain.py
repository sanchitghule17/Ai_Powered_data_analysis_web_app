import shap, pandas as pd, streamlit as st

def shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("🔍 SHAP Feature Importance")
    st.pyplot(shap.summary_plot(shap_values, X, show=False))
