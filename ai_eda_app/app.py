import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import streamlit as st
from modeling import train_and_evaluate
from eda import show_overview, value_counts, pairplot
from preprocessing import clean_data


# ─────────────────────────────
#  Page & app-level settings
# ─────────────────────────────
st.set_page_config(
    page_title='AI-Powered Data Analytics',
    page_icon='📊',
    layout='wide'
)
st.title('📊 AI-Powered Data Preprocessing & ML')
st.subheader('Upload your dataset and apply preprocessing & AutoML')


# ─────────────────────────────
#  File upload
# ─────────────────────────────
file = st.file_uploader('Upload CSV or Excel file', type=['csv', 'xlsx'])

if file:  # everything else lives inside this block
    try:
        data = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
    except Exception as e:
        st.error(f'Error loading file: {e}')
        st.stop()

    # immutable snapshot for “before” visuals
    raw_data = data.copy()
    dt_cols = raw_data.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
    raw_data.drop(columns=dt_cols, inplace=True, errors='ignore')

    # ─────────────────────────────
    #  Basic dataset information
    # ─────────────────────────────
    st.subheader(':rainbow[Basic information of the dataset]', divider='rainbow')
    t1, t2, t3, t4 = st.tabs(['Summary', 'Top & Bottom', 'Data Types', 'Columns'])

    with t1:
        show_overview(raw_data)

    with t2:
        st.subheader(':gray[Top Rows]')
        top_n = st.slider('Rows to show', 1, raw_data.shape[0], 5, key='topslider')
        st.dataframe(raw_data.head(top_n))

        st.subheader(':gray[Bottom Rows]')
        bottom_n = st.slider('Rows to show', 1, raw_data.shape[0], 5, key='bottomslider')
        st.dataframe(raw_data.tail(bottom_n))

    with t3:
        st.subheader(':grey[Data types]')
        st.dataframe(raw_data.dtypes)

    with t4:
        st.subheader('Column Names')
        st.write(list(raw_data.columns))

    # ─────────────────────────────
    #  Single-column value counts
    # ─────────────────────────────
    st.subheader(':rainbow[Column Values to Count]', divider='rainbow')
    with st.expander('Value Count'):
        c1, c2 = st.columns(2)
        with c1:
            column = st.selectbox('Column', options=list(raw_data.columns))
        with c2:
            top_k = st.number_input('Top rows', min_value=1, value=10, step=1)

        if st.button('Count'):
            value_counts(raw_data, column, int(top_k))

    # ─────────────────────────────
    #  Ad-hoc visualisations
    # ─────────────────────────────
    st.subheader(':rainbow[Data Visualisation]', divider='rainbow')
    gtype = st.selectbox('Graph type', ['line', 'bar', 'scatter', 'pie', 'sunburst'])

    if gtype == 'line':
        x = st.selectbox('X-axis', raw_data.columns)
        y = st.selectbox('Y-axis', raw_data.columns)
        colour = st.selectbox('Colour', [None] + list(raw_data.columns))
        st.plotly_chart(px.line(raw_data, x=x, y=y, color=colour, markers=True))

    elif gtype == 'bar':
        x = st.selectbox('X-axis', raw_data.columns)
        y = st.selectbox('Y-axis', raw_data.columns)
        colour = st.selectbox('Colour', [None] + list(raw_data.columns))
        facet = st.selectbox('Facet column', [None] + list(raw_data.columns))
        st.plotly_chart(px.bar(raw_data, x=x, y=y, color=colour,
                               facet_col=facet, barmode='group'))

    elif gtype == 'scatter':
        x = st.selectbox('X-axis', raw_data.columns)
        y = st.selectbox('Y-axis', raw_data.columns)
        colour = st.selectbox('Colour', [None] + list(raw_data.columns))
        size = st.selectbox('Size', [None] + list(raw_data.columns))
        st.plotly_chart(px.scatter(raw_data, x=x, y=y, color=colour, size=size))

    elif gtype == 'pie':
        values = st.selectbox('Values', raw_data.columns)
        names = st.selectbox('Labels', raw_data.columns)
        st.plotly_chart(px.pie(raw_data, values=values, names=names))

    elif gtype == 'sunburst':
        path = st.multiselect('Path', raw_data.columns)
        if path:
            st.plotly_chart(px.sunburst(raw_data, path=path, values=raw_data.columns[0]))

    # ─────────────────────────────
    #  Pairplots (before & after)
    # ─────────────────────────────
    pairplot(raw_data, "📊 Pairplot – Before Preprocessing")

    st.subheader('🛠 Data Cleaning & Preprocessing')
    cleaned_data = clean_data(raw_data)

    pairplot(cleaned_data, "📊 Pairplot – After Preprocessing")
    data = cleaned_data  # hand off to modelling pipeline

    # ─────────────────────────────
    #  Extra EDA tabs
    # ─────────────────────────────
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        'Summary', 'Top & Bottom', 'Data Types', 'Columns',
        'Missing Values', 'Processed Data', 'Model Predictions', 'Product Prediction'
    ])

    with t1:
        st.write(f'Cleaned dataset → {data.shape[0]} rows × {data.shape[1]} columns')
        st.dataframe(data.describe())

    with t2:
        st.dataframe(data.head(10))
        st.dataframe(data.tail(10))

    with t3:
        st.dataframe(data.dtypes)

    with t4:
        st.write(list(data.columns))

    with t5:
        mv = data.isnull().sum()
        st.dataframe(mv[mv > 0])

    # ─────────────────────────────
    #  Feature selection
    # ─────────────────────────────
    try:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if y.dtype == 'object':  # encode categorical targets
            y = LabelEncoder().fit_transform(y)

        k_best = min(5, X.shape[1])
        selector = SelectKBest(f_classif, k=k_best)
        X_new = selector.fit_transform(X, y)

        sel_cols = X.columns[selector.get_support()].tolist()
        processed_data = pd.DataFrame(X_new, columns=sel_cols)
        processed_data['Target'] = y

        with t6:
            st.write('### Processed Data')
            st.dataframe(processed_data.head())

    except Exception as e:
        st.error(f'Feature selection failed: {e}')
        st.stop()

    # ─────────────────────────────
    #  Model training & comparison
    # ─────────────────────────────
    with st.expander('🔍 Train & Compare Models'):

        task = (
            "regression"
            if np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer)
            else "classification"
        )

        results, best_name, y_test, preds = train_and_evaluate(X_new, y, task)

        st.write(results)
        st.success(f'🎯 Best Model: **{best_name}**')

        st.subheader('📊 Actual vs Predicted')
        st.plotly_chart(
            px.scatter(
                x=y_test,
                y=preds,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Actual vs Predicted'
            )
        )
