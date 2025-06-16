
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import streamlit as st

st.set_page_config(page_title='AI-Powered Data Analytics', page_icon='📊', layout='wide')
st.title('📊 AI-Powered Data Preprocessing & ML')
st.subheader('Upload your dataset and apply preprocessing & AutoML')


file = st.file_uploader('Upload CSV or Excel file', type=['csv', 'xlsx'])
if file:
    try:
        data = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
  
  
    st.subheader(':rainbow[Basic information of the dataset]',divider='rainbow')
    tab1,tab2,tab3,tab4 = st.tabs(['Summary','Top and Bottom Rows','Data Types','Columns'])

    with tab1:
        st.write(f'There are {data.shape[0]} rows in dataset and  {data.shape[1]} columns in the dataset')
        st.subheader(':gray[Statistical summary of the dataset]')
        st.dataframe(data.describe())
    with tab2:
        st.subheader(':gray[Top Rows]')
        toprows = st.slider('Number of rows you want',1,data.shape[0],key='topslider')
        st.dataframe(data.head(toprows))
        st.subheader(':gray[Bottom Rows]')
        bottomrows = st.slider('Number of rows you want',1,data.shape[0],key='bottomslider')
        st.dataframe(data.tail(bottomrows))
    with tab3:
        st.subheader(':grey[Data types of column]')
        st.dataframe(data.dtypes)
    with tab4:
        st.subheader('Column Names in Dataset')
        st.write(list(data.columns))
    
    st.subheader(':rainbow[Column Values To Count]',divider='rainbow')
    with st.expander('Value Count'):
        col1,col2 = st.columns(2)
        with col1:
          column  = st.selectbox('Choose Column name',options=list(data.columns))
        with col2:
            toprows = st.number_input('Top rows',min_value=1,step=1)
        
        count = st.button('Count')
        if(count==True):
            result = data[column].value_counts().reset_index().head(toprows)
            st.dataframe(result)
            st.subheader('Visualization',divider='gray')
            fig = px.bar(data_frame=result,x=column,y='count',text='count',template='plotly_white')
            st.plotly_chart(fig)
            fig = px.line(data_frame=result,x=column,y='count',text='count',template='plotly_white')
            st.plotly_chart(fig)
            fig = px.pie(data_frame=result,names=column,values='count')
            st.plotly_chart(fig) 
   
 
    st.subheader(':rainbow[Data Visualization]', divider='rainbow')
    graphs = st.selectbox('Choose your graphs', options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
    if graphs == 'line':
        x_axis = st.selectbox('Choose X axis', options=list(data.columns))
        y_axis = st.selectbox('Choose Y axis', options=list(data.columns))
        color = st.selectbox('Color Information', options=[None] + list(data.columns))
        fig = px.line(data_frame=data, x=x_axis, y=y_axis, color=color, markers='o')
        st.plotly_chart(fig)
    elif graphs == 'bar':
        x_axis = st.selectbox('Choose X axis', options=list(data.columns))
        y_axis = st.selectbox('Choose Y axis', options=list(data.columns))
        color = st.selectbox('Color Information', options=[None] + list(data.columns))
        facet_col = st.selectbox('Column Information', options=[None] + list(data.columns))
        fig = px.bar(data_frame=data, x=x_axis, y=y_axis, color=color, facet_col=facet_col, barmode='group')
        st.plotly_chart(fig)
    elif graphs == 'scatter':
        x_axis = st.selectbox('Choose X axis', options=list(data.columns))
        y_axis = st.selectbox('Choose Y axis', options=list(data.columns))
        color = st.selectbox('Color Information', options=[None] + list(data.columns))
        size = st.selectbox('Size Column', options=[None] + list(data.columns))
        fig = px.scatter(data_frame=data, x=x_axis, y=y_axis, color=color, size=size)
        st.plotly_chart(fig)
    elif graphs == 'pie':
        values = st.selectbox('Choose Numerical Values', options=list(data.columns))
        names = st.selectbox('Choose labels', options=list(data.columns))
        fig = px.pie(data_frame=data, values=values, names=names)
        st.plotly_chart(fig)
    elif graphs == 'sunburst':
        path = st.multiselect('Choose your Path', options=list(data.columns))
        fig = px.sunburst(data_frame=data, path=path, values=data.columns[0])
        st.plotly_chart(fig)
    
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    data.drop(columns=datetime_cols, inplace=True, errors='ignore')
    
    
    st.write("### Original Data Preview")
    st.dataframe(data.head())
    
    
    st.subheader('📊 Pairplot Visualization (Before Preprocessing)')
    fig = px.scatter_matrix(data, dimensions=data.select_dtypes(include=[np.number]).columns)
    st.plotly_chart(fig)

    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        'Summary', 'Top and Bottom Rows', 'Data Types', 'Columns', 'Missing Values', 'Processed Data', 'Model Predictions', 'Product Prediction'
    ])

    with tab1:
        st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset')
        st.dataframe(data.describe())

    with tab2:
        st.dataframe(data.head(10))
        st.dataframe(data.tail(10))

    with tab3:
        st.subheader('📌 Data Types')
        st.dataframe(data.dtypes)

    with tab4:
        st.subheader('📌 Column Names')
        st.write(list(data.columns))

    with tab5:
        st.subheader('📌 Missing Values by Column')
        missing_values = data.isnull().sum()
        st.dataframe(missing_values[missing_values > 0])

    
    st.subheader('🛠 Data Cleaning & Preprocessing')
    
    
    data.fillna(data.mean(numeric_only=True), inplace=True)
    
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    
    z_scores = (data - data.mean(numeric_only=True)) / data.std(numeric_only=True)
    data = data[(z_scores < 3).all(axis=1)]
    
    
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    
    data.drop_duplicates(inplace=True)

    
    st.subheader('📊 Pairplot Visualization (After Preprocessing)')
    fig = px.scatter_matrix(data, dimensions=data.select_dtypes(include=[np.number]).columns)
    st.plotly_chart(fig)

   
    try:
        X = data.iloc[:, :-1] 
        y = data.iloc[:, -1]  
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        k_best = min(5, X.shape[1]) 
        best_features = SelectKBest(score_func=f_classif, k=k_best)
        X_new = best_features.fit_transform(X, y)
        
        selected_feature_names = X.columns[best_features.get_support()].tolist()
        processed_data = pd.DataFrame(X_new, columns=selected_feature_names)
        processed_data['Target'] = y
        
        with tab6:
            st.write("### Processed Data Preview")
            st.dataframe(processed_data.head())
    except Exception as e:
        st.error(f"Feature selection failed: {e}")
        st.stop()
    
    # Model Training and Comparison
    with st.expander('🔍 Train and Compare Machine Learning Models'):
        st.write("### Model Training & Comparison")

        if np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer):
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }
        else:
            models = {
                "Random Forest Classifier": RandomForestClassifier()
            }
        
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
        
        results = {}
        best_model_instance = None
        best_score = float('-inf')
        predictions = None
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            if isinstance(model, (LinearRegression, RandomForestRegressor)):
                metric = -mean_squared_error(y_test, predictions)  
            else:
                metric = accuracy_score(y_test, predictions)
            
            results[model_name] = metric
            
            if metric > best_score:
                best_score = metric
                best_model_instance = model_name
        
        st.write("### Model Comparison Results")
        st.write(results)
        st.success(f"🎯 Best Performing Model: *{best_model_instance}*")
        
       
        st.subheader("📊 Prediction Results Visualization")
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
        st.plotly_chart(fig)