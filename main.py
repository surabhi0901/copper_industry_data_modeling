# Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import dtale
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.impute import SimpleImputer
from dtale.views import startup
from dtale.app import get_instance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, boxcox
from scipy.special import boxcox1p

#Initializing the session state

if 'df' not in st.session_state:
    st.session_state.df = None

if 'df_imputed_numeric' not in st.session_state:
    st.session_state.df_imputed_numeric = None

if 'data' not in st.session_state:
    st.session_state.data = None

# Setting up the page

st.set_page_config(page_title= "Copper Industry Data Modeling| By Surabhi Yadav",
                   page_icon= ":🏭:", 
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This app is created by *Surabhi Yadav!*"""})

with st.sidebar:
    selected = option_menu('MENU', ["Upload & Read","Data Preprocessing", "EDA", "Feature Engineering", "Model Development & Evaluation"], 
                           icons=["upload", "funnel-fill", "bar-chart-line-fill", "gear-fill", "search"], 
                           menu_icon="menu-up",
                           default_index=0,
                           orientation="vertical",
                           styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "#B87333"},
                                   "icon": {"font-size": "15px"},
                                   "container" : {"max-width": "6000px"},
                                   "nav-link-selected": {"background-color": "#B87333"}})
    

# Uploading and reading the data file 

if selected == "Upload & Read":

    st.title("File Upload and Display")

    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    show = st.button("Show the file uploaded")
    if show:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Filename:", uploaded_file.name)
        st.dataframe(st.session_state.df)

# Data Preprocessing

if selected == "Data Preprocessing":

    #st.session_state.df.dropna()
    
    st.subheader("Before dealing with the null values")
    st.write("")
    st.write("")
    st.write(st.session_state.df.isnull().sum())

    #Removing null values or filling them out
    st.session_state.df['material_ref'].replace('00000', np.nan, inplace=True)
    numeric_columns = st.session_state.df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='median')
    st.session_state.df_imputed_numeric = imputer.fit_transform(st.session_state.df[numeric_columns])
    st.session_state.df_imputed_numeric = pd.DataFrame(st.session_state.df_imputed_numeric, columns=numeric_columns)
    st.session_state.data = pd.concat([st.session_state.df_imputed_numeric, st.session_state.df.select_dtypes(exclude=['number'])], axis=1)

    categorical_columns = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('status')

    #Treating outliers using Isolation Forest
    numerical_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(st.session_state.data[numerical_columns])
    mask = yhat != -1
    st.session_state.data = st.session_state.data[mask]

    # Assuming st.session_state.data is already defined and numerical_columns, categorical_columns are identified
    numerical_columns = st.session_state.data.select_dtypes(include=['number']).columns.tolist()

    # Checking skewness and applying transformations
    skewed_features = st.session_state.data[numerical_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_features[abs(skewed_features) > 0.75]
    skewed_columns = high_skew.index

    # Debug print for skewed features
    print("Skewed features:\n", high_skew)s

    # Apply transformation to numerical columns if necessary (for example, log or Box-Cox transformation)
    for col in skewed_columns:
        if all(st.session_state.data[col] > 0):  # Apply Box-Cox if all values are positive
            st.session_state.data[col] = boxcox1p(st.session_state.data[col], 0.15)
        else:  # Otherwise, apply log transformation safely
            st.session_state.data[col] = np.log1p(st.session_state.data[col].clip(lower=0))

    # Handling categorical columns
    categorical_columns = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('status')  # Assuming 'status' is excluded

    for col in categorical_columns:
        top_categories = st.session_state.data[col].value_counts().nlargest(10).index
        print(f"Top categories for {col}:\n", top_categories)  # Debug print for top categories
        st.session_state.data[col] = np.where(st.session_state.data[col].isin(top_categories), st.session_state.data[col], 'Other')

        # Debug print for transformed column
        print(f"Transformed {col}:\n", st.session_state.data[col].value_counts())
    # #Checking skewness and apply transformations
    # skewed_features = st.session_state.data[numerical_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # high_skew = skewed_features[abs(skewed_features) > 0.75]
    # skewed_columns = high_skew.index

    # for col in categorical_columns:
    #     top_categories = st.session_state.data[col].value_counts().nlargest(10).index
    #     st.session_state.data[col] = np.where(st.session_state.data[col].isin(top_categories), st.session_state.data[col], 'Other')

    #Encoding categorical variables
    st.session_state.data = pd.get_dummies(st.session_state.data, columns=categorical_columns)

    #Scale numerical features
    scaler = StandardScaler()
    st.session_state.data[numerical_columns] = scaler.fit_transform(st.session_state.data[numerical_columns])

    if 'status' in st.session_state.data.columns:
        st.session_state.data = st.session_state.data.dropna(subset=['status'])
    
    st.subheader("After dealing with the null values")
    st.write("")
    st.write("")
    st.write(st.session_state.data.isnull().sum())
    st.dataframe(st.session_state.data)

    st.session_state.data.to_csv(r'C:\Users\sy090\Downloads\PROJECTS\copper_industry_data_modeling\preprocessed_data.csv', index=False)

# #EDA : Histogram and Boxplot: For exploring skewness and outliers respectively

# #Plotting the histogram of numerical columns
# raw_numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
# columns_to_drop = ['item_date', 'customer', 'country', 'product_ref', 'delivery date']
# numerical_columns = [column for column in raw_numerical_columns if column not in columns_to_drop]

# nc_df = data.drop(columns=['item_date', 'customer', 'country', 'product_ref', 'delivery date'])
# num_plots = len(numerical_columns)
# num_cols = min(num_plots, 3)  
# num_rows = (num_plots - 1) // num_cols + 1 if num_plots > 1 else 1 

# plt.figure() 
# fig1, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# if num_plots > 1:
#     axes = axes.flatten()
# else:
#     axes = [axes]

# for i, col in enumerate(numerical_columns):
#     sns.histplot(nc_df[col], ax=axes[i], kde=True, bins=20)  # Adjust bins as needed
#     axes[i].set_title(col)

# for i in range(num_plots, num_rows * num_cols):
#     fig1.delaxes(axes[i])

# plt.tight_layout()

# plt.show()

# #Plotting the boxplots of numerical columns
# raw_numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
# columns_to_drop = ['item_date', 'customer', 'country', 'product_ref', 'delivery date']
# numerical_columns = [column for column in raw_numerical_columns if column not in columns_to_drop]

# nc_df = data.drop(columns=['item_date', 'customer', 'country', 'product_ref', 'delivery date'])
# num_plots = len(numerical_columns)
# num_cols = min(num_plots, 3)  
# num_rows = (num_plots - 1) // num_cols + 1 if num_plots > 1 else 1 

# plt.figure()
# fig2, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# if num_plots > 1:
#     axes = axes.flatten()
# else:
#     axes = [axes]
    
# for i, col in enumerate(numerical_columns):
#             sns.boxplot(data=nc_df[col], ax=axes[i])
#             axes[i].set_title(col)

# for i in range(num_plots, num_rows * num_cols):
#     fig2.delaxes(axes[i])

# plt.tight_layout()

# plt.show()