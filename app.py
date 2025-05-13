import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark theme CSS
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTextInput > div > div > input, .stNumberInput input, .stSelectbox div div div {
        background-color: #262730;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Function to prepare features
def prepare_features(df):
    cols_to_use = ['Contract', 'InternetService', 'TechSupport', 'OnlineSecurity', 'DeviceProtection', 'tenure', 'MonthlyCharges']
    temp_df = df[cols_to_use].copy()
    temp_df = pd.get_dummies(temp_df, columns=['Contract', 'InternetService', 'TechSupport', 'OnlineSecurity', 'DeviceProtection'])
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in temp_df.columns:
            temp_df[col] = 0
    return temp_df[expected_cols]

# Sidebar
st.sidebar.title("Customer Info")
tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=100, value=10)
monthlycharge = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encode single input
def encode_input():
    return {
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthlycharge
    }

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Insights", "📈 Confusion Matrix"])

# --------- TAB 1: PREDICTION ----------
with tab1:
    st.header("🔍 Churn Prediction")

    if st.button("Predict Churn"):
        user_input_dict = encode_input()
        input_df = pd.DataFrame([user_input_dict])
        for col in scaler.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[scaler.feature_names_in_]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")

# --------- TAB 2: INSIGHTS ----------
with tab2:
    st.header("📊 Customer Churn Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.pie(df, names='Churn', title="Churn Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='tenure', color='Churn', title="Tenure vs Churn")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.box(df, x='Churn', y='MonthlyCharges', title="Monthly Charges by Churn")
        st.plotly_chart(fig3, use_container_width=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        fig4 = px.histogram(df, x='InternetService', color='Churn', title="Internet Service & Churn")
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        contract_counts = df['Contract'].value_counts().reset_index()
        contract_counts.columns = ['Contract', 'Count']
        fig5 = px.bar(contract_counts, x='Contract', y='Count', title="Contract Type Distribution")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.histogram(df, x='TechSupport', color='Churn', title="Tech Support vs Churn")
        st.plotly_chart(fig6, use_container_width=True)

# --------- TAB 3: CONFUSION MATRIX ----------
with tab3:
    st.header("📈 Confusion Matrix on Full Dataset")
    y_true = (df['Churn'] == 'Yes').astype(int)
    X = prepare_features(df)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax_cm, cmap="cividis")  # Elegant colormap
    st.pyplot(fig_cm)
