import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    layout="wide"
)

# Optional subtle styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #36454F;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.markdown("## 📊 Churn Analytics App")
st.sidebar.markdown("End-to-End ML Deployment")

page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🔮 Predict Customer", "📈 Model Insights", "📘 About"]
)

# ---------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


# ===================================================
# DASHBOARD PAGE
# ===================================================
if page == "📊 Dashboard":

    st.title("📊 Customer Churn Dashboard")

    df = pd.read_csv("telco_cleaned.csv")

    churn_rate = df["Churn"].value_counts(normalize=True)["Yes"] * 100

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(df))
    col2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    # Contract Chart
    st.subheader("Churn Rate by Contract Type")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .value_counts(normalize=True)
        .unstack() * 100
    ).reset_index()

    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="Yes",
        color="Contract",
        title="Churn Percentage by Contract Type",
        labels={"Yes": "Churn Rate (%)"}
    )

    st.plotly_chart(fig_contract, use_container_width=True)

    # Tenure Chart
    st.subheader("Churn Rate by Tenure Group")

    df["TenureGroup"] = np.where(df["tenure"] <= 12, "0-12 Months", "12+ Months")

    tenure_churn = (
        df.groupby("TenureGroup")["Churn"]
        .value_counts(normalize=True)
        .unstack() * 100
    ).reset_index()

    fig_tenure = px.bar(
        tenure_churn,
        x="TenureGroup",
        y="Yes",
        color="TenureGroup",
        title="Churn Rate by Tenure Segment",
        labels={"Yes": "Churn Rate (%)"}
    )

    st.plotly_chart(fig_tenure, use_container_width=True)


# ===================================================
# PREDICT CUSTOMER PAGE
# ===================================================
elif page == "🔮 Predict Customer":

    st.title("🔮 Customer Risk Prediction")
    st.markdown("Adjust customer parameters below to evaluate churn risk.")

    model, scaler, feature_columns = load_model()

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        senior = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    # Create input dictionary
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": tenure * monthly_charges,
        "SeniorCitizen": senior
    }

    input_df = pd.DataFrame([input_data])

    # Add missing encoded columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Risk Assessment")

    st.metric("Churn Probability", f"{probability:.2%}")
    st.progress(min(probability, 1.0))

    if probability > 0.7:
        st.error("🚨 Very High Risk — Immediate Retention Action Recommended")
    elif probability > 0.5:
        st.warning("⚠️ Moderate Risk — Consider Engagement Strategy")
    else:
        st.success("✅ Low Risk — Customer Likely to Stay")


# ===================================================
# MODEL INSIGHTS PAGE
# ===================================================
elif page == "📈 Model Insights":

    st.title("📈 Model Performance & Explainability")

    st.metric("ROC-AUC Score", "0.83")
    st.metric("Recall (Churn Class)", "0.79")
    st.metric("Precision (Churn Class)", "0.50")

    st.markdown("---")

    st.subheader("Top Drivers Increasing Churn Risk")

    if os.path.exists("feature_importance.csv"):

        importance_df = pd.read_csv("feature_importance.csv")

        top_features = importance_df.sort_values(
            by="Coefficient",
            ascending=False
        ).head(10)

        fig_importance = px.bar(
            top_features,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Top Positive Drivers of Churn"
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    else:
        st.info("Feature importance file not found.")

    st.markdown("---")

    st.subheader("Why Logistic Regression Was Selected")

    st.write("""
    - Higher recall compared to Random Forest.
    - Better ROC-AUC performance.
    - Prioritizes minimizing false negatives.
    - Provides interpretable coefficients.
    """)


# ===================================================
# ABOUT PAGE
# ===================================================
elif page == "📘 About":

    st.title("📘 About This Project")

    st.write("""
    This project analyzes telecom customer churn using SQL and Machine Learning.

    Key Insights:
    • Month-to-month customers show highest churn  
    • Early tenure customers are high-risk  
    • High monthly charges correlate with churn  
    • Identified high-risk segment with ~69% churn  

    ML Model:
    • Logistic Regression (class-balanced)  
    • ROC-AUC: 0.83  
    • Recall: 0.79  

    End-to-End Workflow:
    SQL Analysis → Feature Engineering → Model Training → Streamlit Deployment
    """)
