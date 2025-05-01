# app.py
import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_input, load_model  # make sure utils.py contains these

st.set_page_config(page_title="Depression Predictor", layout="centered")

# Load model
model = load_model("model.pk1")  # Make sure the file name matches exactly

st.title("🧠 Depression Risk Predictor")
st.markdown("Enter the individual's information to predict depression risk.")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Live Predictor", "Model Metrics"])

# --- Live Predictor ---
if page == "Live Predictor":
    st.header("Live Prediction")

    # Input form
    age = st.slider("Age", 10, 100, 30)
    income = st.number_input("Income", min_value=0, value=50000)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "Children": children,
            "MaritalStatus": marital_status
        }])

        try:
            processed = preprocess_input(input_df)
            prediction = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0][1]

            st.success(f"Prediction: {'At Risk of Depression' if prediction == 1 else 'Not at Risk'}")
            st.info(f"Probability: {proba:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Model Metrics ---
elif page == "Model Metrics":
    st.header("Model Performance Metrics")
    st.metric("Accuracy", "85%")
    st.metric("Precision", "80%")
    st.metric("Recall", "82%")

    # Optional: Add image or confusion matrix
    st.image("confusion_matrix.png", caption="Confusion Matrix (optional)")

