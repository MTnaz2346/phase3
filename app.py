# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import preprocess_input, load_model

st.set_page_config(page_title="Depression Insights", layout="wide")

# Load model and data
model = load_model("model.pkl")
data = pd.read_csv("depression_data.csv")  # Placeholder, update with actual CSV

st.title("🧠 Depression Insights & Predictor")
st.markdown("This app explores factors related to depression and allows live prediction using machine learning.")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["EDA Dashboard", "Model Performance", "Live Predictor"])

# --- EDA Dashboard ---
if page == "EDA Dashboard":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        sns.histplot(data["Age"], kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Income Distribution")
        sns.histplot(data["Income"], kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.subheader("Depression by Marital Status")
        st.bar_chart(data.groupby("MaritalStatus")["Depression"].mean())

        st.subheader("Children Count Distribution")
        sns.countplot(x="Children", data=data)
        st.pyplot(plt.gcf())
        plt.clf()

# --- Model Performance ---
elif page == "Model Performance":
    st.header("Model Evaluation")
    st.markdown("Below is the performance summary of our trained model.")

    # Placeholder metrics — will be updated
    st.metric("Accuracy", "85%")
    st.metric("Precision", "80%")
    st.metric("Recall", "82%")

    st.image("confusion_matrix.png", caption="Confusion Matrix")  # Optional

# --- Live Predictor ---
elif page == "Live Predictor":
    st.header("Live Depression Risk Prediction")
    st.markdown("Enter patient details below:")

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
        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0][1]

        st.success(f"Prediction: {'At Risk of Depression' if prediction == 1 else 'Not at Risk'}")
        st.info(f"Probability of Depression: {proba:.2%}")
