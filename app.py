# app.py
import streamlit as st
import pandas as pd
import joblib
# Make sure utils.py contains preprocess_input and load_model
from utils import preprocess_input, load_model

st.set_page_config(page_title="Depression Predictor", layout="centered")

# Load model (Make sure the file name matches exactly)
# It's good practice to also load any scalers if they were used
try:
    model = load_model("model.pkl")
    # Example: If a scaler was saved during training:
    # scaler = load_model("scaler.pkl") # Load scaler if you used one
except FileNotFoundError:
    st.error("Error: model.pkl (or scaler.pkl) not found. Make sure the files are in the correct directory.")
    st.stop() # Stop the app if model can't be loaded
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.title("🧠 Depression Risk Predictor")
st.markdown("Enter the individual's information to predict depression risk.")
page = st.sidebar.radio("Navigate", ["Live Predictor", "Model Metrics"])
if page == "Live Predictor":
    st.header("Live Prediction")

    # Input form with original string values, mapped later
    age = st.slider("Age", 10, 100, 30)

    education = st.selectbox("Education Level", [
        'High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD'
    ])
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    smoking = st.selectbox("Smoking Status", ['Non-smoker', 'Former', 'Current'])
    physical_activity = st.selectbox("Physical Activity Level", ['Sedentary', 'Moderate', 'Active'])

    employment = st.selectbox("Employment Status", ['Unemployed', 'Employed'])
    income = st.number_input("Income", min_value=0, value=50000)

    alcohol = st.selectbox("Alcohol Consumption", ['Low', 'Moderate', 'High'])
    diet = st.selectbox("Dietary Habits", ['Unhealthy', 'Moderate', 'Healthy'])
    sleep = st.selectbox("Sleep Patterns", ['Poor', 'Fair', 'Good'])

    mental_illness = st.selectbox("History of Mental Illness", [0, 1])
    substance_abuse = st.selectbox("History of Substance Abuse", [0, 1])
    family_history = st.selectbox("Family History of Depression", [0, 1])

    if st.button("Predict"):
        # Mapping string labels to encoded values
        input_df = pd.DataFrame([{
            "Age": age,
            "Education Level": {
                'High School': 0,
                'Associate Degree': 1,
                "Bachelor's Degree": 2,
                "Master's Degree": 3,
                'PhD': 4
            }[education],
            "Number of Children": children,
            "Smoking Status": {
                'Non-smoker': 2,
                'Former': 1,
                'Current': 0
            }[smoking],
            "Physical Activity Level": {
                'Sedentary': 0,
                'Moderate': 1,
                'Active': 2
            }[physical_activity],
            "Employment Status": {
                'Unemployed': 0,
                'Employed': 1
            }[employment],
            "Income": income,
            "Alcohol Consumption": {
                'Low': 0,
                'Moderate': 1,
                'High': 2
            }[alcohol],
            "Dietary Habits": {
                'Unhealthy': 0,
                'Moderate': 1,
                'Healthy': 2
            }[diet],
            "Sleep Patterns": {
                'Poor': 0,
                'Fair': 1,
                'Good': 2
            }[sleep],
            "History of Mental Illness": mental_illness,
            "History of Substance Abuse": substance_abuse,
            "Family History of Depression": family_history
        }])

        try:
            # If your model expects scaled input, make sure preprocess_input() does scaling
            processed = preprocess_input(input_df)
            prediction = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0][1]

            st.success(f"Prediction: {'At Risk of Chronic Medical Conditions' if prediction == 1 else 'Not at Risk'}")
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

