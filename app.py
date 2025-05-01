
# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import (
    load_data, preprocess_data, train_income_model,
    create_histogram, create_marital_status_plot, 
    create_medical_conditions_plot, create_education_income_plot
)

# Set page configuration
st.set_page_config(page_title="Depression Data Analysis", layout="wide", page_icon="📊")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Income Predictor"])

# Load the data
data = load_data()
processed_data = preprocess_data(data)

if page == "Data Exploration":
    st.title('Depression Data Exploration')

    # Show dataset description
    st.write("Dataset overview:")
    st.write(data.describe())

    # Display histograms for Age, Number of Children, and Income
    st.header('Age Distribution')
    age_fig = create_histogram(data, 'Age')
    st.pyplot(age_fig)

    st.header('Number of Children Distribution')
    children_fig = create_histogram(data, 'Number of Children')
    st.pyplot(children_fig)

    st.header('Income Distribution')
    income_fig = create_histogram(data, 'Income')
    st.pyplot(income_fig)

    # Marital Status Breakdown
    st.header('Breakdown of Marital Status Among Depressed Individuals')
    marital_fig = create_marital_status_plot(data)
    st.pyplot(marital_fig)

    # Chronic Medical Conditions and Depression
    st.header('Breakdown of Chronic Medical Conditions and Depression')
    medical_fig = create_medical_conditions_plot(data)
    st.pyplot(medical_fig)

elif page == "Income Predictor":
    st.title('Income Predictor')
    
    # Display the relation between education and income
    st.header('Education Level vs Income')
    education_income_fig = create_education_income_plot(data)
    st.pyplot(education_income_fig)
    
    # Get the trained model and its metrics
    model, mse, r2 = train_income_model(processed_data)
    
    # Display model metrics
    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.4f}")
    
    # Create a predictor
    st.subheader('Predict Income Based on Education Level')
    
    # Create dropdown for education level
    education_options = {
        'High School': 0,
        'Associate Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        'PhD': 4
    }
    selected_education = st.selectbox('Select Education Level', list(education_options.keys()))
    
    # Convert selected education to numerical value
    education_value = education_options[selected_education]
    
    # Make prediction
    predicted_income = model.predict([[education_value]])[0]
    
    # Display prediction
    st.success(f"Predicted Income: ${predicted_income:.2f}")
    
    # Add explanation
    st.markdown("""
    ### How it works
    
    This predictor uses a simple linear regression model to estimate income based on education level.
    The model was trained on our dataset with the following mappings:
    
    - High School (0)
    - Associate Degree (1)
    - Bachelor's Degree (2)
    - Master's Degree (3)
    - PhD (4)
    
    The linear model finds the relationship between education level and income using the equation:
    
    Income = (Coefficient × Education Level) + Intercept
    
    Where:
    - Coefficient: {model.coef_[0]:.2f}
    - Intercept: {model.intercept_:.2f}
    """)