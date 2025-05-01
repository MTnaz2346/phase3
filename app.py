

# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, create_histogram, create_marital_status_plot, create_medical_conditions_plot

# Title of the Streamlit app
st.title('Depression Data Exploration')

# Load the data
data = load_data()

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
