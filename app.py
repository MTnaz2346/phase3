import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, plot_histogram, plot_marital_status, plot_medical_conditions

# Title of the Streamlit app
st.title('Depression Data Exploration')

# Load the data
data = load_data()

# Show dataset description
st.write("Dataset overview:")
st.write(data.describe())

# Display histograms for Age, Number of Children, and Income
st.header('Age Distribution')
plot_histogram(data, 'Age')

st.header('Number of Children Distribution')
plot_histogram(data, 'Number of Children')

st.header('Income Distribution')
plot_histogram(data, 'Income')

# Marital Status Breakdown
st.header('Breakdown of Marital Status Among Depressed Individuals')
plot_marital_status(data)

# Chronic Medical Conditions and Depression
st.header('Breakdown of Chronic Medical Conditions and Depression')
plot_medical_conditions(data)
