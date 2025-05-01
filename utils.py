# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_data():
    return pd.read_csv('depression_data.csv')

# Function to create histogram figure
def create_histogram(data, column_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(f'Histogram of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequency')
    return fig

# Function to create marital status figure
def create_marital_status_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    maritalStatusCounts = data['Marital Status'].value_counts()
    maritalStatusCounts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Marital Status')
    ax.set_ylabel('Frequency')
    ax.set_title('Breakdown of Marital Status Among Depressed Individuals')
    plt.xticks(rotation=0)
    return fig

# Function to create medical conditions figure
def create_medical_conditions_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    medCount = data['Chronic Medical Conditions'].value_counts()
    medCount.plot(kind='bar', ax=ax)
    ax.set_xlabel('Individual has Chronic Medical Conditions?')
    ax.set_ylabel('Frequency')
    ax.set_title('Breakdown of Depressed Individuals with Chronic Medical Conditions')
    plt.xticks(rotation=0)
    return fig




