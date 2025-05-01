import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_data():
    return pd.read_csv('depression_data.csv')

# Function to plot histograms
def plot_histogram(data, column_name):
    plt.figure(figsize=(8, 6))
    plt.hist(data[column_name], bins=10, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

# Function to plot marital status
def plot_marital_status(data):
    maritalStatusCounts = data['Marital Status'].value_counts()
    maritalStatusCounts.plot(kind='bar', figsize=(8, 6))
    plt.xlabel('Marital Status')
    plt.ylabel('Frequency')
    plt.title('Breakdown of Marital Status Among Depressed Individuals')
    plt.xticks(rotation=0)
    plt.show()

# Function to plot chronic medical conditions
def plot_medical_conditions(data):
    medCount = data['Chronic Medical Conditions'].value_counts()
    medCount.plot(kind='bar', figsize=(8, 6))
    plt.xlabel('Individual has Chronic Medical Conditions?')
    plt.ylabel('Frequency')
    plt.title('Breakdown of Depressed Individuals with Chronic Medical Conditions')
    plt.xticks(rotation=0)
    plt.show()



