import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to load the dataset
def load_data():
    return pd.read_csv('depression_data.csv')

# Function to preprocess data
def preprocess_data(data):
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Education level mapping
    eduMap = {
        'High School': 0,
        'Associate Degree': 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        'PhD': 4
    }
    processed_data["Education Level"] = processed_data["Education Level"].map(eduMap)
    
    # Physical activity level mapping
    actMap = {
        'Sedentary': 0,
        'Moderate': 1,
        'Active': 2
    }
    processed_data["Physical Activity Level"] = processed_data["Physical Activity Level"].map(actMap)
    
    # Sleep patterns mapping
    sleepMap = {
        'Fair': 1,
        'Good': 2,
        'Poor': 0
    }
    processed_data['Sleep Patterns'] = processed_data['Sleep Patterns'].map(sleepMap)
    
    # Alcohol consumption mapping
    alcMap = {
        'Moderate': 1,
        'High': 2,
        'Low': 0
    }
    processed_data['Alcohol Consumption'] = processed_data['Alcohol Consumption'].map(alcMap)
    
    # Dietary habits mapping
    dietMap = {
        'Moderate': 1,
        'Unhealthy': 0,
        'Healthy': 2
    }
    processed_data['Dietary Habits'] = processed_data['Dietary Habits'].map(dietMap)
    
    # Smoking status mapping
    smokeMap = {
        'Non-smoker': 2,
        'Former': 1,
        'Current': 0
    }
    processed_data['Smoking Status'] = processed_data['Smoking Status'].map(smokeMap)
    
    # Yes/No mappings
    yesMap = {
        'Yes': 1,
        'No': 0
    }
    processed_data['History of Mental Illness'] = processed_data['History of Mental Illness'].map(yesMap)
    processed_data['History of Substance Abuse'] = processed_data['History of Substance Abuse'].map(yesMap)
    processed_data['Family History of Depression'] = processed_data['Family History of Depression'].map(yesMap)
    processed_data['Chronic Medical Conditions'] = processed_data['Chronic Medical Conditions'].map(yesMap)
    
    # Employment status mapping
    empMap = {
        'Unemployed': 0,
        'Employed': 1
    }
    processed_data['Employment Status'] = processed_data['Employment Status'].map(empMap)
    
    return processed_data

# Function to train the linear regression model
def train_income_model(data):
    X = data[["Education Level"]]
    y = data["Income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=83)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate model metrics
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = model.score(X_test, y_test)
    
    return model, mse, r2

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

# Function to create education vs income scatter plot
def create_education_income_plot(data):
    processed_data = preprocess_data(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Education Level', y='Income', data=processed_data, ax=ax)
    
    # Add regression line
    model, _, _ = train_income_model(processed_data)
    education_levels = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
    predicted_income = model.predict(education_levels)
    
    ax.plot(education_levels, predicted_income, color='red', linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Income')
    ax.set_title('Income vs Education Level with Linear Regression')
    
    # Set x-axis ticks to original education levels
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['High School', 'Associate', 'Bachelor\'s', 'Master\'s', 'PhD'])
    
    return fig


