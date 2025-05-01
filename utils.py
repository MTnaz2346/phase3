
import pandas as pd
import joblib

def load_model(path: str):
    """Load the trained model from a pickle file."""
    return joblib.load(path)

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing as in training to incoming data."""

    df = df.copy()

    # No need for one-hot encoding or MaritalStatus logic
    # Ensure columns match the order and names used in training

    expected_cols = [
        'Age',
        'Education Level',
        'Number of Children',
        'Smoking Status',
        'Physical Activity Level',
        'Employment Status',
        'Income',
        'Alcohol Consumption',
        'Dietary Habits',
        'Sleep Patterns',
        'History of Mental Illness',
        'History of Substance Abuse',
        'Family History of Depression'
    ]

    # Fill missing columns with zeros if needed
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training
    df = df[expected_cols]

    return df
