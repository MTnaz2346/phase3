import pandas as pd
import joblib

def load_model(path: str):
    return joblib.load(path)

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # These are the exact features the model expects (based on the error)
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
        'History of Substance Abuse',
        'Family History of Depression',
        'Chronic Medical Conditions'  # required because it was mistakenly included during training
    ]

    # Fill missing columns with 0 if not included in form
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]
    return df


