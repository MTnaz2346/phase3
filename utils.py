# utils.py
import pandas as pd
import joblib

def load_model(path: str):
    """Load the trained model from a pickle file."""
    return joblib.load(path)

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing as in training to incoming data."""

    # Example preprocessing — this should match what was done in the notebook
    df = df.copy()

    # Encode MaritalStatus as dummy variables (one-hot)
    df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=True)

    # Ensure all expected columns are present (some may be missing depending on selected value)
    expected_cols = ['Age', 'Income', 'Children',
                     'MaritalStatus_Married',
                     'MaritalStatus_Single',
                     'MaritalStatus_Widowed']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_cols]

    return df
