import pathlib

import joblib
import pandas as pd

from src.config import DEFAULT_MODEL, TARGET_COL, TEST_PROCESSED_PATH  #MODEL_PATH,MODEL_VERSION_DIR,
from src.logger import logger
from src.model_registry import get_latest_model_path


def predict_default_probability(data_path: pathlib.Path = TEST_PROCESSED_PATH, model_name: pathlib.Path = DEFAULT_MODEL) -> pd.DataFrame:
    """
    Load trained classifier model and predict probability of default on input features.
    
    Parameters
    ----------
    data_path : pathlib.Path
        Input processed test dataset.

    model_name : str
        Model to use for prediction.
        Example:
        logistic_regression
        random_forest
        xgboost
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions attached.
    """
    
    model_path = get_latest_model_path(model_name)
    
    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run 'python -m src.train' first."
        )
    logger.info(
        f"Loading model for prediction: {model_name}"
    )
     
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run 'make train' first.")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    df_results = df.copy()
    df_results['Predicted_Default_Prob'] = probabilities
    df_results['Predicted_Status'] = predictions

    return df_results

if __name__ == '__main__':
    print("="*60)
    print("BATCH PREDICTION INFERENCE")
    print("="*60)
    results = predict_default_probability()
    print(f"Model Used: {DEFAULT_MODEL}")
    print(f"Evaluated {len(results):,} sample applications.")
    print("Sample Prediction Output:")
    print(results[['Predicted_Default_Prob', 'Predicted_Status']].head(10))
    print("="*60)
