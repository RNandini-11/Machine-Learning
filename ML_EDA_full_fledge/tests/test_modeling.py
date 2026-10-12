
import pandas as pd
import pytest

from src.config import DEFAULT_MODEL, TEST_PROCESSED_PATH  #MODEL_PATH
from src.model_registry import get_latest_model_path
from src.modeling.predict import predict_default_probability


def test_prediction_output():
    """Verify that batch prediction outputs valid probability bounds [0, 1]."""
    
    model_path = get_latest_model_path(DEFAULT_MODEL)
    
    if not TEST_PROCESSED_PATH.exists() or not model_path.exists():
        pytest.skip("Processed test data or model artifact missing")

    results = predict_default_probability(data_path=TEST_PROCESSED_PATH, model_name=DEFAULT_MODEL)

    assert isinstance(results, pd.DataFrame)
    assert 'Predicted_Default_Prob' in results.columns
    assert 'Predicted_Status' in results.columns

    # Probabilities must be strictly bounded between 0 and 1
    probs = results['Predicted_Default_Prob']
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Predicted status must be binary {0, 1}
    preds = results['Predicted_Status']
    assert set(preds.unique()).issubset({0, 1})
