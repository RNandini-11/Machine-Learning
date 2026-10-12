#Do not use browser for /predict.
#Use your test_api.py.

from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from src.config import DEFAULT_MODEL
from src.logger import logger
from src.model_registry import get_latest_model_path

# Create Flask app
app = Flask(__name__)

# Load latest versioned model
MODEL_PATH = get_latest_model_path(
    DEFAULT_MODEL
)

logger.info(
    f"Loading API model: {MODEL_PATH}"
)

model = joblib.load(
    MODEL_PATH
)


@app.route("/")
def home():
    return jsonify({
            "message":
            "Loan Default Prediction API is running!",
            "model":
            DEFAULT_MODEL
        })


@app.route("/predict", methods=["POST"])
def predict():

    # POST request
    if not request.is_json:
        return jsonify({
            "error": "Request must be JSON"
        }), 415

    data = request.get_json()

    if "features" not in data:
            return jsonify({
                    "error": "Missing features field"
                }), 400
        
    try:

        prediction = model.predict_proba([data["features"]])
        probability = float(prediction[0][1])

        logger.info("Prediction request completed successfully")

        return jsonify({
                "model": DEFAULT_MODEL,
                "default_probability": probability
            })


    except Exception as e:

        logger.error(f"Prediction failed: {str(e)}")

        return jsonify({
                "error": str(e)
            }), 500
   


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
