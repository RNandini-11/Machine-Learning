from src.config import DEFAULT_MODEL
from src.logger import logger
from src.modeling.predict import predict_default_probability

if __name__ == '__main__':
    
    logger.info(
        f"Batch prediction started using model: {DEFAULT_MODEL}"
    )
    
    results = predict_default_probability()
    print("="*60)
    print("BATCH PREDICTION INFERENCE")
    print("="*60)
    print(f"Model Used: {DEFAULT_MODEL}")
    print(f"Evaluated {len(results):,} sample applications.")
    print("Sample Prediction Output:")
    print(results[['Predicted_Default_Prob', 'Predicted_Status']].head(10))
    print("="*60)
    
    logger.info(
        "Batch prediction completed successfully"
    )
