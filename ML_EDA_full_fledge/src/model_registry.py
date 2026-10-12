import json
from datetime import datetime
from pathlib import Path

#MODEL_DIR = Path("models")
from src.config import MODEL_REGISTRY, MODELS_DIR


def create_version():

    version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    path = MODELS_DIR / version

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path



def update_registry(version):

    registry = {
        "latest_version": version,
        "created_at": str(datetime.now()),
        "models": [
            "logistic_regression",
            "random_forest",
            "xgboost"
        ]
    }


    with open(MODELS_DIR/"model_registry.json","w") as f:
        json.dump(registry,f,indent=4)
        
def get_latest_version():

    if not MODEL_REGISTRY.exists():
        raise FileNotFoundError("Model registry not found. Run training first.")


    with open(MODEL_REGISTRY,"r") as f:
        registry = json.load(f)
        
    return registry["latest_version"]



def get_latest_model_path(model_name):

    version = get_latest_version()


    model_path = (MODELS_DIR /version /f"{model_name}.joblib")

    return model_path
