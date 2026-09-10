import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_sklearn_model_loads_and_predicts():
    pipeline = joblib.load(MODELS_DIR / "model.joblib")
    pred = pipeline.predict(["Patient with chest pain and shortness of breath"])
    assert pred[0] in {
        "cardiovascular diseases",
        "digestive system diseases",
        "general pathological conditions",
        "neoplasms",
        "nervous system diseases",
    }


def test_onnx_files_exist():
    assert (MODELS_DIR / "model.onnx").exists()
    assert (MODELS_DIR / "tfidf_vectorizer.joblib").exists()
    assert (MODELS_DIR / "classes.json").exists()
