"""
Carrega o(s) modelo(s) e expõe uma interface única de predição.

Suporta dois backends de inferência, selecionáveis via variável de ambiente
MODEL_BACKEND=sklearn|onnx (padrão: onnx, por ser o otimizado):
  - sklearn: usa o pipeline completo (TF-IDF + RandomForest) via scikit-learn.
  - onnx: usa TF-IDF (scikit-learn) + classificador RandomForest via ONNX Runtime.
"""
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
BACKEND = os.getenv("MODEL_BACKEND", "onnx")


class ModelService:
    def __init__(self, backend: str = BACKEND, models_dir: Path = MODELS_DIR):
        self.backend = backend
        self.models_dir = models_dir
        self._load()

    def _load(self):
        if self.backend == "onnx":
            import onnxruntime as rt

            self.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.session = rt.InferenceSession(
                str(self.models_dir / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            with open(self.models_dir / "classes.json") as f:
                self.classes = json.load(f)
        else:
            self.pipeline = joblib.load(self.models_dir / "model.joblib")
            self.classes = list(self.pipeline.named_steps["clf"].classes_)

    def predict(self, texto: str) -> dict:
        start = time.perf_counter()

        if self.backend == "onnx":
            X = self.vectorizer.transform([texto]).toarray().astype(np.float32)
            _, proba = self.session.run(None, {self.input_name: X})
            probs = proba[0]
        else:
            probs = self.pipeline.predict_proba([texto])[0]

        latency_ms = (time.perf_counter() - start) * 1000
        idx = int(np.argmax(probs))
        classe = self.classes[idx]
        confianca = float(probs[idx])
        probabilidades = {c: float(p) for c, p in zip(self.classes, probs)}

        return {
            "classificacao": classe,
            "confianca": confianca,
            "probabilidades": probabilidades,
            "modelo": self.backend,
            "latencia_ms": round(latency_ms, 4),
        }
