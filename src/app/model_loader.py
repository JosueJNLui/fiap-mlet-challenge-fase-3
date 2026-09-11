"""
Carrega o(s) modelo(s) e expõe uma interface única de predição.

Suporta dois backends de inferência, selecionáveis via variável de ambiente
MODEL_BACKEND=sklearn|onnx (padrão: onnx, por ser o otimizado):
  - sklearn: usa o pipeline completo (TF-IDF + RandomForest) via scikit-learn.
  - onnx: usa TF-IDF (scikit-learn) + classificador RandomForest via ONNX Runtime.
"""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np

from app import tracing

_logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
BACKEND = os.getenv("MODEL_BACKEND", "onnx")


class ModelService:
    def __init__(self, backend: str = BACKEND, models_dir: Path = MODELS_DIR):
        self.backend = backend
        self.models_dir = models_dir
        self._load()

    def _load(self):
        with tracing.trace_step(
            "model.load", attributes={"backend": self.backend, "models.dir": str(self.models_dir)}
        ):
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
            _logger.info(
                "Modelo carregado",
                extra={"backend": self.backend, "models_dir": str(self.models_dir), "classes": len(self.classes)},
            )

    def predict(self, texto: str) -> dict:
        start = time.perf_counter()
        texto = texto.strip()
        attrs_base = {"backend": self.backend, "input.texto_chars": len(texto)}
        preview = tracing.text_preview(texto)
        if preview is not None:
            attrs_base["input.texto"] = preview

        if self.backend == "onnx":
            with tracing.trace_step("model.vectorize", attributes=attrs_base) as span:
                t0 = time.perf_counter()
                X = self.vectorizer.transform([texto])
                vectorize_ms = round((time.perf_counter() - t0) * 1000, 4)
                span.set_attribute("features", int(X.shape[1]))
                span.set_attribute("input.shape", [int(X.shape[0]), int(X.shape[1])])
                X = X.toarray().astype(np.float32)
                span.set_attribute("vectorize_ms", vectorize_ms)
                _logger.info(
                    "Texto vetorizado (TF-IDF)",
                    extra={
                        "backend": self.backend,
                        "features": int(X.shape[1]),
                        "vectorize_ms": vectorize_ms,
                        "texto_chars": len(texto),
                    },
                )

            with tracing.trace_step(
                "model.inference",
                attributes={"backend": self.backend, "provider": "CPUExecutionProvider"},
            ) as span:
                t0 = time.perf_counter()
                _, proba = self.session.run(None, {self.input_name: X})
                inference_ms = round((time.perf_counter() - t0) * 1000, 4)
                span.set_attribute("output.shape", [int(proba.shape[0]), int(proba.shape[1])])
                probs = proba[0]
                _logger.info(
                    "Inferência concluída (ONNX Runtime)",
                    extra={
                        "backend": self.backend,
                        "output_shape": [int(proba.shape[0]), int(proba.shape[1])],
                        "inference_ms": inference_ms,
                    },
                )
        else:
            with tracing.trace_step("model.predict_proba", attributes=attrs_base) as span:
                t0 = time.perf_counter()
                probs = self.pipeline.predict_proba([texto])[0]
                inference_ms = round((time.perf_counter() - t0) * 1000, 4)
                span.set_attribute("output.classes", int(len(probs)))
                _logger.info(
                    "Inferência concluída (sklearn)",
                    extra={"backend": self.backend, "output_classes": int(len(probs)), "inference_ms": inference_ms},
                )

        with tracing.trace_step(
            "model.postprocess",
            attributes={"backend": self.backend, "input.texto_chars": len(texto)},
        ) as span:
            latency_ms = (time.perf_counter() - start) * 1000
            idx = int(np.argmax(probs))
            classe = self.classes[idx]
            confianca = float(probs[idx])
            probabilidades = {c: float(p) for c, p in zip(self.classes, probs)}
            span.set_attribute("response.classificacao", classe)
            span.set_attribute("response.confianca", confianca)
            span.set_attribute("response.latencia_ms", round(latency_ms, 4))
            _logger.info(
                "Predição gerada no postprocess",
                extra={"classe": classe, "confianca": confianca, "latencia_ms": round(latency_ms, 4)},
            )

        return {
            "classificacao": classe,
            "confianca": confianca,
            "probabilidades": probabilidades,
            "modelo": self.backend,
            "latencia_ms": round(latency_ms, 4),
        }
