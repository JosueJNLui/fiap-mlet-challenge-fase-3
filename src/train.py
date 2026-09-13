"""
Treina o classificador de condições médicas a partir de abstracts médicos.

Pipeline: TF-IDF (vetorização) + RandomForestClassifier (classificação leve).
Salva o pipeline treinado em models/model.joblib.
Loga métricas e modelo no MLflow (DagsHub se credenciais disponíveis, senão local).

Uso:
    python src/train.py --data data/laudos.csv --out models/model.joblib
"""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from triage.config import load_settings
from triage.tracking import init_mlflow, log_classifier, promote_to_production


def build_pipeline(cfg) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=cfg.tfidf_max_features,
                ngram_range=tuple(cfg.tfidf_ngram_range)
            )),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=cfg.rf_n_estimators,
                    max_depth=cfg.rf_max_depth,
                    min_samples_leaf=cfg.rf_min_samples_leaf,
                    class_weight=cfg.rf_class_weight,
                    random_state=cfg.rf_random_state,
                    n_jobs=-1
                ),
            ),
        ]
    )


def set_global_seeds(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/laudos.csv")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    settings = load_settings()
    cfg = settings.train

    set_global_seeds(cfg.random_state)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["texto", "label"])

    # Verificar classes
    classes = sorted(df["label"].unique())
    print(f"Classes encontradas ({len(classes)}): {classes}")
    print(f"Distribuição:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["label"], test_size=cfg.test_size,
        random_state=cfg.random_state, stratify=df["label"]
    )

    pipeline = build_pipeline(cfg)

    # Inicializa MLflow (DagsHub ou fallback local)
    tracking_uri = init_mlflow(settings)
    print(f"MLflow tracking: {tracking_uri}")

    start = time.time()
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nTempo de treino: {train_time:.2f}s")
        print(classification_report(y_test, y_pred))

        # Log métricas
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("tfidf_max_features", cfg.tfidf_max_features)
        mlflow.log_param("tfidf_ngram_range", cfg.tfidf_ngram_range)
        mlflow.log_param("rf_n_estimators", cfg.rf_n_estimators)
        mlflow.log_param("rf_max_depth", cfg.rf_max_depth)
        mlflow.log_param("rf_min_samples_leaf", cfg.rf_min_samples_leaf)
        mlflow.log_param("rf_class_weight", cfg.rf_class_weight)
        mlflow.log_param("test_size", cfg.test_size)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))
        mlflow.log_param("n_classes", len(classes))

        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])
        mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
        for cls in classes:
            if cls in report:
                mlflow.log_metric(f"f1_{cls.replace(' ', '_')}", report[cls]["f1-score"])

        # Log modelo e registra
        example = pd.DataFrame({"texto": X_test[:5].tolist()})
        registered_name = "MedicalAbstractsClassifier"
        log_classifier(pipeline, example, registered_name)

        # Promove para production se métrica melhorar (só no DagsHub)
        if settings.dagshub_token:
            promote_to_production(registered_name, report["macro avg"]["f1-score"], "f1_macro")

    # Salva artefato local (para API/Docker)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Modelo salvo em {out_path}")


if __name__ == "__main__":
    import mlflow
    main()
