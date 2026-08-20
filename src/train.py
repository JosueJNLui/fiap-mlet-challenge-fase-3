"""
Treina o classificador de urgência de laudos médicos.

Pipeline: TF-IDF (vetorização) + RandomForestClassifier (classificação leve).
Salva o pipeline treinado em models/model.joblib.

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


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/laudos.csv")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["texto", "label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = build_pipeline()

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)
    print(f"Tempo de treino: {train_time:.2f}s")
    print(classification_report(y_test, y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Modelo salvo em {out_path}")


if __name__ == "__main__":
    main()
