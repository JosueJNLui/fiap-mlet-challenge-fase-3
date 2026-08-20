"""
Converte o classificador (RandomForest) do pipeline treinado para ONNX,
visando reduzir a latência de inferência em produção.

Decisão de design: o estágio de TF-IDF permanece em Python/scikit-learn e
SOMENTE o classificador é exportado para ONNX. O operador ONNX de
normalização de texto (StringNormalizer), usado ao converter o TfidfVectorizer
inteiro, depende de locale do sistema operacional (en_US.UTF-8) e costuma
falhar em imagens Docker mínimas — um problema comum e conhecido do
onnxruntime. Exportar apenas o classificador é mais robusto e ainda captura
o principal ganho de performance, já que a árvore de decisão é o componente
computacionalmente mais custoso na inferência.

Uso:
    python src/export_onnx.py --model models/model.joblib --out models/model.onnx
"""
import argparse
import json
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--out", default="models/model.onnx")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    n_features = len(vectorizer.get_feature_names_out())
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_type,
        target_opset=15,
        options={"zipmap": False},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Salva também o vectorizer TF-IDF separadamente (usado no pré-processamento
    # antes de chamar a sessão ONNX) e a lista de classes na mesma ordem de saída.
    joblib.dump(vectorizer, out_path.parent / "tfidf_vectorizer.joblib")
    with open(out_path.parent / "classes.json", "w") as f:
        json.dump(list(classifier.classes_), f)

    print(f"Modelo ONNX (classificador) salvo em {out_path}")
    print(f"Vectorizer TF-IDF salvo em {out_path.parent / 'tfidf_vectorizer.joblib'}")


if __name__ == "__main__":
    main()
