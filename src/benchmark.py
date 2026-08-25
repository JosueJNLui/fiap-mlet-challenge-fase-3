"""
Compara a latência de inferência do CLASSIFICADOR: sklearn (RandomForest)
original versus o classificador convertido para ONNX Runtime.
O estágio de TF-IDF é idêntico nos dois casos (não é o gargalo otimizado).

O sklearn é medido em duas configurações (n_jobs=-1, como o modelo foi
treinado, e n_jobs=1) porque em inferência unitária o overhead de despacho
de threads do joblib domina o tempo e distorceria a comparação.

Uso:
    python src/benchmark.py --n 300
"""
import argparse
import time

import joblib
import numpy as np
import onnxruntime as rt

SAMPLE_TEXTS = [
    "Paciente apresenta dor torácica intensa e falta de ar súbita.",
    "Exame de sangue dentro dos parâmetros normais, sem alterações significativas.",
    "Queixa de dor abdominal leve a moderada, sem sinais de alarme.",
]


def bench_sklearn(clf, X, n_runs):
    clf.predict_proba(X[:1])  # warmup
    start = time.perf_counter()
    for _ in range(n_runs):
        clf.predict_proba(X[:1])
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms por requisição


def bench_onnx(session, X, n_runs):
    input_name = session.get_inputs()[0].name
    x = X[:1].astype(np.float32)
    session.run(None, {input_name: x})  # warmup
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: x})
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--onnx", default="models/model.onnx")
    parser.add_argument("--n", type=int, default=300, help="numero de requisicoes simuladas")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    session = rt.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    X = pipeline.named_steps["tfidf"].transform(SAMPLE_TEXTS).toarray()
    clf = pipeline.named_steps["clf"]

    # Duas baselines sklearn: como o modelo foi treinado (n_jobs=-1) e com o
    # paralelismo desligado. Para uma unica amostra o joblib gasta mais tempo
    # despachando threads do que percorrendo as arvores, entao comparar so com
    # n_jobs=-1 superestimaria o ganho do ONNX.
    clf.n_jobs = -1
    sk_parallel = bench_sklearn(clf, X, args.n)
    clf.n_jobs = 1
    sk_serial = bench_sklearn(clf, X, args.n)
    onnx_latency = bench_onnx(session, X, args.n)

    print(f"=== Latencia do classificador por requisicao (1 amostra, {args.n} execucoes) ===")
    print(f"Sklearn RandomForest, n_jobs=-1 : {sk_parallel:8.4f} ms")
    print(f"Sklearn RandomForest, n_jobs=1  : {sk_serial:8.4f} ms")
    print(f"ONNX Runtime (otimizado)        : {onnx_latency:8.4f} ms")
    print()
    print(f"Melhoria vs n_jobs=-1 : {(1 - onnx_latency / sk_parallel) * 100:5.1f}%"
          f"  ({sk_parallel / onnx_latency:.0f}x mais rapido)")
    print(f"Melhoria vs n_jobs=1  : {(1 - onnx_latency / sk_serial) * 100:5.1f}%"
          f"  ({sk_serial / onnx_latency:.0f}x mais rapido)")
    print()
    print("Notas:")
    print("- O estagio de TF-IDF e identico nos dois casos e ficou de fora, para")
    print("  isolar o ganho da otimizacao aplicada ao classificador.")
    print("- n_jobs=-1 e a configuracao real do modelo treinado (src/train.py); em")
    print("  inferencia unitaria ela e a pior das duas, porque o overhead de")
    print("  despacho do joblib domina o tempo. n_jobs=1 e a baseline mais justa")
    print("  para medir o ganho do ONNX Runtime em si.")


if __name__ == "__main__":
    main()
