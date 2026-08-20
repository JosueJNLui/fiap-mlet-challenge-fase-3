"""
Compara a latência de inferência do CLASSIFICADOR: sklearn (RandomForest)
original versus o classificador convertido para ONNX Runtime.
O estágio de TF-IDF é idêntico nos dois casos (não é o gargalo otimizado).

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


def bench_sklearn(pipeline, X, n_runs):
    pipeline.named_steps["clf"].predict_proba(X[:1])  # warmup
    start = time.perf_counter()
    for _ in range(n_runs):
        pipeline.named_steps["clf"].predict_proba(X[:1])
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

    sk_latency = bench_sklearn(pipeline, X, args.n)
    onnx_latency = bench_onnx(session, X, args.n)

    improvement = (1 - onnx_latency / sk_latency) * 100

    print("=== Resultado do Benchmark de Latência do Classificador (por requisição) ===")
    print(f"Sklearn RandomForest (original) : {sk_latency:.4f} ms")
    print(f"ONNX Runtime (otimizado)         : {onnx_latency:.4f} ms")
    print(f"Melhoria                          : {improvement:.1f}%")
    print()
    print("Obs: o estágio de TF-IDF (idêntico em ambos os casos) não está incluso,")
    print("pois o objetivo é isolar o ganho da otimização aplicada ao classificador.")


if __name__ == "__main__":
    main()
