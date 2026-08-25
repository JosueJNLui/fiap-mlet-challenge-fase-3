"""
Baseline de latência end-to-end da API em Docker (entregável da Etapa 1).

Diferente de src/benchmark.py, que isola o classificador, aqui é medido o
caminho completo: rede local -> FastAPI -> Pydantic -> TF-IDF -> classificador
-> JSON de resposta.

São reportadas três camadas, para deixar claro onde o tempo é gasto:
  - cliente  : relógio de parede do lado de quem chama (inclui rede e JSON);
  - servidor : histograma http_request_duration_seconds exposto em /metrics;
  - modelo   : campo latencia_ms devolvido pela própria resposta.

Uso:
    python src/benchmark_http.py --n 200
"""
import argparse
import json
import time
import urllib.request

TEXTO = "Paciente com dor torácica intensa e falta de ar súbita"


def post_predict(url: str, texto: str) -> dict:
    data = json.dumps({"texto": texto}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def server_latency_ms(base_url: str, endpoint: str = "/predict") -> float:
    """Média do histograma do Prometheus: _sum / _count para o endpoint."""
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        body = resp.read().decode()
    marker = f'endpoint="{endpoint}"'
    total = count = 0.0
    for line in body.splitlines():
        if marker not in line:
            continue
        if line.startswith("http_request_duration_seconds_sum"):
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith("http_request_duration_seconds_count"):
            count = float(line.rsplit(" ", 1)[1])
    return (total / count) * 1000 if count else float("nan")


def percentile(values: list, pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="número de requisições")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.url}/health", timeout=10) as resp:
        backend = json.load(resp)["modelo"]

    post_predict(f"{args.url}/predict", TEXTO)  # warmup

    client_ms, model_ms = [], []
    for _ in range(args.n):
        start = time.perf_counter()
        body = post_predict(f"{args.url}/predict", TEXTO)
        client_ms.append((time.perf_counter() - start) * 1000)
        model_ms.append(body["latencia_ms"])

    avg = sum(client_ms) / len(client_ms)
    print(f"=== Latência HTTP end-to-end, backend={backend} ({args.n} requisições) ===")
    print(f"Cliente  média : {avg:8.4f} ms")
    print(f"Cliente  p50   : {percentile(client_ms, 0.50):8.4f} ms")
    print(f"Cliente  p95   : {percentile(client_ms, 0.95):8.4f} ms")
    print(f"Servidor média : {server_latency_ms(args.url):8.4f} ms  (histograma /metrics)")
    print(f"Modelo   média : {sum(model_ms) / len(model_ms):8.4f} ms  (campo latencia_ms)")


if __name__ == "__main__":
    main()
