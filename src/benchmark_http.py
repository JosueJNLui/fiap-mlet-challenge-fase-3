"""
Baseline de latência end-to-end da API em Docker (entregável da Etapa 1).

Diferente de src/benchmark.py, que isola o classificador, aqui é medido o
caminho completo: rede local -> FastAPI -> Pydantic -> TF-IDF -> classificador
-> JSON de resposta.

São reportadas três camadas, para deixar claro onde o tempo é gasto:
  - cliente  : relógio de parede do lado de quem chama (inclui rede e JSON);
  - servidor : delta do histograma http_request_duration_seconds em /metrics;
  - modelo   : campo latencia_ms devolvido pela própria resposta.

Uso:
    python src/benchmark_http.py --n 200
"""
import argparse
import json
import time
import urllib.request

# Abstract real de "cardiovascular diseases" do data/laudos.csv, com tamanho próximo da
# mediana do corpus (1.161 caracteres; mediana 1.210). O TF-IDF foi treinado em inglês:
# texto em português vira um vetor todo zero e a medição deixa de representar uma entrada real.
# Também é usado por src/benchmark.py.
SAMPLE_TEXT = (
    "Multivariate analysis in the prediction of death in hospital after acute myocardial infarction. "
    "Prognostic factors in patients with acute myocardial infarction based on clinical and investigative "
    "data on admission were evaluated prospectively in 111 consecutive patients. Seventeen patients "
    "(15.3%) died during hospital stay. Age, a previous infarct, high Killip class, cardiomegaly, high "
    "serum concentrations of cardiac enzymes, a low ejection fraction, and a high wall motion score index "
    "correlated significantly with in-hospital mortality; whereas sex, risk factors, and pericardial "
    "effusion did not. Multivariate analysis showed that age and the wall motion score index were the "
    "best predictors of death in hospital. Wall motion detected by cross sectional echocardiography may "
    "reflect the extent of myocardial involvement. Age and wall motion score index predicted in-hospital "
    "mortality with a sensitivity of 76.5%, a specificity of 91.5%, and a predictive accuracy of 89.2%. "
    "Age and the wall motion score index can be determined on admission and are useful for identifying "
    "patients at high risk of cardiac death who might benefit from early intervention. "
)


def post_predict(url: str, texto: str) -> dict:
    data = json.dumps({"texto": texto}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def histogram_snapshot(base_url: str, endpoint: str = "/predict") -> tuple:
    """Lê (_sum, _count) do histograma do Prometheus para o endpoint."""
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
    return total, count


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

    post_predict(f"{args.url}/predict", SAMPLE_TEXT)  # warmup

    # o histograma é acumulado desde o start do container: só o delta do laço
    # descreve este benchmark, senão as requisições frias do warm-up entram na média
    sum_before, count_before = histogram_snapshot(args.url)

    client_ms, model_ms = [], []
    for _ in range(args.n):
        start = time.perf_counter()
        body = post_predict(f"{args.url}/predict", SAMPLE_TEXT)
        client_ms.append((time.perf_counter() - start) * 1000)
        model_ms.append(body["latencia_ms"])

    sum_after, count_after = histogram_snapshot(args.url)
    requests = count_after - count_before
    server_avg = (sum_after - sum_before) / requests * 1000 if requests else float("nan")

    avg = sum(client_ms) / len(client_ms)
    print(f"=== Latência HTTP end-to-end, backend={backend} ({args.n} requisições) ===")
    print(f"Cliente  média : {avg:8.4f} ms")
    print(f"Cliente  p50   : {percentile(client_ms, 0.50):8.4f} ms")
    print(f"Cliente  p95   : {percentile(client_ms, 0.95):8.4f} ms")
    print(f"Servidor média : {server_avg:8.4f} ms  (histograma /metrics)")
    print(f"Modelo   média : {sum(model_ms) / len(model_ms):8.4f} ms  (campo latencia_ms)")


if __name__ == "__main__":
    main()
