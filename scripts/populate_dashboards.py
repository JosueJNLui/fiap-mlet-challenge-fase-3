"""
Gerador de tráfego para popular os dashboards de observabilidade da API em execução.

Envia várias predições (POST /predict) com abstracts médicos reais das cinco
condições do Medical Abstracts TC Corpus (cardiovascular diseases, digestive
system diseases, general pathological conditions, neoplasms e nervous system
diseases), além de requests de health e casos de validação (400 e 422),
enriquecendo métricas, logs e traces nos dashboards do Grafana.

Os textos são sorteados de data/laudos.csv (a base atual), agrupados por label,
então o tráfego exercita exatamente as classes do modelo em produção.

Uso:
    python scripts/populate_dashboards.py                          # padrão: 300 predições
    python scripts/populate_dashboards.py --n 1000 --interval 0.1 \
        --url http://localhost:8000 --data data/laudos.csv
"""
import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

HEALTH_OK = (200,)
STATUS_LABEL = "status"

CLASSES = [
    "cardiovascular diseases",
    "digestive system diseases",
    "general pathological conditions",
    "neoplasms",
    "nervous system diseases",
]

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "laudos.csv"


def load_textos_por_classe(data_path: Path) -> dict[str, list[str]]:
    """Lê data/laudos.csv e agrupa os abstracts por classe (colunas texto/label)."""
    textos: dict[str, list[str]] = {c: [] for c in CLASSES}
    with open(data_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            texto = row.get("texto")
            if label in textos and texto:
                textos[label].append(texto)
    return textos


def request(
    url: str, payload: dict | None = None, timeout: int = 10
) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        return exc.code, body
    except urllib.error.URLError:
        return 0, {}


def fmt_status(status: int) -> str:
    return "ok" if status in HEALTH_OK else str(status)


def main():
    parser = argparse.ArgumentParser(
        description="Popula métricas/logs/traces da API com várias predições."
    )
    parser.add_argument("--url", default="http://localhost:8000", help="base da API")
    parser.add_argument(
        "--n", type=int, default=300, help="número de predições a enviar"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="segundos entre requisições (espalha os pontos no tempo)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="semente do sorteio dos textos"
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="CSV com colunas 'texto' e 'label' (padrão: data/laudos.csv)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    classes = list(CLASSES)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Arquivo de dados não encontrado: {data_path}")
        print("Gere o dataset com 'make model' ou aponte --data para um CSV válido.")
        raise SystemExit(1)

    textos_por_classe = load_textos_por_classe(data_path)
    for classe in classes:
        if not textos_por_classe[classe]:
            print(f"Classe '{classe}' sem textos em {data_path}.")
            raise SystemExit(1)

    status_code, _ = request(f"{args.url}/health")
    if status_code not in HEALTH_OK:
        print(f"API não respondeu /health (HTTP {status_code}). Ela está de pé?")
        raise SystemExit(1)
    print(f"API detectada em {args.url}, enviando {args.n} predições...\n")

    counts: dict[str, int] = {}
    erros_esperados = max(1, args.n // 12)  # ~8% de validações (400/422)

    for i in range(1, args.n + 1):
        classe = classes[i % len(classes)]
        texto = rng.choice(textos_por_classe[classe])
        status_code, _ = request(f"{args.url}/predict", {"texto": texto})
        counts[fmt_status(status_code)] = counts.get(fmt_status(status_code), 0) + 1

        if i % 10 == 0 or i == args.n:
            print(f"\r  {i:>5}/{args.n} predições  {dict(counts)}", end="", flush=True)
        time.sleep(args.interval)

    # Validações para enriquecer os painéis de erro (400 = texto vazio, 422 = campo ausente).
    total_erros = 0
    for _ in range(erros_esperados):
        status_code, _ = request(f"{args.url}/predict", {"texto": "   "})
        if status_code == 400:
            total_erros += 1
        status_code, _ = request(f"{args.url}/predict", {})
        if status_code == 422:
            total_erros += 1
        time.sleep(args.interval)

    print("\n")
    print(f"Predições enviadas: {args.n}   (validações 400/422: {2 * erros_esperados})")
    print(f"Distribuição HTTP: {counts}")
    print("Confira agora: Grafana > Dashboards > 'Triagem de Laudos - Logs / Métricas / Traces'.")


if __name__ == "__main__":
    main()
