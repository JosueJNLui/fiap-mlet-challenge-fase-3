"""
Gerador de tráfego para popular os dashboards de observabilidade da API em execução.

Envia várias triagens (POST /predict) com textos de sintomas variados por classe
(normal / atencao / urgente), além de requests de health e casos de validação
(400 e 422), enriquecendo métricas, logs e traces nos dashboards do Grafana.

Uso:
    python scripts/populate_dashboards.py                          # padrão: 300 predições
    python scripts/populate_dashboards.py --n 1000 --interval 0.1 \
        --url http://localhost:8000
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

HEALTH_OK = (200,)
STATUS_LABEL = "status"

TEXTOS = {
    "normal": [
        "Paciente relata check-up de rotina, sem queixas relevantes.",
        "Raio-x de tórax sem alterações, paciente assintomático.",
        "Consulta de retorno, paciente estável, sem sintomas novos.",
        "Hemograma completo dentro da normalidade, sem sinais de infecção.",
        "Vacinação de rotina realizada sem intercorrências.",
    ],
    "atencao": [
        "Paciente com febre baixa persistente há 3 dias, sem outros sintomas graves.",
        "Paciente relata tosse persistente há uma semana, sem falta de ar.",
        "Pressão arterial levemente elevada, recomendado acompanhamento.",
        "Dor lombar moderada, sem sinais neurológicos associados.",
        "Glicemia de jejum levemente alterada, indicado acompanhamento nutricional.",
    ],
    "urgente": [
        "Paciente apresenta dor torácica intensa e falta de ar súbita.",
        "Sinais de acidente vascular cerebral: perda de força e fala arrastrada.",
        "Paciente com febre alta e rigidez de nuca, suspeita de meningite.",
        "Reação alérgica grave com edema de glote e dificuldade respiratória.",
        "Politraumatismo grave após queda de altura, necessita intervenção imediata.",
    ],
}


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
        description="Popula métricas/logs/traces da API com várias triagens."
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
    args = parser.parse_args()

    random.seed(args.seed)
    classes = list(TEXTOS)

    status_code, _ = request(f"{args.url}/health")
    if status_code not in HEALTH_OK:
        print(f"API não respondeu /health (HTTP {status_code}). Ela está de pé?")
        raise SystemExit(1)
    print(f"API detectada em {args.url} — enviando {args.n} predições...\n")

    counts: dict[str, int] = {}
    erros_esperados = max(1, args.n // 12)  # ~8% de validações (400/422)

    for i in range(1, args.n + 1):
        classe = classes[i % len(classes)]
        texto = random.choice(TEXTOS[classe])
        status_code, body = request(f"{args.url}/predict", {"texto": texto})
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