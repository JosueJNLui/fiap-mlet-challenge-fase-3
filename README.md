# Classificação de Condições Médicas a partir de Abstracts

Projeto do Tech Challenge Fase 3 (FIAP MLET). Um classificador de texto leve, servido por uma
API REST em container, com pipeline CI/CD, orquestração de retreino, observabilidade completa e
otimização de latência.

```mermaid
flowchart LR
    subgraph treino["Treino e otimização"]
        CSV["data/laudos.csv<br/>14.438 abstracts (Medical Abstracts TC Corpus)"]
        TRAIN["src/train.py<br/>TF-IDF + RandomForest"]
        EXPORT["src/export_onnx.py"]
        ONNX["models/model.onnx"]
    end
    subgraph servico["Serviço"]
        API["FastAPI em Docker<br/>POST /predict via ONNX Runtime"]
    end
    subgraph obs["Observabilidade"]
        PROM["Prometheus"]
        TEMPO["Tempo"]
        LOKI["Loki"]
        GRAF["Grafana<br/>3 dashboards"]
    end
    DAG["Airflow<br/>DAG semanal de retreino"]
    CI["GitHub Actions<br/>lint / test / build"]

    CSV --> TRAIN --> EXPORT --> ONNX --> API
    API -->|scrape /metrics| PROM
    API -->|OTLP traces| TEMPO
    API -->|OTLP logs| LOKI
    PROM --> GRAF
    TEMPO --> GRAF
    LOKI --> GRAF
    DAG -.retreino.-> CSV
    CI -.build da imagem.-> API
```

---

## Links

| O quê | Onde |
|---|---|
| Vídeo STAR (até 5 min) | _a preencher_ |
| Documentação interativa da API | http://localhost:8000/docs (após `make up`) |
| Dashboards do Grafana | http://localhost:3000 (após `make up`) |
| Airflow | http://localhost:8080 (após `make airflow-up`) |
| Resultados de latência | [`docs/benchmark.txt`](docs/benchmark.txt) |
| Evidências visuais | [`docs/`](docs/): 3 prints de dashboard e o print da DAG |

---

## Mapeamento dos critérios de avaliação

| Critério | Peso | Onde está |
|---|---|---|
| Modelagem e Otimização | 20% | [seção 5](#5-otimização-de-latência), `src/train.py`, `src/export_onnx.py`, `docs/benchmark.txt` |
| CI/CD (GitHub Actions) | 15% | [seção 7](#7-cicd-github-actions), `.github/workflows/ci.yml` |
| Orquestração (Airflow) | 15% | [seção 8](#8-orquestração-de-retreino-airflow), `airflow/dags/triage_training_dag.py`, `docs/airflow_dag.png` |
| Monitoramento | 20% | [seção 6](#6-observabilidade), `docker/docker-compose.yml`, `docker/grafana/dashboards/` |
| Documentação (README) | 15% | [seção 2](#2-arquitetura-de-deploy-em-nuvem) (decisão de nuvem) e [seção 3](#3-como-executar) (execução) |
| Vídeo STAR | 15% | [seção 11](#11-vídeo-star) |

---

## 1. Visão geral

O sistema classifica abstracts médicos em uma de cinco condições médicas, usando o dataset real **Medical Abstracts TC Corpus**:

| Classe | Significado |
|---|---|
| `cardiovascular diseases` | Doenças cardiovasculares |
| `digestive system diseases` | Doenças do sistema digestivo |
| `general pathological conditions` | Condições patológicas gerais |
| `neoplasms` | Neoplasias |
| `nervous system diseases` | Doenças do sistema nervoso |

**Modelo:** TF-IDF (`max_features=5000`, n-gramas 1 a 2) seguido de `RandomForestClassifier`
(100 árvores, `max_depth=15`), treinado com `scikit-learn` sobre 14.438 abstracts reais.
O classificador é exportado para ONNX e servido pelo ONNX Runtime, que é o backend padrão da API.

**Stack:** FastAPI, ONNX Runtime, Docker Compose, Prometheus, Grafana, Tempo, Loki, Airflow e
GitHub Actions.

---

## 2. Arquitetura de deploy em nuvem

### Batch ou real-time?

A triagem existe para decidir a ordem de atendimento **enquanto o paciente está no hospital**.
Um lote noturno entregaria a classificação depois que a decisão já foi tomada, o que anula o
propósito do sistema. O requisito é, portanto, **inferência real-time (síncrona)**: uma chamada
HTTP por laudo, com resposta em poucos milissegundos.

O volume ajuda a fechar a decisão: um hospital de referência gera dezenas a centenas de laudos por
hora, não milhões. É carga baixa e contínua, com picos previsíveis nos horários de pico do
pronto-socorro. Isso pede um serviço pequeno sempre de pé, não um cluster elástico.

### Recomendação: AWS ECS Fargate + Application Load Balancer

```
                       +---------------------------+
                       |   Sistema hospitalar       |
                       |   (HIS / prontuário)       |
                       +-------------+-------------+
                                     | HTTPS POST /predict
                                     v
                       +---------------------------+
                       |  Application Load Balancer |
                       |  (health check em /health) |
                       +-------------+-------------+
                                     |
                     +---------------+---------------+
                     |                               |
             +-------v-------+               +-------v-------+
             | ECS Fargate   |               | ECS Fargate   |
             | task (API)    |     ...       | task (API)    |
             | imagem do ECR |               | imagem do ECR |
             +---+-------+---+               +---+-------+---+
                 |       |                       |       |
        /metrics |       | OTLP (traces e logs)  |       |
                 v       v                       v       v
        +--------+--+  +-+---------------------+-+  +----+-------+
        | Amazon    |  | AWS Distro for        |   | Amazon      |
        | Managed   |  | OpenTelemetry (ADOT)  |   | CloudWatch  |
        | Prometheus|  | -> X-Ray / CloudWatch |   | Logs        |
        +-----+-----+  +-----------------------+   +-------------+
              |
              v
        +-----+---------------+          +------------------------+
        | Amazon Managed      |          | S3: artefatos do modelo |
        | Grafana (dashboards)|          | (model.onnx, vectorizer)|
        +---------------------+          +------------+------------+
                                                      ^
                                                      | escreve o novo modelo
                                         +------------+------------+
                                         | Amazon MWAA (Airflow)   |
                                         | DAG semanal de retreino |
                                         +-------------------------+
```

**Por que ECS Fargate:** a imagem Docker do serviço já existe e roda igual em qualquer lugar; o
Fargate a executa sem nenhum servidor para provisionar, corrigir ou escalar manualmente. Os artefatos
do modelo somam cerca de 9 MB e carregam em memória no start, então uma task com 0.5 vCPU e 1 GB
atende com folga, e o autoscaling por número de requisições cobre os picos. O ALB entrega TLS, health check em `/health`
e distribuição entre tasks sem código adicional.

**Alternativas descartadas:**

- **AWS Lambda:** cold start com `onnxruntime` e `scikit-learn` no pacote custa centenas de
  milissegundos, o que anula o ganho de latência que este projeto foi otimizar.
- **SageMaker Endpoint:** custo e complexidade operacional desproporcionais para um RandomForest de
  200 árvores; faz sentido para modelos grandes com GPU, não para este.
- **EKS:** o overhead de operar um cluster Kubernetes não se paga para um único serviço stateless.
- **Batch (S3 + AWS Batch / Glue):** descartado pelo requisito de negócio, conforme acima.

**Componentes de apoio:**

- **ECR** guarda a imagem; o job de build do CI já produz exatamente essa imagem e só precisaria de
  um passo de `push` com credenciais OIDC.
- **S3** guarda os artefatos do modelo (`model.onnx`, `tfidf_vectorizer.joblib`, `classes.json`),
  versionados. Hoje eles são copiados para dentro da imagem no build; em nuvem o retreino publica
  no S3 e a task busca no start, o que desacopla o ciclo do modelo do ciclo do código.
- **Amazon Managed Prometheus + Amazon Managed Grafana** recebem os mesmos sinais da stack local.
  A API expõe métricas no formato de exposição do Prometheus e envia traces e logs por OTLP puro,
  que é o protocolo que o ADOT Collector fala nativamente, então **a migração não exige mudança de
  código**, apenas de endpoint.
- **Amazon MWAA** executa a mesma DAG de retreino sem alteração.

---

## 3. Como executar

Pré-requisitos: Docker (com o daemon rodando, usado inclusive pelos lints), `uv` e Python 3.11.

```bash
make setup    # cria o .venv com Python 3.11 e instala todas as dependências
make model    # gera o dataset sintético, treina e exporta para ONNX
make check    # lint (flake8 + hadolint + DCLint + ty) e pytest, o mesmo que o CI roda
make up       # sobe a stack completa: API + Prometheus + Grafana + Tempo + Loki
```

Serviços após o `make up`:

| Serviço | URL | Observação |
|---|---|---|
| API | http://localhost:8000 | docs interativas em `/docs` |
| Prometheus | http://localhost:9090 | alvo `triagem-api` em `/targets` |
| Grafana | http://localhost:3000 | `admin` / `admin`, ou acesso anônimo somente leitura |
| Tempo | http://localhost:3200 | consultado pelo Grafana, não pelo navegador |
| Loki | http://localhost:3100 | consultado pelo Grafana, não pelo navegador |

Demais alvos:

```bash
make bench        # latência do classificador e da API HTTP (exige a stack de pé)
make airflow-up   # sobe o Airflow standalone em http://localhost:8080 (UI sem login)
make airflow-down # derruba o Airflow
make down         # derruba a stack de observabilidade
make help         # lista todos os alvos
```

Para popular os dashboards com tráfego realista:

```bash
.venv/bin/python scripts/populate_dashboards.py --n 300
```

---

## 4. API

### `POST /predict`

```bash
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto": "Patient presents with acute chest pain radiating to left arm, shortness of breath, and diaphoresis. ECG shows ST elevation in leads V1-V4. Troponin markedly elevated. Clinical picture consistent with acute anterior myocardial infarction."}'
```

```json
{
  "classificacao": "cardiovascular diseases",
  "confianca": 0.875,
  "probabilidades": {
    "cardiovascular diseases": 0.875,
    "digestive system diseases": 0.025,
    "general pathological conditions": 0.065,
    "neoplasms": 0.020,
    "nervous system diseases": 0.015
  },
  "modelo": "onnx",
  "latencia_ms": 1.42
}
```

O campo `latencia_ms` mede apenas TF-IDF mais classificador, sem o custo de HTTP.

### `GET /health`

```bash
curl -s localhost:8000/health
# {"status":"ok","modelo":"onnx"}
```

### `GET /metrics`

Formato de exposição do Prometheus, gerado pelo `prometheus_client`. É o alvo do scrape.

### Erros

| Situação | Status |
|---|---|
| `texto` com menos de 3 caracteres ou campo ausente | `422` (validação do Pydantic) |
| `texto` só com espaços | `400` |

O backend de inferência é escolhido pela variável `MODEL_BACKEND` (`onnx`, o padrão, ou `sklearn`).

---

## 5. Otimização de latência

A técnica aplicada foi a **conversão do classificador para ONNX e a inferência via ONNX Runtime**.

**Por que só o classificador foi para ONNX:** converter o `TfidfVectorizer` inteiro traz o operador
`StringNormalizer`, que depende de locale do sistema operacional e falha em imagens Docker mínimas,
um problema conhecido do onnxruntime. O TF-IDF continua em Python e apenas o RandomForest, que é o
componente computacionalmente caro, vai para ONNX. A justificativa completa está em
`src/export_onnx.py`.

Medições em `docs/benchmark.txt` (Linux, Ubuntu 24.04, Python 3.11, onnxruntime 1.19.2, scikit-learn
1.5.2), medido em 2026-09-11, reprodutíveis com `make bench`:

**Escopo 1: classificador isolado** (1 amostra, 500 execuções)

| Backend | Latência média | Ganho |
|---|---|---|
| sklearn RandomForest, `n_jobs=1` | 3.9613 ms | baseline |
| ONNX Runtime | 0.0606 ms | **65x mais rápido** |
| sklearn RandomForest, `n_jobs=-1` | 22.7492 ms | 376x (nota de rodapé) |

A baseline honesta é `n_jobs=1`. Com `n_jobs=-1`, que é a configuração real do treino, quase todo o
tempo é despacho de threads do joblib e não trabalho do modelo, o que infla o ganho por um motivo
que não tem relação com a otimização.

**Escopo 2: HTTP end-to-end**, API em Docker (200 requisições)

| Backend | Cliente média | p50 | p95 | Servidor média | Modelo média |
|---|---|---|---|---|---|
| sklearn | 33.2259 ms | 28.9411 ms | 44.6106 ms | 29.6937 ms | 26.2552 ms |
| onnx | 15.6711 ms | 12.5346 ms | 34.1369 ms | 8.3586 ms | 2.1027 ms |

**Ganho end-to-end: 52.8%, ou 2.1x mais rápido.**

O ganho de 2.1x é menor que os 65x do classificador isolado, e isso é o resultado esperado: a
otimização **move o gargalo**. Com sklearn o modelo consome 79.0% do tempo do cliente; com ONNX cai
para 13.4%, e o que sobra é a camada HTTP mais o custo dos três sinais de observabilidade por
requisição.

---

## 6. Observabilidade

A API emite três sinais:

| Sinal | Como | Destino |
|---|---|---|
| Métricas | `prometheus_client` em `GET /metrics` | Prometheus (scrape a cada 5s) |
| Traces | OpenTelemetry: instrumentação automática do FastAPI + spans/fases manuais | Tempo (OTLP/HTTP) |
| Logs | `logging` do Python com `trace_id` e `span_id` correlacionados | Loki (OTLP/HTTP) |

**Decisão:** métricas ficaram integralmente com o `prometheus_client`, citado nominalmente no
enunciado, e o OpenTelemetry ficou apenas com traces e logs. Assim existe **um** caminho de
métricas, e o nome de cada série é literalmente o que está escrito em `src/app/main.py`. O
middleware ignora o próprio `/metrics` para o scrape não se autocontabilizar, e o
`FastAPIInstrumentor` recebe `excluded_urls="metrics"` para que o scrape não vire trace no Tempo.

Séries expostas: `http_requests_total`, `http_errors_total`, `http_request_duration_seconds`,
`http_server_active_requests`, `predictions_total`, `model_predict_duration_seconds` e
`model_predict_confidence`. `tests/test_metrics.py` garante que nenhuma delas suma numa renomeação
silenciosa e derrube um painel.

São 3 dashboards provisionados automaticamente, 19 painéis no total.

### Métricas da API (10 painéis)

Total de requisições por status, latência p50/p95/p99, QPS por endpoint, taxa de erro, requisições
em andamento, predições por classe, latência de inferência, confiança das predições e total de
predições.

![Dashboard de métricas](docs/dashboard_metricas.png)

### Traces (5 painéis)

![Dashboard de traces](docs/dashboard_traces.png)

#### Node graph detalhado por fase

Além do nó HTTP automático do FastAPI (`POST /predict`), cada predição abre nós
(fases) com eventos e atributos próprios, visíveis no **node graph** / *Trace
View* do Tempo:

| Nó | O que registra |
|---|---|
| `predict` | validação do payload, classes/confiança/modelo retornados, aviso de baixa confiança (`< 0.5`) |
| `model.vectorize` | TF-IDF: nº de features, shape da entrada, duração da fase |
| `model.inference` | ONNX Runtime: shape da saída (probs), duração da fase |
| `model.predict_proba` | backend sklearn: nº de classes de saída |
| `model.postprocess` | argmax/classe escolhida, confiança, latência total |
| `model.load` | carregamento do modelo no boot (backend, diretório) |

Cada fase emite eventos `*.start` e `*.ok` (ou `*.error` + `exception` em falha)
e o atributo `phase.latency_ms`. Exceções ficam com `StatusCode.ERROR` e a
mensagem/StackTrace no span.

#### Verbosidade controlada por env var

O quanto entra em cada nó é controlado por variáveis de ambiente na API (não
exige rebuild de imagem):

| Env var | Valores | Efeito |
|---|---|---|
| `OTEL_ENABLED` | `true`/`false` | liga/desliga traces + logs inteiramente |
| `TRACING_LEVEL` | `error`/`warning`/`info` | `error` só exceções; `warning` soma avisos (payload vazio, baixa confiança); `info` soma eventos de progresso por fase |
| `TRACING_LOG_PAYLOAD` | `true`/`false` | inclui o texto do abstract (truncado) como atributo do span |
| `TRACING_MAX_TEXT_CHARS` | int | tamanho máximo do texto armazenado em atributo de span (padrão `200`) |

A implementação é `src/app/tracing.py` (`trace_step`, `add_event`); as env vars
padrão estão em `docker/docker-compose.yml` e documentadas em `.env.example`.

#### Logs estruturados por fase (Loki)

Cada fase que gera span também emite um log estruturado com os mesmos campos do
nó de trace (correlacionados via `trace_id`/`span_id` na linha), permitindo o
salto log -> trace no Grafana:

| Log | Nível | Campos |
|---|---|---|
| `Requisição recebida` | info | método, path, content-length |
| `Requisição concluída` | info | método, path, status, duração |
| `Requisição finalizada com erro de cliente` | warning | path, status (4xx/5xx) |
| `Erro não tratado na requisição` | error | método, path + stacktrace |
| `Payload validado` / `Predição rejeitada: texto vazio.` | info / warning | tamanho do texto, motivo |
| `Texto vetorizado (TF-IDF)` / `Inferência concluída (ONNX Runtime)` / `Inferência concluída (sklearn)` | info | backend, features, shape, duração da fase |
| `Predição gerada no postprocess` / `Predição realizada` | info | classe, confiança, latência |
| `Confiança abaixo de 0.5` | warning | classe, confiança |
| `Modelo carregado` | info | backend, diretório, nº de classes |

A verbosidade é controlada por `LOG_LEVEL=error|warning|info` (padrão `info`),
que limita tanto o handler OTLP quanto os loggers do pacote `app`; o env var é
lido em `src/app/telemetry.py`.

### Logs (4 painéis)

![Dashboard de logs](docs/dashboard_logs.png)

Os dashboards são provisionados por `docker/grafana/provisioning/`, então sobem prontos com o
`make up`. Os JSONs estão em `docker/grafana/dashboards/`.

---

## 7. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` dispara em push para `main` e `develop` e em pull request para `main`,
com três jobs encadeados (`lint` -> `test` -> `build`):

| Job | O que faz |
|---|---|
| **lint** | flake8 (Python), hadolint (Dockerfile), DCLint (os dois composes) e `ty` (validação estática de anotações de tipo) |
| **test** | gera o dataset, treina o modelo e roda o pytest (21 testes) |
| **build** | reconstrói os artefatos e gera a imagem Docker com tag `${{ github.sha }}`, sem publicar |

O CI reaproveita os mesmos alvos do `Makefile` usados localmente (`make ci-install PY=python`,
`make lint-python`, `make test`, `make build`), então `make check` na máquina reproduz exatamente o
que roda no runner.

---

## 8. Orquestração de retreino (Airflow)

A DAG `triage_model_training_pipeline` (`airflow/dags/triage_training_dag.py`) tem
`schedule="@weekly"` e quatro tasks sequenciais:

```
load_data  ->  train_model  ->  export_onnx  ->  validate_model
```

| Task | O que faz |
|---|---|
| `load_data` | baixa e processa o Medical Abstracts TC Corpus em `data/laudos.csv` |
| `train_model` | treina o pipeline TF-IDF + RandomForest e salva `models/model.joblib` |
| `export_onnx` | converte o classificador para `models/model.onnx` |
| `validate_model` | falha a DAG se qualquer um dos quatro artefatos não tiver sido gerado |

```bash
make airflow-up   # http://localhost:8080, UI sem login
```

O compose (`docker/docker-compose.airflow.yml`) sobe o Airflow em modo `standalone` e monta o
repositório em `/opt/airflow/project`, então os artefatos gerados pelas tasks aparecem direto em
`models/` no host. As quatro tasks concluem em cerca de 6 segundos:

![DAG do Airflow](docs/airflow_dag.png)

O `validate_model` é o portão de qualidade do pipeline: sem os quatro artefatos, nada é liberado
para a API.

---

## 9. Dataset

`data/laudos.csv`: **14.438 abstracts médicos reais** do **Medical Abstracts TC Corpus** (https://github.com/sebischair/Medical-Abstracts-TC-Corpus).

| Classe | Amostras |
|---|---|
| `general pathological conditions` | 4.805 |
| `neoplasms` | 3.163 |
| `cardiovascular diseases` | 3.051 |
| `nervous system diseases` | 1.925 |
| `digestive system diseases` | 1.494 |
| **Total** | **14.438** |

**Origem:** O dataset é público, contém abstracts de artigos biomédicos rotulados com 5 condições médicas. O script `data/download_medical_abstracts.py` baixa os arquivos `medical_tc_train.csv` e `medical_tc_test.csv` do repositório oficial, combina train+test, mapeia os labels numéricos (1-5) para nomes legíveis, e salva como `data/laudos.csv` com colunas `texto` e `label`.

**Nota sobre desbalanceamento:** A distribuição não é balanceada (a classe majoritária tem 3.2x mais amostras que a minoritária).

**Como trocar/atualizar:** Basta rodar `make model` que executa o script de download, treina e exporta para ONNX. Nenhuma outra alteração é necessária.

---

## 10. Estrutura do repositório

```
.
├── .github/workflows/ci.yml          pipeline de CI (lint -> test -> build)
├── airflow/dags/                     DAG de retreino (load -> train -> export -> validate)
├── configs/
│   └── config.yaml                   configuração reprodutível do pipeline
├── data/
│   ├── download_medical_abstracts.py download do Medical Abstracts TC Corpus
│   └── laudos.csv                    14.438 abstracts, 5 classes
├── docker/
│   ├── Dockerfile                    imagem da API (python:3.11-slim)
│   ├── docker-compose.yml            API + Prometheus + Grafana + Tempo + Loki
│   ├── docker-compose.airflow.yml    Airflow standalone (só para demonstrar a DAG)
│   ├── prometheus.yml                scrape de api:8000/metrics
│   ├── tempo.yml / loki-config.yml   backends de traces e logs
│   └── grafana/                      datasources e os 3 dashboards provisionados
├── docs/                             benchmark e prints (dashboards e DAG)
├── models/                           artefatos: .joblib, .onnx, vectorizer, classes.json
├── scripts/populate_dashboards.py    gerador de tráfego para popular os painéis
├── src/
│   ├── app/                          API FastAPI: main, model_loader, schemas, telemetry
│   ├── triage/
│   │   ├── config.py                 configuração central (Pydantic Settings + YAML)
│   │   └── tracking.py               integração MLflow/DagsHub (tracking + registry)
│   ├── train.py                      treino do pipeline TF-IDF + RandomForest + MLflow
│   ├── export_onnx.py                conversão do classificador para ONNX
│   ├── benchmark.py                  latência do classificador isolado
│   └── benchmark_http.py             latência HTTP end-to-end
├── tests/                            pytest: API, artefatos, métricas, telemetria
├── Makefile                          todos os atalhos (make help)
├── .env.example                      exemplo de variáveis de ambiente (DagsHub token)
└── requirements*.txt                 dependências (a da API é enxuta, sem treino)
```

`requirements-api.txt` é intencionalmente menor que `requirements.txt`: a imagem da API não precisa
de `pandas`, `skl2onnx` nem das ferramentas de treino, apenas do necessário para servir.

---

## 11. MLflow & DagsHub Model Registry

O pipeline integra **MLflow** para experiment tracking e **DagsHub** como backend remoto
(Model Registry, artifact store, UI de comparação de runs).

### Como funciona

- **Com credenciais DagsHub** (`DAGSHUB_TOKEN` no `.env`): tracking remoto em
  `https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-3.mlflow`, modelos registrados
  no Model Registry com aliases `staging`/`production`, promoção automática baseada em `f1_macro`.
- **Sem credenciais** (CI, desenvolvimento local): fallback para SQLite local
  (`/tmp/mlflow_local/mlflow.db`), registra modelo localmente, **não promove** para production.

### Configuração

```bash
cp .env.example .env
# Edite .env com suas credenciais DagsHub
```

### Modelo registrado

- **Nome**: `MedicalAbstractsClassifier`
- **Flavor**: `pyfunc` (wrapper `ClassifierPyfunc` expõe `Pipeline.predict` + `predict_proba`)
- **Signature**: inferida automaticamente (input: `DataFrame[texto]`, output: `DataFrame[prediction, probabilities]`)
- **Aliases**: `staging` (toda versão), `production` (só se `f1_macro` > produção atual)

### Promoção manual (se necessário)

```bash
# Com token configurado
PYTHONPATH=src .venv/bin/python -c "
from triage.tracking import promote_to_production
promote_to_production('MedicalAbstractsClassifier', 0.35, 'f1_macro')
"
```

---

## 12. Vídeo STAR

Link: _a preencher_

Roteiro (formato STAR, até 5 minutos): a situação da triagem manual, a tarefa de colocar o modelo
em produção, as ações demonstradas ao vivo (API, dashboards, CI verde, DAG e a tabela de latência)
e o resultado medido.
