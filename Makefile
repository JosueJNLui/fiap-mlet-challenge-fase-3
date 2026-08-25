# Atalhos de desenvolvimento. PY aponta para o venv local por padrão;
# no CI (que instala as dependências no próprio runner) use: make check PY=python
PY ?= .venv/bin/python
COMPOSE = docker compose -f docker/docker-compose.yml

.PHONY: setup model lint test check bench up down clean

setup:  ## cria o venv (Python 3.11, igual ao Dockerfile e ao CI) e instala tudo
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-api.txt \
		pandas==2.2.2 skl2onnx==1.20.0 onnx==1.22.0 \
		pytest==8.3.3 httpx==0.27.2 flake8

model:  ## gera o dataset, treina e exporta para ONNX (os testes dependem dos artefatos)
	$(PY) data/generate_data.py
	$(PY) src/train.py
	$(PY) src/export_onnx.py

lint:   ## flake8 (regras em .flake8, as mesmas do CI)
	$(PY) -m flake8 src tests

test:   ## pytest
	$(PY) -m pytest tests/ -v

check: lint test  ## o mesmo que o CI roda antes do build

bench:  ## latência do classificador (sklearn vs ONNX) e da API HTTP (precisa de `make up`)
	$(PY) src/benchmark.py --n 500
	$(PY) src/benchmark_http.py --n 200

up:     ## sobe API + Prometheus + Grafana (8000 / 9090 / 3000)
	$(COMPOSE) up -d --build

down:   ## derruba a stack
	$(COMPOSE) down

clean:  ## remove caches do Python e do pytest
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
