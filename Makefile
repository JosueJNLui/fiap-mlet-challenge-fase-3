# Atalhos de desenvolvimento. PY aponta para o venv local por padrão;
# no CI (que instala as dependências no próprio runner) use: make ci-install PY=python
PY ?= .venv/bin/python
IMAGE_TAG ?= latest
COMPOSE = docker compose -f docker/docker-compose.yml
HADOLINT = docker run --rm -v $(CURDIR):/app -w /app hadolint/hadolint:latest-alpine hadolint
DCLINT = docker run --rm -v $(CURDIR):/app -w /app zavoloklom/dclint:latest-alpine

.DEFAULT_GOAL := help

.PHONY: help setup ci-install model lint lint-python lint-docker lint-compose lint-ty test check bench build up down clean

help:  ## mostra esta ajuda (alvos disponíveis)
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

setup:  ## cria o venv (Python 3.11, igual ao Dockerfile e ao CI) e instala tudo
	uv tool install ty@latest
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-api.txt \
		pandas==2.2.2 skl2onnx==1.20.0 onnx==1.22.0 \
		pytest==8.3.3 httpx==0.27.2 flake8 ty==0.0.75

ci-install:  ## instala as dependências usadas pelos alvos do CI
	$(PY) -m pip install -r requirements-api.txt pandas==2.2.2 \
		skl2onnx==1.20.0 onnx==1.22.0 pytest==8.3.3 httpx==0.27.2 flake8 ty==0.0.75

model:  ## gera o dataset, treina e exporta para ONNX (os testes dependem dos artefatos)
	$(PY) data/generate_data.py
	$(PY) src/train.py
	$(PY) src/export_onnx.py

lint-python:  ## flake8 (regras em .flake8, as mesmas do CI)
	$(PY) -m flake8 src tests

lint-docker:  ## hadolint no docker/Dockerfile (roda via Docker; exige o daemon)
	$(HADOLINT) docker/Dockerfile

lint-compose:  ## DCLint no docker/docker-compose.yml (roda via Docker; exige o daemon)
	$(DCLINT) docker/docker-compose.yml

lint-ty:  ## ty: validação estática das anotações de tipo (config em pyproject.toml)
	$(PY) -m ty check

lint: lint-python lint-docker lint-compose lint-ty  ## todos os lints (Python, Dockerfile, docker-compose e ty)

test:   ## pytest
	$(PY) -m pytest tests/ -v

check: lint test  ## o mesmo que o CI roda antes do build

bench:  ## latência do classificador (sklearn vs ONNX) e da API HTTP (precisa de `make up`)
	$(PY) src/benchmark.py --n 500
	$(PY) src/benchmark_http.py --n 200

build:  ## gera a imagem Docker sem publicar
	docker build -f docker/Dockerfile -t triagem-laudos-api:$(IMAGE_TAG) .

up:     ## sobe API + Prometheus + Grafana (8000 / 9090 / 3000)
	$(COMPOSE) up -d --build

down:   ## derruba a stack
	$(COMPOSE) down

clean:  ## remove caches do Python e do pytest
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
