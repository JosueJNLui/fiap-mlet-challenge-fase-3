FROM python:3.11-slim

# Locale completo para evitar problemas com operadores ONNX que dependem de
# locale (ex.: StringNormalizer). Ver nota em src/export_onnx.py.
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    MODEL_BACKEND=onnx

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/app ./app
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
