"""Configuração do webserver do Airflow para a demo local.

Mantém o backend de autenticação padrão (o usuário admin criado pelo
`standalone` continua existindo), mas concede papel Admin a visitantes
anônimos, para que a UI abra sem login, igual ao que o compose já faz com o
Grafana. Só é aceitável porque a stack roda em localhost; em nuvem essa linha
sai.
"""
from __future__ import annotations

from flask_appbuilder.const import AUTH_DB

WTF_CSRF_ENABLED = True
WTF_CSRF_TIME_LIMIT = None

AUTH_TYPE = AUTH_DB
AUTH_ROLE_ADMIN = "Admin"
AUTH_ROLE_PUBLIC = "Admin"
