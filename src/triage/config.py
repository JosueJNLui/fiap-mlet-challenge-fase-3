"""Configuração central via Pydantic Settings (configs/config.yaml sobreposto por .env).

Fonte única de verdade para caminhos e hiperparâmetros. O YAML guarda os valores
reprodutíveis; o ``.env`` sobrepõe valores de ambiente/secret (credenciais DagsHub, URIs).
Precedência: init > env > .env > YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Paths(BaseModel):
    """Diretórios de dados e artefatos."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("models")


class TrainCfg(BaseModel):
    """Hiperparâmetros de treino."""

    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 10
    rf_class_weight: str | None = "balanced"
    rf_random_state: int = 42
    test_size: float = 0.2
    random_state: int = 42


class MlflowCfg(BaseModel):
    """Configuração de tracking do MLflow."""

    experiment_name: str = "Medical-Abstracts-Classification"


class Settings(BaseSettings):
    """Configuração do pipeline: YAML como base, ``.env``/ambiente sobrepõem."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file="configs/config.yaml",
        extra="ignore",
    )

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    train: TrainCfg = Field(default_factory=TrainCfg)
    mlflow: MlflowCfg = Field(default_factory=MlflowCfg)

    # Ambiente/secret — sem default no YAML, vêm do .env
    dagshub_repo_owner: str = "JosueJNLui"
    dagshub_repo_name: str = "fiap-mlet-challenge-fase-3"
    dagshub_user: str | None = None
    dagshub_token: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Adiciona o YAML como fonte de menor precedência."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def load_settings() -> Settings:
    """Carrega (e memoiza) a configuração do pipeline."""
    return Settings()
