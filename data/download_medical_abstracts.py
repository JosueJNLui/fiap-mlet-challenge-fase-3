#!/usr/bin/env python3
"""
Baixa e processa o dataset Medical Abstracts TC Corpus.

Dataset: https://github.com/sebischair/Medical-Abstracts-TC-Corpus
Classes:
    1 -> neoplasms
    2 -> digestive system diseases
    3 -> nervous system diseases
    4 -> cardiovascular diseases
    5 -> general pathological conditions

Total: ~14.438 amostras (11.550 treino + 2.888 teste)

Gera um CSV unificado em data/laudos.csv com colunas: texto, label
Onde label é o condition_name (ex: 'neoplasms', 'digestive system diseases', etc.)
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
OUTPUT_CSV = DATA_DIR / "laudos.csv"

LABEL_MAP = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

TRAIN_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv"
TEST_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv"


def download_csv(url: str) -> pd.DataFrame:
    """Baixa um CSV do GitHub e retorna como DataFrame."""
    print(f"Baixando {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Parse CSV content
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    return df


def main():
    print("=== Baixando Medical Abstracts TC Corpus ===")
    
    # Baixar treino e teste
    train_df = download_csv(TRAIN_URL)
    test_df = download_csv(TEST_URL)
    
    print(f"Treino: {len(train_df)} amostras")
    print(f"Teste: {len(test_df)} amostras")
    
    # Unificar
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Total: {len(df)} amostras")
    
    # Verificar colunas esperadas
    expected_cols = {"condition_label", "medical_abstract"}
    if not expected_cols.issubset(df.columns):
        print(f"ERRO: Colunas esperadas {expected_cols}, encontradas {set(df.columns)}")
        sys.exit(1)
    
    # Mapear labels numéricos para nomes
    df["label"] = df["condition_label"].map(LABEL_MAP)  # type: ignore[arg-type]
    
    # Verificar se todos foram mapeados
    if df["label"].isna().any():  # type: ignore[attr-defined]
        print("ERRO: Alguns labels não foram mapeados")
        missing = df.loc[df["label"].isna(), "condition_label"].unique()  # type: ignore[attr-defined]
        print(missing)
        sys.exit(1)
    
    # Renomear coluna de texto
    df = df.rename(columns={"medical_abstract": "texto"})
    
    # Selecionar apenas as colunas necessárias
    df = df[["texto", "label"]]
    
    # Remover nulos
    df = df.dropna(subset=["texto", "label"])  # type: ignore[arg-type]
    
    # Estatísticas por classe
    print("\nDistribuição por classe:")
    for label, count in df["label"].value_counts().items():
        print(f"  {label}: {count}")
    
    # Salvar
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
    print(f"\nDataset salvo em {OUTPUT_CSV} com {len(df)} amostras.")


if __name__ == "__main__":
    main()