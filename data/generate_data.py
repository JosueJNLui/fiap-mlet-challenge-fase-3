"""
Gera um dataset sintético de laudos médicos para triagem (normal / atencao / urgente).

Este script existe para permitir rodar o projeto de ponta a ponta sem depender
de download externo (Kaggle etc). Para usar um dataset real, basta substituir
o arquivo gerado em data/laudos.csv por um CSV com as colunas:
    texto  -> texto do laudo/sintomas
    label  -> normal | atencao | urgente

Datasets sugeridos no enunciado: Medical Abstracts TC Corpus (Kaggle), MIMIC-III.
"""
import csv
import random
from pathlib import Path

random.seed(42)

NORMAL_TEMPLATES = [
    "Paciente relata check-up de rotina, sem queixas relevantes.",
    "Exame de sangue dentro dos parâmetros normais, sem alterações significativas.",
    "Consulta de retorno, paciente estável, sem sintomas novos.",
    "Raio-x de tórax sem alterações, paciente assintomático.",
    "Resultado de exame de rotina normal, sem necessidade de intervenção.",
    "Paciente comparece para avaliação preventiva anual, tudo normal.",
    "Hemograma completo dentro da normalidade, sem sinais de infecção.",
    "Paciente relata bem-estar geral, sinais vitais normais.",
    "Exame oftalmológico de rotina sem achados relevantes.",
    "Vacinação de rotina realizada sem intercorrências.",
]

ATENCAO_TEMPLATES = [
    "Paciente com febre baixa persistente há 3 dias, sem outros sintomas graves.",
    "Queixa de dor abdominal leve a moderada, sem sinais de alarme.",
    "Pressão arterial levemente elevada, recomendado acompanhamento.",
    "Paciente relata tosse persistente há uma semana, sem falta de ar.",
    "Exame indica leve alteração em enzimas hepáticas, necessita reavaliação.",
    "Dor lombar moderada, sem sinais neurológicos associados.",
    "Paciente com histórico de alergia relata leve reação cutânea.",
    "Glicemia de jejum levemente alterada, indicado acompanhamento nutricional.",
    "Paciente relata cansaço e mal-estar leve nos últimos dias.",
    "Pequena ferida com sinais iniciais de inflamação, sem pus.",
]

URGENTE_TEMPLATES = [
    "Paciente apresenta dor torácica intensa e falta de ar súbita.",
    "Sinais de acidente vascular cerebral: perda de força e fala arrastada.",
    "Hemorragia intensa, paciente com sinais de choque hipovolêmico.",
    "Trauma craniano grave após acidente, paciente inconsciente.",
    "Paciente com febre alta e rigidez de nuca, suspeita de meningite.",
    "Dor abdominal súbita e intensa com abdome em tábua, suspeita de abdome agudo.",
    "Reação alérgica grave com edema de glote e dificuldade respiratória.",
    "Paciente com convulsões recorrentes e perda de consciência.",
    "Sinais vitais instáveis, saturação de oxigênio criticamente baixa.",
    "Politraumatismo grave após queda de altura, necessita intervenção imediata.",
]

SUFFIXES = [
    "", " Paciente do sexo feminino, 45 anos.", " Paciente do sexo masculino, 62 anos.",
    " Encaminhado pelo pronto atendimento.", " Histórico de comorbidades relatado.",
    " Sem histórico médico relevante.", " Acompanhado por familiar.",
]

def build_rows(templates, label, n):
    rows = []
    for _ in range(n):
        text = random.choice(templates) + random.choice(SUFFIXES)
        rows.append((text, label))
    return rows

def main():
    n_per_class = 700  # 700*3 = 2100 amostras, acima do mínimo de 2000 exigido
    rows = []
    rows += build_rows(NORMAL_TEMPLATES, "normal", n_per_class)
    rows += build_rows(ATENCAO_TEMPLATES, "atencao", n_per_class)
    rows += build_rows(URGENTE_TEMPLATES, "urgente", n_per_class)
    random.shuffle(rows)

    out_path = Path(__file__).parent / "laudos.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["texto", "label"])
        writer.writerows(rows)

    print(f"Dataset gerado em {out_path} com {len(rows)} amostras.")

if __name__ == "__main__":
    main()
