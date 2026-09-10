from pydantic import BaseModel, Field


class LaudoRequest(BaseModel):
    texto: str = Field(..., min_length=3, description="Texto do abstract médico / relato clínico")


class LaudoResponse(BaseModel):
    classificacao: str
    confianca: float
    probabilidades: dict[str, float]
    modelo: str  # "sklearn" ou "onnx"
    latencia_ms: float
