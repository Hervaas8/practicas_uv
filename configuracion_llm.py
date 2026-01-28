# **Archivo: configuracion_llm.py**

from langchain_openai import ChatOpenAI
from esquemas import LLMEvaluationOutput # <-- Importamos el nuevo esquema

# --- Configuración de "poligpt" ---

POLIGPT_API_BASE = "https://api.poligpt.upv.es"
POLIGPT_API_KEY = "[CLAVE_ELIMINADA]"
POLIGPT_MODEL_NAME = "poligpt"

# --- Clientes LLM ---

def get_llm():
    """Retorna el cliente LLM estándar para "poligpt"."""
        
    return ChatOpenAI(
        model=POLIGPT_MODEL_NAME,
        api_key=POLIGPT_API_KEY,
        base_url=POLIGPT_API_BASE
    )

def get_llm_evaluador():
    """
    Retorna el cliente LLM configurado para forzar la salida JSON 
    basada en nuestra NUEVA rúbrica de salida (LLMEvaluationOutput).
    """
    llm = get_llm()
    if llm is None:
        return None

    # Forzamos al LLM a que SIEMPRE devuelva un JSON con la estructura
    # de nuestra clase LLMEvaluationOutput
    return llm.with_structured_output(LLMEvaluationOutput)

