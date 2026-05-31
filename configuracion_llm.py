from langchain_openai import ChatOpenAI
from esquemas import LLMEvaluationOutput, LLMPlagiarismOutput, LLMFeedbackEstrategia

POLIGPT_API_BASE = "https://api.poligpt.upv.es"
POLIGPT_API_KEY = "[CLAVE_ELIMINADA]"

# --- MAPAS DE ENRUTAMIENTO ---
MAPA_MODELOS = {
    "evaluacion": "poligpt",
    "feedback_estrategia": "poligpt",
    "plagio": "poligpt",
    "general": "poligpt"
}

MAPA_ESQUEMAS = {
    "evaluacion": LLMEvaluationOutput,
    "feedback_estrategia": LLMFeedbackEstrategia,
    "plagio": LLMPlagiarismOutput
}

# --- FUNCIONES LLM ---

def get_llm(tipo_tarea="general"):
    nombre_modelo = MAPA_MODELOS.get(tipo_tarea, MAPA_MODELOS["general"])
    return ChatOpenAI(
        model=nombre_modelo,
        api_key=POLIGPT_API_KEY,
        base_url=POLIGPT_API_BASE
    )

def get_llm_estructurado(tipo_tarea):
    llm = get_llm(tipo_tarea)
    esquema = MAPA_ESQUEMAS.get(tipo_tarea)
    if esquema is None:
        raise ValueError(f"No hay esquema para la tarea: {tipo_tarea}")
    return llm.with_structured_output(esquema)

def get_llm_evaluador():
    return get_llm_estructurado(tipo_tarea="evaluacion")