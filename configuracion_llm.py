from langchain_openai import ChatOpenAI
from esquemas import LLMEvaluationOutput, LLMPlagiarismOutput, LLMFeedbackEstrategia, LLMScaffoldingOutput

POLIGPT_API_BASE = "https://api.poligpt.upv.es"
POLIGPT_API_KEY = "[CLAVE_ELIMINADA]"

# --- MODELOS CON CAPACIDAD VISUAL ---
MODELOS_VISION = ["poligpt-code", "qwen3-vl:30b",]

# --- MAPAS DE ENRUTAMIENTO ---
MAPA_MODELOS = {
    "evaluacion": "poligpt-vision",
    "feedback_estrategia": "poligpt",
    "plagio": "poligpt",
    "general": "poligpt",
    "generador": "poligpt",
    "ocr": "llama3.2:3b"  # <-- Nuevo modelo integrado
}

# --- MAPA DE TEMPERATURAS ---
MAPA_TEMPERATURAS = {
    "evaluacion": 0.0,           
    "plagio": 0.0,               
    "general": 0.3,              
    "generador": 0.4,            
    "feedback_estrategia": 0.5,  
    "ocr": 0.0                   # Determinista: Solo debe extraer texto, sin inventar.
}

MAPA_ESQUEMAS = {
    "evaluacion": LLMEvaluationOutput,
    "feedback_estrategia": LLMFeedbackEstrategia,
    "plagio": LLMPlagiarismOutput,
    "generador": LLMScaffoldingOutput
}

def is_vision_model(model_name: str) -> bool:
    """Verifica si un modelo tiene capacidades de visión."""
    return model_name in MODELOS_VISION

# --- FUNCIONES LLM ---
def get_llm(tipo_tarea="general", override_model=None):
    nombre_modelo = override_model if override_model else MAPA_MODELOS.get(tipo_tarea, MAPA_MODELOS["general"])
    
    temperatura_asignada = MAPA_TEMPERATURAS.get(tipo_tarea, 0.3)
    
    return ChatOpenAI(
        model=nombre_modelo,
        api_key=POLIGPT_API_KEY,
        base_url=POLIGPT_API_BASE,
        temperature=temperatura_asignada,
        max_tokens=10000,
    )

def get_llm_estructurado(tipo_tarea, override_model=None):
    llm = get_llm(tipo_tarea, override_model)
    esquema = MAPA_ESQUEMAS.get(tipo_tarea)
    if esquema is None:
        raise ValueError(f"No hay esquema para la tarea: {tipo_tarea}")
    return llm.with_structured_output(esquema)

def get_llm_evaluador(override_model=None):
    return get_llm_estructurado(tipo_tarea="evaluacion", override_model=override_model)