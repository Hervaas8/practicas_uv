from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# ==========================================
# --- ESQUEMAS DE SALIDA DEL LLM ---
# ==========================================

class LLMEvaluationOutput(BaseModel):
    """Estructura forzada para la respuesta del LLM evaluador de tareas."""
    score: float = Field(description="Nota numérica global sumando todos los ítems.")
    scores_por_item: Dict[str, float] = Field(
        description="Diccionario con la nota desglosada por cada ítem evaluado en la rúbrica. Clave: Nombre del ítem. Valor: Nota."
    )
    reasoning: str = Field(description="Explicación técnica de la nota basada en la rúbrica")

class LLMPlagiarismOutput(BaseModel):
    """Estructura forzada para el LLM detective de plagio."""
    veredicto: Literal["Posible Plagio", "Dudoso", "Posible No Plagio"] = Field(
        description="Clasificación estricta del nivel de plagio detectado."
    )
    justificacion: str = Field(
        description="Análisis estilístico breve explicando por qué es copia o coincidencia natural."
    )
    frases_identicas: List[str] = Field(
        description="Lista de bloques de palabras exactos que comparten ambas respuestas."
    )

class LLMFeedbackEstrategia(BaseModel):
    """Estructura forzada de UN SOLO CAMPO para el feedback final por estrategia."""
    texto_feedback: str = Field(
        description="Párrafo cualitativo completo con las consideraciones y análisis del alumno para esta estrategia."
    )

class LLMScaffoldingOutput(BaseModel):
    """Estructura forzada para la generación de pistas (scaffolding)."""
    pista_1: str = Field(
        description="Pista de nivel 1 (Formato). Describe de forma general cómo funciona el entorno digital o el formato de la información."
    )
    pista_2: str = Field(
        description="Pista de nivel 2 (Características Específicas). Instrucción más directa y accionable."
    )
    enunciado_nivel_medio: str = Field(
        description="Enunciado base con la pista 1 integrada."
    )
    enunciado_nivel_bajo: str = Field(
        description="Enunciado base con la pista 1 y la pista 2 integradas."
    )

# ==========================================
# --- HISTORIAL Y ESTADO DEL GRAFO ---
# ==========================================

class TaskEvaluationHistory(BaseModel):
    task_id: str
    task_prompt: str = ""
    strategy: str
    sub_strategy: Optional[str] = None
    student_response: Any
    score: float
    score_weighted: float 
    max_score: float = 1.0  
    feedback_shown: str
    reasoning: str
    is_test_task: bool = False 
    time_taken: float = 0.0  
    contenido: bool = False
    conceptos: List[str] = []
    current_levels: Dict[str, str] = {} 

class StudentGraphState(BaseModel):
    student_id: str
    current_subject: Optional[str] = None
    current_session: Optional[str] = None
    test_mode: bool = False 
    remaining_tasks: List[dict] = [] 
    total_tasks: int = 0  
    current_task: Optional[dict] = None
    current_task_prompt: Optional[str] = None
    current_student_response: Optional[Any] = None
    current_response_time: float = 0.0  
    competency_levels: Dict[str, str] = {} 
    detailed_feedback: Dict[str, str] = {} 
    task_history: List[TaskEvaluationHistory] = []
    last_feedback_message: Optional[str] = None
    last_evaluation: Optional[dict] = None