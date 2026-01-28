# **Archivo: esquemas.py**

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any, Dict

# --- El Esquema de Salida del LLM (SOLO PARA TAREAS SUBJETIVAS) ---

class LLMEvaluationOutput(BaseModel):
    """
    Salida estructurada para la evaluación subjetiva.
    El feedback al alumno YA NO lo genera el LLM, se selecciona del JSON.
    """
    score: float = Field(
        description="La puntuación numérica final de 0 a 10."
    )
    
    reasoning: str = Field(
        description="La justificación técnica de la nota. Explica qué criterios de la rúbrica se cumplieron y cuáles no."
    )
    
    # Eliminamos 'student_feedback' porque ahora es estático en el JSON

# --- Esquemas del Estado del Agente ---

# Definimos los niveles posibles
StrategyLevel = Literal["nivel_1", "nivel_2", "nivel_3"]

class TaskEvaluationHistory(BaseModel):
    """Un registro de una sola tarea completada."""
    task_id: str
    strategy: str
    sub_strategy: str
    student_response: Any
    score: float
    score_weighted: float
    feedback_shown: str # El mensaje exacto que se le mostró al alumno
    reasoning: str # La razón interna (del LLM o Python)

class StudentGraphState(BaseModel):
    """
    El estado completo de nuestro grafo.
    """
    student_id: str
    
    # Perfil de competencia por estrategia (Nivel actual en cada una)
    competency_levels: Dict[str, StrategyLevel] = {
        "navegar": "nivel_3", # Valor por defecto inicial
        "evaluar": "nivel_3",
        "integrar": "nivel_3"
    }
    
    task_history: List[TaskEvaluationHistory] = Field(default_factory=list)
    remaining_tasks: List[dict] = Field(default_factory=list)
    
    # Campos temporales
    current_task: Optional[dict] = None
    current_student_response: Any = None
    last_evaluation: Optional[dict] = None
    
    current_subject: Optional[str] = None
    current_session: Optional[str] = None

    current_task_prompt: Optional[str] = None

    last_feedback_message: Optional[str] = None

    last_parent_id: Optional[str] = None

    # --- NUEVOS CAMPOS PARA EL TEST INICIAL ---
    in_initial_assessment: bool = False       # ¿Estamos haciendo el test?
    ask_assessment_permission: bool = True    # ¿Falta preguntar si quiere hacerlo?
    assessment_questions: List[Dict[str, Any]] = Field(default_factory=list) # Preguntas cargadas
    assessment_index: int = 0                 # Índice de la pregunta actual
    # Guardamos aciertos (1) o fallos (0) por cada estrategia
    assessment_scores: Dict[str, List[int]] = Field(default_factory=lambda: {"navegar": [], "evaluar": [], "integrar": []})