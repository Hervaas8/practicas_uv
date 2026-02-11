from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- ESQUEMA DE SALIDA DEL LLM ---
class LLMEvaluationOutput(BaseModel):
    """Estructura forzada para la respuesta del LLM evaluador."""
    score: float = Field(description="Nota numérica del 0 al 1 (ej: 0.85)")
    reasoning: str = Field(description="Explicación técnica de la nota basada en la rúbrica")

# --- HISTORIAL DE EVALUACIONES ---
class TaskEvaluationHistory(BaseModel):
    """Registro de cada tarea completada por el alumno."""
    task_id: str
    strategy: str
    sub_strategy: Optional[str] = None
    student_response: Any
    score: float
    score_weighted: float # En el nuevo sistema será igual a score al no haber dificultad
    feedback_shown: str
    reasoning: str

# --- ESTADO DEL GRAFO (LangGraph) ---
class StudentGraphState(BaseModel):
    """Estado global que viaja a través de los nodos del agente."""
    student_id: str
    current_subject: Optional[str] = None
    current_session: Optional[str] = None
    
    # Navegación de tareas (ahora plano)
    remaining_tasks: List[dict] = [] # Lista de tareas directamente de tareas2.json
    current_task: Optional[dict] = None
    current_task_prompt: Optional[str] = None
    
    # Respuesta del alumno
    current_student_response: Optional[Any] = None
    
    # Adaptatividad y Progreso
    competency_levels: Dict[str, str] = {} # ej: {"navegar": "nivel_2"}
    task_history: List[TaskEvaluationHistory] = []
    
    # Feedback para la interfaz
    last_feedback_message: Optional[str] = None
    last_evaluation: Optional[dict] = None