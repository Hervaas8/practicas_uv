# **Archivo: esquemas.py**

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

# --- El Esquema de Salida del LLM ---

class LLMEvaluationOutput(BaseModel):
    """
    La salida estructurada estándar para CUALQUIER tarea evaluada por el LLM.
    """
    score: float = Field(
        description="La puntuación numérica final de 0 a 10."
    )
    
    reasoning_for_professor: str = Field(
        description="La justificación técnica y detallada de la nota para el profesor, "
                    "explicando cómo la respuesta se compara con el contexto y la rúbrica."
    )
    
    student_feedback: str = Field(
        description="El feedback completo, redactado, amigable y constructivo "
                    "para el alumno. Este texto debe ser la salida final, "
                    "explicando los aciertos y fallos basándose en los criterios de la rúbrica."
    )

# --- Esquemas del Estado del Agente (los usaremos después) ---

StudentGroup = Literal["nivel_1", "nivel_2", "nivel_3", "no_asignado"]

class TaskEvaluationHistory(BaseModel):
    """Un registro de una sola tarea completada."""
    task_id: str
    student_response: Any # La respuesta puede ser string, lista, etc.
    evaluation: LLMEvaluationOutput # Usamos el nuevo esquema de salida
    previous_group: StudentGroup
    new_group: StudentGroup

class StudentGraphState(BaseModel):
    """
    El estado completo de nuestro grafo, persistido por alumno.
    """
    student_id: str
    current_group: StudentGroup = "no_asignado"
    task_history: List[TaskEvaluationHistory] = Field(default_factory=list)
    remaining_tasks: List[dict] = Field(default_factory=list)
    
    # --- Campos temporales (usados durante un solo ciclo) ---
    current_task: dict = {}
    current_student_response: Any = None
    current_rag_context: str = ""
    last_evaluation: Optional[LLMEvaluationOutput] = None
    
    # --- Metadatos para el RAG ---
    current_subject: str
    current_session: str