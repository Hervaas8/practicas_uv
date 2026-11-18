from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import math


def get_strategies(task) -> List[str]:
    """Lee las competencias, también en historiales con el formato antiguo."""
    values = task if isinstance(task, dict) else task.model_dump()
    strategies = values.get("strategies") or values.get("strategy", "")
    if isinstance(strategies, str):
        strategies = strategies.split("-")
    aliases = {"navegacion": "navegar", "integracion": "integrar", "evaluacion": "evaluar"}
    result = []
    for strategy in strategies:
        strategy = aliases.get(strategy, strategy)
        if strategy in ("navegar", "integrar", "evaluar") and strategy not in result:
            result.append(strategy)
    return result


def validate_true_false_task(task: dict) -> list:
    """Comprueba las afirmaciones y sus respuestas para una tarea de verdadero/falso."""
    statements = task.get("afirmaciones")
    if not isinstance(statements, list) or not statements:
        raise ValueError("La tarea de verdadero/falso debe tener afirmaciones.")
    for statement in statements:
        if (not isinstance(statement, dict)
                or not isinstance(statement.get("texto"), str)
                or not statement["texto"].strip()
                or not isinstance(statement.get("correcta"), bool)):
            raise ValueError("Cada afirmación debe tener texto y una respuesta correcta booleana.")
    return statements


def normalized_score(entry) -> float:
    """Normaliza puntuaciones y descarta máximos inválidos de datos antiguos."""
    values = entry if isinstance(entry, dict) else entry.model_dump()
    score = float(values.get("score", 0))
    maximum = float(values.get("max_score", 1))
    if not math.isfinite(score) or not math.isfinite(maximum) or maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, score / maximum))


def aggregate_score(history) -> float:
    """Calcula el acierto global dando el mismo peso a cada punto evaluable."""
    points = maximum = 0.0
    for entry in history:
        values = entry if isinstance(entry, dict) else entry.model_dump()
        task_maximum = float(values.get("max_score", 1))
        if math.isfinite(task_maximum) and task_maximum > 0:
            points += normalized_score(values) * task_maximum
            maximum += task_maximum
    return points / maximum if maximum else 0.0


def normalize_rubric(rubric: dict) -> dict:
    """Convierte rúbricas anteriores al contrato explícito de ítems e instrucciones."""
    if not isinstance(rubric, dict) or not rubric:
        raise ValueError("La tarea abierta debe tener una rúbrica con apartados evaluables.")
    instructions = rubric.get("instrucciones", rubric.get("Instrucciones", ""))
    if "items" in rubric:
        items = rubric["items"]
    else:
        criteria = {key: value for key, value in rubric.items()
                    if key not in ("Instrucciones", "instrucciones", "feedback")}
        items = {"respuesta": criteria} if criteria and all(
            isinstance(value, str) for value in criteria.values()
        ) else criteria
    if not isinstance(items, dict) or not items:
        raise ValueError("La rúbrica no contiene apartados evaluables.")
    for name, criteria in items.items():
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"El apartado '{name}' no tiene criterios válidos.")
        for value, description in criteria.items():
            try:
                points = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Puntuación inválida en el apartado '{name}'.") from exc
            if not math.isfinite(points) or points < 0 or not isinstance(description, str):
                raise ValueError(f"Criterio inválido en el apartado '{name}'.")
        if max(float(value) for value in criteria) <= 0:
            raise ValueError(f"El apartado '{name}' debe tener un máximo positivo.")
    return {"items": items, "instrucciones": instructions}

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
    strategies: List[str] = Field(default_factory=list)
    sub_strategy: Optional[str] = None
    student_response: Any
    score: float
    score_weighted: float  # Alias de score conservado para historiales anteriores.
    max_score: float = 1.0  
    normalized_score: float = 0.0
    scores_por_item: Dict[str, float] = Field(default_factory=dict)
    feedback_shown: str
    reasoning: str
    is_test_task: bool = False 
    time_taken: float = 0.0  
    evaluation_time: float = 0.0
    support_level: str = "nivel_1"
    navigation_history: List[str] = Field(default_factory=list)
    contenido: bool = False
    conceptos: List[str] = []
    current_levels: Dict[str, str] = {} 

class StudentGraphState(BaseModel):
    student_id: str
    current_subject: Optional[str] = None
    current_scenario: Optional[str] = None
    current_session: Optional[str] = None
    test_mode: bool = False 
    remaining_tasks: List[dict] = [] 
    total_tasks: int = 0  
    current_task: Optional[dict] = None
    current_task_prompt: Optional[str] = None
    current_student_response: Optional[Any] = None
    current_response_time: float = 0.0  
    current_evaluation_time: float = 0.0
    current_navigation_history: List[str] = Field(default_factory=list)
    competency_levels: Dict[str, str] = {} 
    detailed_feedback: Dict[str, str] = {} 
    task_history: List[TaskEvaluationHistory] = []
    last_feedback_message: Optional[str] = None
    last_evaluation: Optional[dict] = None
