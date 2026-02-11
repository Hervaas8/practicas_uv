import json
import sys
from typing import List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- IMPORTACIONES PROPIAS ---
from esquemas import StudentGraphState, TaskEvaluationHistory
from motor_evaluacion import evaluate_task

def load_curriculum():
    """Carga el nuevo formato plano de tareas2.json"""
    try:
        with open("tareas2.json", "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            # Extraemos la lista plana de la sesión
            tasks = data.get("ciencias", {}).get("sesion1", [])
            return tasks
    except Exception as e:
        print(f"Error cargando tareas2.json: {e}")
        return []

def calculate_new_level(current_strategy: str, current_score: float, history: List[TaskEvaluationHistory]) -> str:
    """Calcula el nivel basándose en el historial de aciertos"""
    relevant = [h.score for h in history if h.strategy == current_strategy] + [current_score]
    last_4 = relevant[-4:]
    avg = sum(last_4) / len(last_4) if last_4 else 0
    
    if avg < 0.50: return "nivel_1" 
    elif avg < 0.80: return "nivel_2"
    else: return "nivel_3"

# --- NODOS DEL GRAFO ---

def node_load_curriculum(state: StudentGraphState):
    """Carga las tareas y establece niveles iniciales"""
    if state.remaining_tasks: 
        return {}
    tasks = load_curriculum()
    levels = state.competency_levels or {
        "navegar": "nivel_1", 
        "integrar": "nivel_1", 
        "evaluar": "nivel_1", 
        "contenido": "nivel_1"
    }
    return {
        "remaining_tasks": tasks,
        "competency_levels": levels,
        "current_subject": "ciencias",
        "current_session": "sesion1"
    }

def node_select_task(state: StudentGraphState):
    """Selecciona la siguiente tarea de la lista"""
    if not state.remaining_tasks:
        return {"current_task": None}
    
    # Al ser Pydantic, accedemos directamente a la lista
    next_task = state.remaining_tasks[0]
    prompt = next_task.get("task_prompt", "")
    
    return {
        "current_task": next_task,
        "current_task_prompt": prompt
    }

def node_evaluate(state: StudentGraphState):
    """Evalúa la respuesta usando el motor de evaluación"""
    # Acceso directo a atributos de la clase Pydantic
    current_task = state.current_task
    answer = state.current_student_response
    
    if not current_task: 
        return {}

    # Llamada al motor adaptado a tareas2.json
    level = state.competency_levels.get(current_task.get("strategy"), "nivel_1")
    res = evaluate_task(current_task, answer, level)
    
    if not res: 
        return {"last_feedback_message": "Error técnico en la evaluación."}

    # Actualización de niveles
    strat = current_task.get("strategy", "general")
    new_level = calculate_new_level(strat, res["score"], state.task_history)
    new_levels = state.competency_levels.copy()
    new_levels[strat] = new_level

    # Registro en historial
    history_entry = TaskEvaluationHistory(
        task_id=current_task.get("task_id"),
        strategy=strat,
        sub_strategy=current_task.get("sub_strategy", ""),
        student_response=str(answer),
        score=res["score"],
        score_weighted=res["score"], 
        feedback_shown=res["feedback_shown"],
        reasoning=res["reasoning"]
    )

    return {
        "remaining_tasks": state.remaining_tasks[1:],
        "task_history": state.task_history + [history_entry],
        "current_task": None,
        "competency_levels": new_levels,
        "last_feedback_message": res["feedback_shown"]
    }

# --- DEFINICIÓN DEL GRAFO ---

builder = StateGraph(StudentGraphState)

builder.add_node("load_curriculum", node_load_curriculum)
builder.add_node("select_task", node_select_task)
builder.add_node("evaluate", node_evaluate)

builder.set_entry_point("load_curriculum")

builder.add_edge("load_curriculum", "select_task")

# CORRECCIÓN CLAVE: Uso de atributos en lugar de .get() en la arista condicional
builder.add_conditional_edges(
    "select_task",
    lambda s: "evaluate" if s.current_task else "end",
    {"evaluate": "evaluate", "end": END}
)

builder.add_edge("evaluate", "select_task")

memory = MemorySaver()
app = builder.compile(checkpointer=memory, interrupt_before=["evaluate"])