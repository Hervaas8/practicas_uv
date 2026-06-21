import json
import sys
from typing import List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from esquemas import StudentGraphState, TaskEvaluationHistory
from motor_evaluacion import evaluate_task

def get_resolved_prompt(task_data: dict, competency_levels: dict) -> str:
    """
    Resuelve el enunciado basándose en el nivel del alumno para esa estrategia.
    Si la estrategia es 'contenido' (o tiene la flag contenido=True), fuerza siempre el nivel medio.
    """
    prompt_data = task_data.get("task_prompt", "")
    strat = task_data.get("strategy", "general")
    is_contenido = task_data.get("contenido", False)
    
    # Condición para forzar el nivel medio si es de contenido
    if strat == "contenido" or is_contenido:
        level_val = "nivel_2"
    else:
        # Obtenemos el nivel del alumno en esta estrategia concreta (por defecto nivel_1)
        level_val = competency_levels.get(strat, "nivel_1")
        
    if isinstance(prompt_data, dict):
        nivel_map = {"nivel_1": "nivel_bajo", "nivel_2": "nivel_medio", "nivel_3": "nivel_alto"}
        return prompt_data.get(nivel_map.get(level_val, "nivel_bajo"), str(prompt_data))
    
    return str(prompt_data)

def load_test_inicial():
    try:
        with open("test_inicial.json", "r", encoding="utf-8-sig") as f:
            raw_tasks = json.load(f)
        adapted_tasks = []
        for t in raw_tasks:
            # Le pasamos la tarea completa y un diccionario de niveles vacío. 
            # Automáticamente asignará nivel_2 si es contenido, o nivel_1 en el resto.
            prompt_str = get_resolved_prompt(t, {})
            adapted_tasks.append({
                "task_id": t.get("id"),
                "task_prompt": prompt_str,
                "task_image": t.get("task_image", []),
                "task_type": t.get("task_type", "objetivo"),
                "correct_answer": t.get("correct_answer"),
                "strategy": t.get("strategy", "general"),
                "sub_strategy": None,
                "description": "Test Inicial",
                "pantallas": t.get("pantallas", {})
            })
        return adapted_tasks
    except Exception:
        return []

def load_curriculum(session_id="sesion1"):
    try:
        with open("tareas2.json", "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            return data.get("ciencias", {}).get(session_id, [])
    except Exception:
        return []

def calculate_levels_from_test(history: List[TaskEvaluationHistory]) -> Dict[str, str]:
    scores_by_strat = {}
    for h in history:
        if h.is_test_task:
            if h.strategy not in scores_by_strat:
                scores_by_strat[h.strategy] = []
            # Normalizamos al % de acierto del test
            scores_by_strat[h.strategy].append(h.score / h.max_score if h.max_score > 0 else 0)
    
    new_levels = {}
    for strat in ["navegar", "integrar", "evaluar"]:
        scores = scores_by_strat.get(strat, [])
        if not scores:
            new_levels[strat] = "nivel_1"
        else:
            avg = sum(scores) / len(scores)
            if avg < 0.50: new_levels[strat] = "nivel_1"
            elif avg < 0.80: new_levels[strat] = "nivel_2"
            else: new_levels[strat] = "nivel_3"
    return new_levels

def calculate_dynamic_level(current_strategy: str, current_level: str, current_score: float, current_max_score: float, history: List[TaskEvaluationHistory]) -> str:
    relevant_history = [h for h in history if getattr(h, 'strategy', '') == current_strategy and not getattr(h, 'is_test_task', False)]
    
    huecos_acumulados = current_max_score
    hits = current_score
    
    for h in reversed(relevant_history):
        if huecos_acumulados >= 3.0:
            break 
        
        # OPCIÓN B: Las tareas con varios ítems (max_score > 1) llenan más rápido la ventana de evaluación.
        h_max = getattr(h, 'max_score', 1.0)
        huecos_acumulados += h_max
        hits += getattr(h, 'score_weighted', 0.0)
        
    if huecos_acumulados < 3.0:
        return current_level

    hits_normalizados = (hits / huecos_acumulados) * 3.0

    if current_level == "nivel_1":
        if hits_normalizados >= 2.0: return "nivel_2"
        else: return "nivel_1"
    elif current_level == "nivel_2":
        if hits_normalizados >= 3.0: return "nivel_3"
        elif hits_normalizados <= 1.0: return "nivel_1"
        else: return "nivel_2"
    elif current_level == "nivel_3":
        if hits_normalizados >= 2.5: return "nivel_3"
        else: return "nivel_2"

    return current_level

# --- NODOS DEL GRAFO ---
def node_load_curriculum(state: StudentGraphState): return {} 

def node_select_task(state: StudentGraphState):
    if state.remaining_tasks:
        next_task = state.remaining_tasks[0]
        # Delegamos toda la lógica de selección de nivel a la nueva función
        resolved_prompt = get_resolved_prompt(next_task, state.competency_levels)
        return {"current_task": next_task, "current_task_prompt": resolved_prompt}
    
    if state.test_mode:
        new_levels = calculate_levels_from_test(state.task_history)
        main_tasks = load_curriculum(state.current_session or "sesion1")
        msg = "Test inicial completado. Hemos adaptado el curso a tu nivel."
        first_task = main_tasks[0] if main_tasks else None
        
        # Igual aquí, utilizamos la función actualizada
        first_prompt = get_resolved_prompt(first_task, new_levels) if first_task else ""

        return {"test_mode": False, "competency_levels": new_levels, "remaining_tasks": main_tasks, "total_tasks": len(main_tasks), "current_task": first_task, "current_task_prompt": first_prompt, "last_feedback_message": msg}

    return {"current_task": None}

def node_evaluate(state: StudentGraphState):
    current_task = state.current_task
    if not current_task: return {}

    strat = current_task.get("strategy", "general")
    current_level_val = state.competency_levels.get(strat, "nivel_1")
    
    # Se llama al evaluate original (usará el modelo por defecto)
    res = evaluate_task(current_task, state.current_student_response, current_level_val)
    if not res: return {"last_feedback_message": "Error técnico en la evaluación."}

    is_test = state.test_mode
    new_levels = state.competency_levels.copy()
    
    if not is_test and strat != "contenido":
        new_level = calculate_dynamic_level(strat, current_level_val, res["score"], res["max_score"], state.task_history)
        new_levels[strat] = new_level
    
    history_entry = TaskEvaluationHistory(
        task_id=current_task.get("task_id"),
        task_prompt=state.current_task_prompt,
        strategy=strat,
        sub_strategy=current_task.get("sub_strategy", ""),
        student_response=str(state.current_student_response),
        score=res["score"], 
        score_weighted=res["score"], 
        max_score=res["max_score"], 
        feedback_shown=res["feedback_shown"] if not is_test else "Respuesta registrada.",
        reasoning=res["reasoning"],
        is_test_task=is_test,
        contenido=current_task.get("contenido", False),
        conceptos=current_task.get("conceptos", []),
        current_levels=new_levels.copy()
    )

    return {
        "remaining_tasks": state.remaining_tasks[1:], 
        "task_history": state.task_history + [history_entry],
        "current_task": None,
        "competency_levels": new_levels,
        "last_feedback_message": res["feedback_shown"] if not is_test else "Respuesta registrada."
    }

builder = StateGraph(StudentGraphState)
builder.add_node("load_curriculum", node_load_curriculum) 
builder.add_node("select_task", node_select_task)
builder.add_node("evaluate", node_evaluate)
builder.set_entry_point("load_curriculum")
builder.add_edge("load_curriculum", "select_task")
builder.add_conditional_edges("select_task", lambda s: "evaluate" if s.current_task else "end", {"evaluate": "evaluate", "end": END})
builder.add_edge("evaluate", "select_task")

memory = MemorySaver()
app = builder.compile(checkpointer=memory, interrupt_before=["evaluate"])