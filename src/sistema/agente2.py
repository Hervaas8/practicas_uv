import json
import sys
import time
from typing import List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sistema.utils.esquemas import (
    StudentGraphState, TaskEvaluationHistory, get_strategies, normalize_rubric,
    validate_true_false_task,
)
from sistema.modulos.motor_evaluacion import evaluate_task
from sistema.config import INITIAL_PATH, TASKS_PATH

def get_task_level(task_data: dict, competency_levels: dict) -> str:
    levels = [competency_levels.get(strategy, "nivel_1")
              for strategy in get_strategies(task_data)]
    return min(levels, default="nivel_2")


def get_resolved_prompt(task_data: dict, competency_levels: dict) -> str:
    """Resuelve el enunciado con el menor nivel de las competencias implicadas."""
    prompt_data = task_data.get("task_prompt", "")
    level_val = get_task_level(task_data, competency_levels)
    if isinstance(prompt_data, dict):
        nivel_map = {"nivel_1": "nivel_bajo", "nivel_2": "nivel_medio", "nivel_3": "nivel_alto"}
        prompt = prompt_data.get(nivel_map.get(level_val, "nivel_bajo"))
        if not isinstance(prompt, str):
            raise ValueError(f"Falta el enunciado de {level_val} en la tarea {task_data.get('task_id')}.")
        return prompt
    
    return str(prompt_data)

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"No se encuentra el archivo de tareas {path.name}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se puede leer el archivo de tareas {path.name}.") from exc


def load_test_inicial():
    raw_tasks = _load_json(INITIAL_PATH)
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("El test inicial debe contener una lista de tareas no vacía.")
    adapted_tasks = []
    for t in raw_tasks:
        # Le pasamos la tarea completa y un diccionario de niveles vacío.
        # Sin estrategia usa nivel_2; sin nivel para la estrategia usa nivel_1.
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


def load_curriculum_data():
    data = _load_json(TASKS_PATH)
    if not isinstance(data, dict) or not data:
        raise ValueError("El currículo debe contener asignaturas y sesiones.")
    for subject, scenarios in data.items():
        if not isinstance(scenarios, dict) or not scenarios:
            raise ValueError(f"La asignatura {subject} debe contener escenarios.")
        for scenario, sessions in scenarios.items():
            if not isinstance(sessions, dict) or not sessions:
                raise ValueError(f"El escenario {subject}/{scenario} debe contener sesiones.")
            for session, tasks in sessions.items():
                if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
                    raise ValueError(f"Las tareas de {subject}/{scenario}/{session} deben ser una lista de objetos.")
    return data

def load_curriculum(subject_id="ciencias", session_id="sesion1", scenario_id="escenario1"):
    data = load_curriculum_data()
    if (subject_id not in data or scenario_id not in data[subject_id]
            or session_id not in data[subject_id][scenario_id]):
        raise ValueError(f"No existe la sesión {subject_id}/{scenario_id}/{session_id}.")
    tasks = data[subject_id][scenario_id][session_id]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"La sesión {subject_id}/{session_id} no contiene tareas.")
    prepared_tasks = []
    for raw_task in tasks:
        task = dict(raw_task)
        task["strategies"] = get_strategies(task)
        if task.get("task_type") in ("subjetivo", "v_y_d") or task.get("es_tabla"):
            task["rubric"] = normalize_rubric(task.get("rubric"))
        if task.get("task_type") == "verdadero_falso":
            validate_true_false_task(task)
        get_resolved_prompt(task, {})
        prepared_tasks.append(task)
    return prepared_tasks

def calculate_levels_from_test(history: List[TaskEvaluationHistory]) -> Dict[str, str]:
    scores_by_strat = {}
    for h in history:
        if h.is_test_task:
            for strategy in get_strategies(h):
                scores_by_strat.setdefault(strategy, []).append(
                    h.score / h.max_score if h.max_score > 0 else 0
                )
    
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
    relevant_history = [h for h in history if current_strategy in get_strategies(h)
                        and not getattr(h, 'is_test_task', False)]
    
    huecos_acumulados = current_max_score
    hits = current_score
    
    for h in reversed(relevant_history):
        if huecos_acumulados >= 3.0:
            break 
        
        # OPCIÓN B: Las tareas con varios ítems (max_score > 1) llenan más rápido la ventana de evaluación.
        h_max = getattr(h, 'max_score', 1.0)
        huecos_acumulados += h_max
        hits += h.score
        
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
        main_tasks = load_curriculum(state.current_subject or "ciencias", state.current_session or "sesion1",
                                     state.current_scenario or "escenario1")
        msg = "Test inicial completado. Hemos adaptado el curso a tu nivel."
        first_task = main_tasks[0] if main_tasks else None
        
        # Igual aquí, utilizamos la función actualizada
        first_prompt = get_resolved_prompt(first_task, new_levels) if first_task else ""

        return {"test_mode": False, "competency_levels": new_levels, "remaining_tasks": main_tasks, "total_tasks": len(main_tasks), "current_task": first_task, "current_task_prompt": first_prompt, "last_feedback_message": msg}

    return {"current_task": None}

def node_evaluate(state: StudentGraphState):
    current_task = state.current_task
    if not current_task: return {}

    strategies = get_strategies(current_task)
    strat = current_task.get("strategy", "-".join(strategies) or "contenido")
    support_level = get_task_level(current_task, state.competency_levels)
    started = time.perf_counter()
    try:
        res = evaluate_task(current_task, state.current_student_response, support_level,
                            resolved_prompt=state.current_task_prompt)
    except Exception:
        res = None
    evaluation_time = state.current_evaluation_time + time.perf_counter() - started
    if res is None:
        return {
            "last_feedback_message": "No se pudo evaluar tu respuesta. Se ha conservado para reintentar.",
            "last_evaluation": {"status": "error"},
            "current_evaluation_time": evaluation_time,
        }

    is_test = state.test_mode
    new_levels = state.competency_levels.copy()
    
    if not is_test:
        for strategy in strategies:
            new_levels[strategy] = calculate_dynamic_level(
                strategy, state.competency_levels.get(strategy, "nivel_1"),
                res["score"], res["max_score"], state.task_history,
            )
    
    history_entry = TaskEvaluationHistory(
        task_id=current_task.get("task_id"),
        task_prompt=state.current_task_prompt,
        strategy=strat,
        strategies=strategies,
        sub_strategy=current_task.get("sub_strategy", ""),
        student_response=state.current_student_response,
        score=res["score"], 
        score_weighted=res["score"], 
        max_score=res["max_score"], 
        normalized_score=res["normalized_score"],
        scores_por_item=res["scores_por_item"],
        time_taken=state.current_response_time,
        evaluation_time=evaluation_time,
        support_level=support_level,
        navigation_history=state.current_navigation_history,
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
        "last_evaluation": {"status": "ok", **res},
        "current_evaluation_time": 0.0,
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
