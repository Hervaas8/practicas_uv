import json
import sys
from typing import List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- IMPORTACIONES PROPIAS ---
from esquemas import StudentGraphState, TaskEvaluationHistory
from motor_evaluacion import evaluate_task

def load_test_inicial():
    """Carga y adapta el test inicial al formato de tareas del sistema."""
    try:
        with open("test_inicial.json", "r", encoding="utf-8-sig") as f:
            raw_tasks = json.load(f)
        
        adapted_tasks = []
        for t in raw_tasks:
            adapted_tasks.append({
                "task_id": t.get("id"),
                "task_prompt": t.get("task_prompt"),
                "task_image": t.get("task_image", []),
                "task_type": "objetivo",
                "correct_answer": t.get("correct_answer"),
                "strategy": t.get("strategy", "general"),
                "sub_strategy": None,
                "description": "Test Inicial"
            })
        return adapted_tasks
    except Exception as e:
        print(f"Error cargando test_inicial.json: {e}")
        return []

def load_curriculum():
    """Carga el curriculum principal."""
    try:
        with open("tareas2.json", "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            return data.get("ciencias", {}).get("sesion1", [])
    except Exception as e:
        print(f"Error cargando tareas2.json: {e}")
        return []

def calculate_levels_from_test(history: List[TaskEvaluationHistory]) -> Dict[str, str]:
    """Calcula niveles iniciales basándose SOLO en las tareas del test."""
    scores_by_strat = {}
    for h in history:
        if h.is_test_task:
            if h.strategy not in scores_by_strat:
                scores_by_strat[h.strategy] = []
            scores_by_strat[h.strategy].append(h.score)
    
    new_levels = {}
    # MODIFICACIÓN: Eliminamos "contenido" de las estrategias con nivel
    strategies = ["navegar", "integrar", "evaluar"] 
    
    for strat in strategies:
        scores = scores_by_strat.get(strat, [])
        if not scores:
            new_levels[strat] = "nivel_1"
        else:
            avg = sum(scores) / len(scores)
            if avg < 0.50: new_levels[strat] = "nivel_1"
            elif avg < 0.80: new_levels[strat] = "nivel_2"
            else: new_levels[strat] = "nivel_3"
                
    return new_levels

def calculate_dynamic_level(current_strategy: str, current_level: str, current_score: float, history: List[TaskEvaluationHistory]) -> str:
    """
    Calcula el nuevo nivel basándose en la matriz de transición de las últimas 3 respuestas.
    Lógica basada en aciertos (score >= 0.5) y fallos.
    """
    # 1. Recuperamos las últimas tareas de esta estrategia (excluyendo el test)
    relevant_history = [h for h in history if h.strategy == current_strategy and not h.is_test_task]
    
    # 2. Cogemos las últimas 2 para sumar a la actual (Total ventana: max 3)
    last_entries = relevant_history[-2:]
    
    # 3. Determinamos aciertos/fallos (True/False)
    # Consideramos Acierto si score >= 0.5
    results = [h.score >= 0.5 for h in last_entries]
    results.append(current_score >= 0.5) # Añadimos la actual
    
    hits = sum(results)         # Número de aciertos en la ventana
    
    # --- LÓGICA DE TRANSICIÓN SEGÚN MATRIZ ---
    
    if current_level == "nivel_1":
        # NIVEL 1:
        # 3 aciertos (VVV) -> Sube
        # 2 aciertos (VVx) -> Sube
        # Menos de 2 -> Se mantiene
        if hits >= 2:
            return "nivel_2"
        else:
            return "nivel_1"

    elif current_level == "nivel_2":
        # NIVEL 2:
        # 3 aciertos (VVV) -> Sube a Nivel 3
        # 2 aciertos (VVx) -> Se mantiene
        # 1 o 0 aciertos (Vxx, xxx) -> Baja a Nivel 1
        if hits == 3:
            return "nivel_3"
        elif hits <= 1:
            return "nivel_1"
        else:
            return "nivel_2"

    elif current_level == "nivel_3":
        # NIVEL 3:
        # 3 aciertos (VVV) -> Se mantiene
        # Menos de 3 (VVx, Vxx, xxx) -> Baja a Nivel 2
        if hits == 3:
            return "nivel_3"
        else:
            return "nivel_2"

    return current_level

# --- NODOS DEL GRAFO ---

def node_load_curriculum(state: StudentGraphState):
    return {} 

def node_select_task(state: StudentGraphState):
    """Selecciona la siguiente tarea. Maneja la transición Test -> Curso."""
    
    if state.remaining_tasks:
        next_task = state.remaining_tasks[0]
        return {
            "current_task": next_task,
            "current_task_prompt": next_task.get("task_prompt", "")
        }
    
    if state.test_mode:
        print("--- FINALIZADO TEST INICIAL. CALCULANDO NIVELES ---")
        new_levels = calculate_levels_from_test(state.task_history)
        main_tasks = load_curriculum()
        
        msg = "Test inicial completado. Hemos adaptado el curso a tu nivel."
        
        return {
            "test_mode": False, 
            "competency_levels": new_levels,
            "remaining_tasks": main_tasks,
            "total_tasks": len(main_tasks),
            "current_task": main_tasks[0] if main_tasks else None,
            "current_task_prompt": main_tasks[0].get("task_prompt", "") if main_tasks else "",
            "last_feedback_message": msg
        }

    return {"current_task": None}

def node_evaluate(state: StudentGraphState):
    """Evalúa la respuesta."""
    current_task = state.current_task
    answer = state.current_student_response
    
    if not current_task: return {}

    # Obtenemos el nivel actual para la evaluación (si es contenido, no tiene nivel, usamos default para prompt)
    strat = current_task.get("strategy", "general")
    current_level_val = state.competency_levels.get(strat, "nivel_1")
    
    res = evaluate_task(current_task, answer, current_level_val)
    
    if not res: return {"last_feedback_message": "Error técnico en la evaluación."}

    is_test = state.test_mode
    new_levels = state.competency_levels.copy()
    
    # MODIFICACIÓN: Solo recalculamos nivel si NO es test Y NO es estrategia de contenido
    if not is_test and strat != "contenido":
        new_level = calculate_dynamic_level(
            strat, 
            current_level_val, 
            res["score"], 
            state.task_history
        )
        new_levels[strat] = new_level
    
    feedback_msg = res["feedback_shown"]
    if is_test:
        feedback_msg = "Respuesta registrada." 

    history_entry = TaskEvaluationHistory(
        task_id=current_task.get("task_id"),
        strategy=strat,
        sub_strategy=current_task.get("sub_strategy", ""),
        student_response=str(answer),
        score=res["score"],
        score_weighted=res["score"], 
        feedback_shown=feedback_msg,
        reasoning=res["reasoning"],
        is_test_task=is_test
    )

    return {
        "remaining_tasks": state.remaining_tasks[1:], 
        "task_history": state.task_history + [history_entry],
        "current_task": None,
        "competency_levels": new_levels,
        "last_feedback_message": feedback_msg
    }

# --- DEFINICIÓN DEL GRAFO ---

builder = StateGraph(StudentGraphState)
builder.add_node("load_curriculum", lambda state: {}) 
builder.add_node("select_task", node_select_task)
builder.add_node("evaluate", node_evaluate)

builder.set_entry_point("load_curriculum")
builder.add_edge("load_curriculum", "select_task")

builder.add_conditional_edges(
    "select_task",
    lambda s: "evaluate" if s.current_task else "end",
    {"evaluate": "evaluate", "end": END}
)

builder.add_edge("evaluate", "select_task")

memory = MemorySaver()
app = builder.compile(checkpointer=memory, interrupt_before=["evaluate"])