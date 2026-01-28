import json
import sys
from typing import List, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- IMPORTACIONES PROPIAS ---
# 1. Esquemas de datos
from esquemas import StudentGraphState, TaskEvaluationHistory
# 2. El motor de evaluación (que ya incluye OCR, LLM y selección de feedback)
from motor_evaluacion import evaluate_task

# --- LÓGICA DE ADAPTATIVIDAD (CÁLCULO DE NIVEL) ---
def load_and_flatten_test():
    """
    Carga test_inicial.json y lo prepara.
    Separa explícitamente la 'question' de las 'options'.
    """
    processed_tasks = []
    try:
        with open("test_inicial.json", "r", encoding="utf-8-sig") as f:
            raw_data = json.load(f)
            
        for parent in raw_data:
            parent_prompt = parent.get("task_prompt", "")
            # Normalizamos imagen a lista
            p_imgs = parent.get("task_imagen", [])
            if isinstance(p_imgs, str): p_imgs = [p_imgs]
            
            for sub in parent.get("subtasks", []):
                # Construimos la tarea con el formato interno
                task = {
                    "task_id": sub.get("id"),
                    "strategy": sub.get("strategy", "general"),
                    "task_type": "objetivo", # Asumimos test
                    "parent_prompt_text": parent_prompt,
                    "parent_id": parent.get("id", "test_parent"),
                    
                    # Guardamos la pregunta y las opciones POR SEPARADO
                    "question_text": sub.get("question"), 
                    "options": sub.get("options", []),
                    "correct_answer": sub.get("correct")
                }
                
                # Imagen: Si la subtarea no tiene, hereda la del padre
                if p_imgs:
                    task["task_image"] = p_imgs
                
                processed_tasks.append(task)
                
        return processed_tasks
    except Exception as e:
        print(f"Error cargando test inicial: {e}")
        return []

def calculate_initial_levels(scores: dict) -> dict:
    """Calcula los niveles (1, 2 o 3) según los aciertos del test."""
    levels = {}
    print("\n--- RESULTADOS TEST INICIAL ---")
    for strategy, points in scores.items():
        if not points:
            levels[strategy] = "nivel_2" # Nivel medio por defecto
            continue
            
        avg = sum(points) / len(points)
        print(f"Estrategia {strategy}: {sum(points)}/{len(points)} ({avg:.2f})")
        
        if avg < 0.5:
            levels[strategy] = "nivel_1" # Nivel básico (muchos fallos)
        elif avg < 0.8:
            levels[strategy] = "nivel_2" # Nivel medio
        else:
            levels[strategy] = "nivel_3" # Nivel avanzado
            
    return levels


def calculate_new_level(
    current_strategy: str,
    current_weighted_score: float,
    task_history: List[TaskEvaluationHistory]
) -> str:
    """
    Calcula el nuevo nivel para una estrategia basándose en la media 
    de las últimas 4 notas ponderadas.
    
    Umbrales:
    - < 0.50 -> nivel_3 (Básico)
    - 0.50 - 0.80 -> nivel_2 (Medio)
    - > 0.80 -> nivel_1 (Avanzado)
    """
    
    # 1. Filtrar historial por la estrategia actual
    relevant_scores = [
        item.score_weighted 
        for item in task_history 
        if item.strategy == current_strategy
    ]
    
    # 2. Añadir la nota actual
    relevant_scores.append(current_weighted_score)
    
    # 3. Quedarse con las últimas 4
    last_4_scores = relevant_scores[-4:]
    
    # 4. Calcular media
    if not last_4_scores:
        average = 0.0
    else:
        average = sum(last_4_scores) / len(last_4_scores)
        
    print(f"   [Adaptativo] Estrategia '{current_strategy}': Notas {last_4_scores} -> Media: {average:.2f}")
    
    # 5. Aplicar Umbrales
    if average < 0.50:
        return "nivel_3"
    elif average < 0.80:
        return "nivel_2"
    else:
        return "nivel_1"

# --- NODOS DEL GRAFO ---
def node_ask_permission(state: StudentGraphState):
    """Pregunta al usuario con OPCIONES para que salgan botones."""
    print("--- NODO: PREGUNTAR TEST ---")
    return {
        "current_task": {"task_type": "normal"}, # Dummy
        # TRUCO: Ponemos formato 'a)' y 'b)' para que app.py saque botones
        "current_task_prompt": (
            "👋 ¡Hola! Antes de empezar la sesión, me gustaría conocer tu nivel.\n\n"
            "¿Quieres realizar un breve **Test Inicial** para adaptar los ejercicios a ti?\n\n"
            "a) Sí, quiero hacer el test.\n"
            "b) No, prefiero empezar la sesión directamente."
        ),
        "ask_assessment_permission": False
    }

def node_process_permission(state: StudentGraphState):
    """
    Procesa la respuesta al permiso.
    CORRECCIÓN: Si acepta, PREPARA VISUALMENTE la primera pregunta aquí mismo.
    Esto evita que se muestre el prompt anterior repetido debido al interrupt.
    """
    resp = str(state.current_student_response).lower().strip()
    letter = resp.split(" ")[0].replace(")", "").replace(".", "")
    
    # Opción A: SÍ
    if letter == "a" or resp in ["sí", "si", "yes", "s", "y", "claro"]:
        print("-> Usuario ACEPTA test.")
        test_tasks = load_and_flatten_test()
        
        # --- PREPARAR LA PRIMERA PREGUNTA (FIX VISUAL) ---
        first_task = None
        first_prompt = ""
        last_pid = None

        if test_tasks:
            first_task = test_tasks[0]
            # Usamos la misma lógica de formateo que en node_conduct_assessment
            q_text = first_task["question_text"]
            opts = "\n".join(first_task["options"])
            parent_txt = first_task.get("parent_prompt_text", "")
            
            prompt_show = f"{parent_txt}\n\n{q_text}".strip()
            first_prompt = f"{prompt_show}\n\n{opts}"
            last_pid = first_task.get("parent_id")

        return {
            "in_initial_assessment": True,
            "ask_assessment_permission": False, # Rompe el bucle de preguntar
            "assessment_questions": test_tasks,
            "assessment_index": 0,
            "last_feedback_message": "¡Estupendo! Vamos a evaluar tu nivel.",
            "current_student_response": None, # Limpiamos para no evaluar el "Sí"
            
            # --- NUEVO: Actualizamos la visualización YA ---
            "current_task": first_task,
            "current_task_prompt": first_prompt,
            "last_parent_id": last_pid
        }
    
    # Opción B: NO
    else:
        print("-> Usuario RECHAZA test.")
        return {
            "in_initial_assessment": False,
            "ask_assessment_permission": False,
            "last_feedback_message": "Entendido. Empezaremos con el nivel estándar.",
            "current_student_response": None
        }

def node_conduct_assessment(state: StudentGraphState):
    """Bucle del test inicial con corrección de feedback."""
    questions = state.assessment_questions
    idx = state.assessment_index
    scores = state.assessment_scores
    
    # Variable para guardar si hemos generado un feedback nuevo (Correcto/Incorrecto)
    new_feedback = None

    # 1. EVALUAR RESPUESTA (Si el usuario ha respondido algo)
    if state.current_student_response and state.in_initial_assessment:
        # Recuperamos la pregunta anterior (la que el usuario acaba de contestar)
        # Nota: Si idx es 0, venimos de "Sí", pero eso lo limpiamos en el nodo anterior.
        # Así que aquí solo entra respuestas reales a preguntas (idx > 0 idealmente)
        
        # Como idx apunta a la pregunta que "toca hacer", la que acabamos de hacer es idx-1?
        # En tu flujo: Muestras P0 -> Usuario responde -> Entras aquí (idx sigue siendo 0).
        # Así que la pregunta a evaluar es questions[idx].
        
        task_to_evaluate = questions[idx] # La pregunta actual
        
        user_raw = str(state.current_student_response).lower().strip()
        user_ans = user_raw.split(" ")[0].replace(")", "").replace(".", "")[0]
        
        correct_ans = str(task_to_evaluate.get("correct_answer")).lower().strip()
        strat = task_to_evaluate["strategy"]
        
        is_correct = (user_ans == correct_ans)
        scores[strat].append(1 if is_correct else 0)
        
        new_feedback = "✅ ¡Correcto!" if is_correct else f"❌ Incorrecto (Era la opción {correct_ans})"
        print(f"   Evaluado P{idx}: {new_feedback}")
        
        # ¡IMPORTANTE! Avanzamos el índice AHORA, después de evaluar
        idx += 1

    # 2. MOSTRAR SIGUIENTE PREGUNTA (Si quedan)
    if idx < len(questions):
        next_q = questions[idx]
        
        # Preparar textos
        q_text = next_q["question_text"]
        opts = "\n".join(next_q["options"])
        
        # --- LÓGICA DEL MENSAJE DE FEEDBACK ---
        # Si tenemos un feedback nuevo (de corrección), lo usamos.
        # Si no (es el inicio del test), usamos el que venga de antes ("Estupendo...")
        feedback_to_show = new_feedback if new_feedback else state.last_feedback_message

        # Gestión visual de Padre (para no repetir texto largo)
        parent_txt = next_q.get("parent_prompt_text", "")
        if next_q.get("parent_id") == state.last_parent_id:
            prompt_show = q_text
            next_q["task_text"] = None 
            next_q["task_image"] = None
        else:
            prompt_show = f"{parent_txt}\n\n{q_text}".strip()

        full_prompt = f"{prompt_show}\n\n{opts}"

        return {
            "current_task": next_q,
            "current_task_prompt": full_prompt,
            "assessment_index": idx, # Guardamos el índice actualizado
            "assessment_scores": scores,
            "last_parent_id": next_q.get("parent_id"),
            "last_feedback_message": feedback_to_show, # <--- USAMOS EL FEEDBACK CORRECTO
            "current_student_response": None # Limpiamos input para no re-evaluar lo mismo
        }
    
    else:
        # 3. FIN DEL TEST
        final_levels = calculate_initial_levels(scores)
        summary = "🎉 **Test Completado.** Hemos personalizado tu plan:\n"
        for k, v in final_levels.items():
            summary += f"- {k.capitalize()}: {v}\n"
            
        return {
            "in_initial_assessment": False,
            "competency_levels": final_levels,
            "current_task": None,
            "last_feedback_message": summary,
            "current_student_response": None
        }

def node_load_curriculum(state: StudentGraphState):
    """Carga las tareas normales (ahora respeta los niveles del test)."""
    # Si estamos en test, no hacemos nada
    if state.in_initial_assessment:
        return {}
        
    # Si ya hay tareas, no hacemos nada
    if state.remaining_tasks:
        return {}

    print("\n--- CARGANDO SESIÓN PRINCIPAL ---")
    
    if state.remaining_tasks:
        return {}

    materia = "ciencias"
    sesion = "sesion1"
    
    processed_tasks = []

    try:
        with open("tareas.json", "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            raw_tasks = data.get(materia, {}).get(sesion, [])
            
            for parent in raw_tasks:
                # 1. Datos del Padre (Contexto)
                parent_prompt = parent.get("task_prompt", "")
                parent_text = parent.get("task_text", "")
                # Convertimos imagen padre a lista si es string, o None
                parent_images = parent.get("task_image")
                if isinstance(parent_images, str): parent_images = [parent_images]
                
                # 2. Iterar Subtareas
                subtasks = parent.get("subtasks", [])
                for sub in subtasks:
                    final_task = sub.copy()
                    
                    final_task["parent_id"] = parent.get("task_id")
                    # --- HERENCIA Y COMBINACIÓN DE TEXTOS ---
                    
                    # a) ID: Si no tiene, construimos uno
                    if "task_id" not in final_task:
                        final_task["task_id"] = sub.get("subtask_id", f"{parent.get('task_id')}_sub")

                    # b) Prompt: Concatenamos (Padre + Hijo)
                    # El prompt del padre va fijo al principio
                    # El prompt del hijo depende del nivel (se seleccionará en node_select_task)
                    # Aquí guardamos el padre en un campo separado para unirlo luego
                    final_task["parent_prompt_text"] = parent_prompt 
                    
                    # c) Texto de Contexto (Noticia, Hilo de X, etc)
                    # Si la subtask no tiene, usa el del padre
                    if not final_task.get("task_text"):
                        final_task["task_text"] = parent_text

                    # d) Imágenes
                    # Si la subtask tiene steps con imágenes (interactivo), NO heredamos imagen del padre
                    # Si es test/subjetivo y no tiene imagen, heredamos la del padre
                    if final_task.get("task_type") != "interactivo":
                        if "task_image" not in final_task or not final_task["task_image"]:
                            final_task["task_image"] = parent_images

                    # e) Normalización de claves (arreglando el error subtaskprompts)
                    if "subtaskprompts" in final_task:
                        final_task["task_prompts"] = final_task.pop("subtaskprompts")
                    elif "subtask_prompts" in final_task:
                        final_task["task_prompts"] = final_task.pop("subtask_prompts")
                    
                    # f) Para interactivas: Si la dificultad está en los steps, calculamos media
                    # (Esto es un parche por si no cambias el JSON)
                    if final_task.get("task_type") == "interactivo" and "difficulty" not in final_task:
                        steps = final_task.get("steps", [])
                        if steps:
                            diffs = [s.get("difficulty", 1.0) for s in steps]
                            avg_diff = sum(diffs) / len(diffs)
                            final_task["difficulty"] = avg_diff
                            # Cogemos la estrategia del primer step como principal
                            final_task["strategy"] = steps[0].get("strategy", "general")

                    processed_tasks.append(final_task)
            
            print(f"-> Cargadas {len(processed_tasks)} tareas aplanadas.")
            
            # Niveles por defecto
            current_levels = state.competency_levels.copy()
            if not current_levels:
                current_levels = {"navegar": "nivel_1", "evaluar": "nivel_1", "integrar": "nivel_1"}
                print("-> Niveles por defecto aplicados.")
            else:
                print(f"-> Usando niveles personalizados: {current_levels}")    
            return {
                "remaining_tasks": processed_tasks, 
                "current_subject": materia, 
                "current_session": sesion,
                "competency_levels": current_levels
            }

    except Exception as e:
        print(f"Error crítico cargando curriculum: {e}")
        import traceback
        traceback.print_exc()
        return {}

def node_select_task(state: StudentGraphState):
    """
    Selecciona la tarea.
    Muestra feedback en consola (logs) pero MANTIENE el mensaje en el estado para Streamlit.
    """
    # 1. LOG EN CONSOLA (Opcional, útil para depurar)
    if state.last_feedback_message:
        print(f"{state.last_feedback_message}")
        print("-" * 30)

    print("--- NODO: SELECCIONAR TAREA ---")
    
    # 2. COMPROBAR FIN DE SESIÓN
    if not state.remaining_tasks:
        return {
            "current_task": None,
            # ELIMINAMOS LA LÍNEA DE LIMPIEZA AQUÍ
            # "last_feedback_message": None  <-- BORRAR ESTO
        } 

    next_task = state.remaining_tasks[0].copy()
    
    strategy = next_task.get("strategy", "general")
    current_level = state.competency_levels.get(strategy, "nivel_2")
    
    # --- LÓGICA DE VISUALIZACIÓN INTELIGENTE ---
    
    current_parent = next_task.get("parent_id")
    last_parent = state.last_parent_id
    
    # Obtenemos SOLO la pregunta específica (del hijo)
    prompts_dict = next_task.get("task_prompts", {})
    specific_prompt = prompts_dict.get(current_level, next_task.get("task_prompt", ""))
    parent_part = next_task.get("parent_prompt_text", "")

    prompt_to_show = ""
    
    if current_parent and current_parent == last_parent:
        # CASO A: CONTINUACIÓN (Mismo padre que antes)
        # Solo mostramos la pregunta específica.
        # Eliminamos el texto de contexto e imagen para no repetir visualmente en Streamlit
        print(f"   [UX] Ocultando contexto padre repetido ({current_parent})")
        prompt_to_show = specific_prompt
        
        # Limpiamos visuales redundantes para este turno
        next_task["task_text"] = None 
        next_task["task_image"] = None 
        
    else:
        # CASO B: NUEVO EJERCICIO (Padre distinto o primero)
        # Mostramos Padre + Pregunta
        if parent_part:
            prompt_to_show = f"{parent_part}\n\n{specific_prompt}".strip()
        else:
            prompt_to_show = specific_prompt
            
    # -------------------------------------------
    
    print(f"-> Tarea: {next_task['task_id']} | Nivel: {current_level}")
    
    return {
        "current_task": next_task, 
        "current_task_prompt": prompt_to_show,
        "last_parent_id": current_parent # <--- ACTUALIZAMOS LA MEMORIA AQUÍ
    }

def node_evaluate(state: StudentGraphState):
    """
    Procesa la respuesta usando el motor de evaluación externo.
    """
    print("--- NODO: EVALUAR Y ADAPTAR ---")
    
    current_task = state.current_task
    student_response = state.current_student_response
    
    if not current_task:
        return {}

    # 1. Contexto para evaluación
    strategy = current_task.get("strategy", "general")
    current_level = state.competency_levels.get(strategy, "nivel_2")

    # 2. LLAMADA AL MOTOR DE EVALUACIÓN (Tu nuevo módulo)
    # Devuelve: {'score': float, 'reasoning': str, 'feedback_shown': str}
    eval_result = evaluate_task(current_task, student_response, current_level)
    
    if eval_result is None:
        feedback = "Error técnico evaluando tu respuesta."
        raw_score = 0.0
        reasoning = "Fallo en motor de evaluación."
    else:
        feedback = eval_result['feedback_shown']
        raw_score = eval_result['score']
        reasoning = eval_result['reasoning']
        
    # 3. Cálculo de Nota Ponderada
    difficulty = current_task.get("difficulty", 1.0)
    weighted_score = raw_score * difficulty
    
    print(f"   -> Nota Base: {raw_score} | Dif: {difficulty} | Ponderada: {weighted_score:.2f}")

    # 4. Actualizar Nivel (Adaptatividad)
    new_level = calculate_new_level(strategy, weighted_score, state.task_history)
    
    if new_level != current_level:
        print(f"   >>> CAMBIO DE NIVEL en '{strategy}': {current_level} -> {new_level}")
    
    new_competency_levels = state.competency_levels.copy()
    new_competency_levels[strategy] = new_level

    # 5. Guardar Historial
    history_entry = TaskEvaluationHistory(
        task_id=current_task["task_id"],
        strategy=strategy,
        sub_strategy=current_task.get("sub_strategy", ""),
        student_response=str(student_response),
        score=raw_score,
        score_weighted=weighted_score,
        feedback_shown=feedback,
        reasoning=reasoning
    )
    
    # 6. Avanzar
    remaining = state.remaining_tasks[1:]
    new_history = state.task_history + [history_entry]
    
    # 7. Mensaje al usuario
    # Si es subjetiva -> Feedback inmediato. Si es objetiva -> Silencio.
    if feedback and str(feedback).strip():
        msg_to_user = f"\n[FEEDBACK]: {feedback}"
    else:
        msg_to_user = "\n[Respuesta guardada]"

    return {
        "remaining_tasks": remaining,
        "task_history": new_history,
        "current_task": None,
        "last_evaluation": eval_result,
        "current_student_response": None,
        "competency_levels": new_competency_levels,
        "last_feedback_message": msg_to_user
    }

def node_final_report(state: StudentGraphState):
    """
    Genera el dashboard final.
    """
    print("\n\n===========================================")
    print("       DASHBOARD FINAL DE SESIÓN           ")
    print("===========================================")
    
    history = state.task_history
    total_weighted = 0
    
    print("\n[DETALLE DE TAREAS]")
    for item in history:
        print(f"• Tarea {item.task_id} ({item.strategy})")
        print(f"  - Nota Ponderada: {item.score_weighted:.2f}")
        # Si fue objetiva (feedback diferido), lo mostramos ahora
        if "Auto-corregido" in item.reasoning:
             print(f"  - Feedback Técnico: {item.feedback_shown}")
        total_weighted += item.score_weighted
        
    avg = total_weighted / len(history) if history else 0
    
    print(f"\n[RESUMEN GLOBAL]")
    print(f"Nota Media Ponderada: {avg:.2f}")
    print(f"Niveles Finales: {state.competency_levels}")
    print("===========================================\n")
    return {}

# --- DEFINICIÓN DEL GRAFO ---

# --- ROUTERS ---

def router_start(state: StudentGraphState):
    """Router inicial robusto."""
    # 1. Si estamos DENTRO del test -> Seguir test
    if state.in_initial_assessment:
        return "conduct_assessment"
    
    # 2. Si todavía TENEMOS QUE PREGUNTAR -> Preguntar
    # (Como ahora lo ponemos a False al responder, esto ya no se repite)
    if state.ask_assessment_permission:
        return "ask_permission"
    
    # 3. Resto del flujo...
    if not state.remaining_tasks:
        return "load_curriculum"
        
    return "select_task"

def router_after_permission(state: StudentGraphState):
    if state.in_initial_assessment:
        return "conduct_assessment"
    return "load_curriculum"

def router_assessment_loop(state: StudentGraphState):
    if state.in_initial_assessment:
        return "conduct_assessment"
    return "load_curriculum"

def router_main_loop(state: StudentGraphState):
    if not state.current_task:
        return "final_report"
    return "evaluate"

# --- GRAFO ---

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(StudentGraphState)

# Nodos
builder.add_node("ask_permission", node_ask_permission)
builder.add_node("process_permission", node_process_permission)
builder.add_node("conduct_assessment", node_conduct_assessment)
builder.add_node("load_curriculum", node_load_curriculum)
builder.add_node("select_task", node_select_task)
builder.add_node("evaluate", node_evaluate)
builder.add_node("final_report", node_final_report)

# ENTRADA DINÁMICA (Soluciona el problema de repetir la pregunta)
builder.set_conditional_entry_point(
    router_start,
    {
        "ask_permission": "ask_permission",
        "conduct_assessment": "conduct_assessment",
        "load_curriculum": "load_curriculum",
        "select_task": "select_task"
    }
)

# Aristas
builder.add_edge("ask_permission", "process_permission")

builder.add_conditional_edges(
    "process_permission",
    router_after_permission,
    {"conduct_assessment": "conduct_assessment", "load_curriculum": "load_curriculum"}
)

builder.add_conditional_edges(
    "conduct_assessment",
    router_assessment_loop,
    {"conduct_assessment": "conduct_assessment", "load_curriculum": "load_curriculum"}
)

builder.add_edge("load_curriculum", "select_task")

builder.add_conditional_edges(
    "select_task",
    router_main_loop,
    {"evaluate": "evaluate", "final_report": "final_report"}
)

builder.add_edge("evaluate", "select_task")
builder.add_edge("final_report", END)

memory = MemorySaver()

app = builder.compile(
    checkpointer=memory,
    interrupt_before=["process_permission", "conduct_assessment", "evaluate"]
)
# --- EJECUCIÓN (SIMULACIÓN) ---

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "alumno_demo_visual_fix"}}
    print("\n=== AGENTE TUTOR INICIADO ===")
    
    app.invoke({"student_id": "alumno_1"}, config=config)
    
    while True:
        state_values = app.get_state(config).values
        if not state_values: break
        
        current_task = state_values.get('current_task')
        if not current_task: 
            # El feedback final ya lo habrá impreso node_final_report
            break
        
        # Aquí ya SOLO mostramos el enunciado, el feedback lo imprimió el nodo anterior
        print(f"\n🤖 AGENTE: {state_values.get('current_task_prompt')}")
        
        user_input = input("👤 TÚ: ")
        
        app.update_state(config, {"current_student_response": user_input})
        for event in app.stream(None, config=config): pass