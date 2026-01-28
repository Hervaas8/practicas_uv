import json
import yaml
import sys
from configuracion_llm import get_llm_evaluador
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate

# --- CARGAR CONFIGURACIÓN ---
try:
    with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_TEMPLATE = prompts["eval_strategy_prompt"]
    
    with open("rubricas_map.json", "r", encoding="utf-8-sig") as f:
        rubricas_map = json.load(f)
        
    with open("curriculum.json", "r", encoding="utf-8-sig") as f:
        curriculum = json.load(f)
        
    llm_evaluador = get_llm_evaluador()
    print("--- Configuración Cargada ---")

except Exception as e:
    print(f"Error de carga: {e}")
    sys.exit(1)

# --- FUNCIÓN DE EVALUACIÓN UNIVERSAL ---

def evaluate_task(task: dict, student_answer: any):
    """
    Evalúa cualquier tarea.
    1. Si es OBJETIVA (tiene correct_answer) -> Python calcula score.
    2. Si es SUBJETIVA (no correct_answer) -> LLM calcula score.
    3. Genera feedback basado en la ESTRATEGIA.
    """
    print(f"\nEvaluating Task: {task['task_id']} ({task['strategy']} - {task.get('sub_strategy')})")
    
    strategy = task.get("strategy", "sin_estrategia")
    correct_answer = task.get("correct_answer")
    
    # --- RUTA B: EVALUACIÓN AUTOMÁTICA (Python) ---
    if correct_answer:
        print("-> Modo: Objetivo (Python)")
        # Lógica simple de comparación (adaptar para listas/dicts si es necesario)
        # Aquí asumimos respuesta simple o lista para el ejemplo
        is_correct = False
        if isinstance(correct_answer, list):
             is_correct = str(student_answer) in correct_answer
        else:
             is_correct = str(student_answer) == str(correct_answer)
             
        score = 10.0 if is_correct else 0.0
        
        # Generamos feedback técnico
        feedback = f"Estrategia {strategy.upper()}: Respuesta correcta." if is_correct else f"Estrategia {strategy.upper()}: Respuesta incorrecta. Correcta era: {correct_answer}"
        
        return LLMEvaluationOutput(
            score=score,
            reasoning_for_professor=f"Auto-corregido. Match: {is_correct}",
            student_feedback=feedback
        )

    # --- RUTA A: EVALUACIÓN SUBJETIVA (LLM) ---
    else:
        print("-> Modo: Subjetivo (LLM)")
        sub_strategy = task.get("sub_strategy", "general")
        rubrica = rubricas_map.get(sub_strategy, "Evaluar coherencia.")
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm_evaluador
        
        try:
            return chain.invoke({
                "strategy_name": strategy,
                "sub_strategy": sub_strategy,
                "task_text": task.get("task_text", "No texto"),
                "rubrica_text": rubrica,
                "task_prompt": task["task_prompt"],
                "student_answer": student_answer
            })
        except Exception as e:
            print(f"Error LLM: {e}")
            return None