import json
import yaml
import sys
from configuracion_llm import get_llm_evaluador
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate

# --- IMPORTACIÓN DE UTILIDADES OCR ---
try:
    from utils_easy import extract_text_from_image
except ImportError:
    def extract_text_from_image(path): return "[ERROR: Módulo OCR no encontrado]"

# --- CARGAR CONFIGURACIÓN GLOBAL ---
try:
    with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_TEMPLATE = prompts["eval_subjective_prompt"]
    llm_evaluador = get_llm_evaluador()
except Exception as e:
    print(f"Error crítico al cargar configuración en motor_evaluacion: {e}")
    sys.exit(1)

def select_feedback(task: dict, normalized_score: float):
    """Selecciona feedback basado en el porcentaje de éxito (0.0 a 1.0)."""
    is_success = normalized_score >= 0.5
    key = "success" if is_success else "failure"
    feedback_dict = task.get("feedback", {})
    return feedback_dict.get(key, "Feedback no configurado.")

def evaluate_task(task: dict, student_answer: any, student_level: str):
    task_id = task.get('task_id', 'Desconocida')
    task_type = task.get('task_type')
    
    score = 0.0
    reasoning = ""

    # --- TIPO: CONECTAR ---
    if task_type == "conectar":
        correct_mapping = task.get("correct_mapping", {})
        matches = 0
        total_required = sum(len(defs) if isinstance(defs, list) else 1 for defs in correct_mapping.values())
        for concept, correct_defs in correct_mapping.items():
            st_defs = student_answer.get(concept, [])
            if isinstance(st_defs, str): st_defs = [st_defs]
            for d in st_defs:
                if d in (correct_defs if isinstance(correct_defs, list) else [correct_defs]):
                    matches += 1
        score = matches / total_required if total_required > 0 else 0
        reasoning = f"Auto-corregido: {matches} de {total_required} conexiones."

    # --- TIPO: OBJETIVO ---
    elif task_type == "objetivo":
        correct_answer = task.get("correct_answer")
        def normalize(v): return [str(x).strip().lower() for x in (v if isinstance(v, list) else [v])]
        correct_list = normalize(correct_answer)
        student_list = normalize(student_answer)
        matches = sum(1 for i, v in enumerate(correct_list) if i < len(student_list) and student_list[i] == v)
        score = matches / len(correct_list) if correct_list else 0
        reasoning = f"Auto-corregido: {matches}/{len(correct_list)} aciertos."

    # --- TIPO: INTERACTIVO ---
    elif task_type == "interactivo":
        import re
        match = re.search(r"CLICK: (\d+), (\d+)", str(student_answer))
        if match:
            click_x, click_y = int(match.group(1)), int(match.group(2))
            regions = task.get("regions", [])
            acierto = False
            for reg in regions:
                rx, ry, rw, rh = reg['x'], reg['y'], reg['w'], reg['h']
                if rx <= click_x <= rx + rw and ry <= click_y <= ry + rh:
                    if reg.get("is_correct"):
                        acierto = True; break
            score = 1.0 if acierto else 0.0
            reasoning = f"Clic en ({click_x}, {click_y}). {'Correcto' if acierto else 'Fuera de zona'}."
        else:
            score, reasoning = 0.0, "Coordenadas no válidas."

    # --- TIPO: SUBJETIVO / V_Y_D (Evaluación Multi-Objetivo) ---
    else:
        # Calculamos el peso total de la rúbrica si es un diccionario (varios objetivos)
        rubric = task.get("rubric", {})
        max_possible_points = 1.0
        if isinstance(rubric, dict):
            max_possible_points = float(sum(v for v in rubric.values() if isinstance(v, (int, float))))

        context_parts = [task.get("task_text", "")]
        for img in (task.get("task_image", []) if isinstance(task.get("task_image"), list) else [task.get("task_image")]):
            if img: context_parts.append(extract_text_from_image(img))
        
        final_context = "\n\n".join(filter(None, context_parts))
        rubric_text = json.dumps(rubric, ensure_ascii=False) if isinstance(rubric, dict) else str(rubric)

        prompt_tmpl = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt_tmpl | llm_evaluador
        try:
            # Informamos al LLM del máximo de puntos para que evalúe cada item por separado
            llm_result = chain.invoke({
                "task_text": final_context,
                "rubrica_text": f"Puntuación total máxima: {max_possible_points}. Detalle: {rubric_text}",
                "task_prompt": task.get("task_prompt", ""),
                "student_answer": student_answer
            })
            score = round(llm_result.score, 2)
            reasoning = llm_result.reasoning
        except Exception as e:
            print(f"Error LLM: {e}"); return None

    # Normalizamos el feedback basado en el porcentaje obtenido sobre el máximo
    # En tareas de 1 solo punto, score == normalized_score.
    # En tareas de 2 puntos, si saca 1.0, el normalized_score es 0.5 (aprobado).
    max_ref = max_possible_points if 'max_possible_points' in locals() else 1.0
    return {
        "score": score,
        "reasoning": reasoning,
        "feedback_shown": select_feedback(task, score / max_ref if max_ref > 0 else 0)
    }