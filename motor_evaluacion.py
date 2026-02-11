import json
import yaml
import sys
from configuracion_llm import get_llm_evaluador
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate

# --- IMPORTACIÓN DE UTILIDADES OCR ---
try:
    from utils_ocr import extract_text_from_image
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

def select_feedback(task: dict, score: float):
    """Selecciona feedback basado en el resultado (éxito >= 0.5)"""
    is_success = score >= 0.5
    key = "success" if is_success else "failure"
    feedback_dict = task.get("feedback", {})
    return feedback_dict.get(key, "Feedback no configurado.")

def evaluate_task(task: dict, student_answer: any, student_level: str):
    task_id = task.get('task_id', 'Desconocida')
    task_type = task.get('task_type')
    print(f"\n[Motor Evaluación] Task: {task_id} | Type: {task_type}")
    
    score = 0.0
    reasoning = ""

    # --- TIPO: CONECTAR (Objetiva) ---
    if task_type == "conectar":
        correct_mapping = task.get("correct_mapping", {})
        matches = 0
        total_required = sum(len(defs) if isinstance(defs, list) else 1 for defs in correct_mapping.values())
        
        # student_answer debe ser dict { concepto: [definiciones] }
        for concept, correct_defs in correct_mapping.items():
            st_defs = student_answer.get(concept, [])
            if isinstance(st_defs, str): st_defs = [st_defs]
            for d in st_defs:
                if d in (correct_defs if isinstance(correct_defs, list) else [correct_defs]):
                    matches += 1
        
        score = matches / total_required if total_required > 0 else 0
        reasoning = f"Auto-corregido: {matches} de {total_required} conexiones correctas."

    # --- TIPO: OBJETIVO (Opción Múltiple) ---
    elif task_type == "objetivo":
        correct_answer = task.get("correct_answer")
        def normalize(v): return [str(x).strip().lower() for x in (v if isinstance(v, list) else [v])]
        correct_list = normalize(correct_answer)
        student_list = normalize(student_answer)
        matches = sum(1 for i, v in enumerate(correct_list) if i < len(student_list) and student_list[i] == v)
        score = matches / len(correct_list) if correct_list else 0
        reasoning = f"Auto-corregido: {matches}/{len(correct_list)} aciertos."

    # --- TIPO: INTERACTIVO (Un solo paso) ---
    # --- TAREA TIPO: INTERACTIVO (Validación de Clic) ---
    elif task_type == "interactivo":
        import re
        # Buscamos el patrón "CLICK: x, y" que envía la app
        match = re.search(r"CLICK: (\d+), (\d+)", str(student_answer))
        if match:
            click_x = int(match.group(1))
            click_y = int(match.group(2))
            regions = task.get("regions", [])
            
            acierto = False
            reasoning = f"Clic detectado en ({click_x}, {click_y}). "
            
            # Validamos si el clic está dentro de alguna región correcta
            for reg in regions:
                rx, ry, rw, rh = reg['x'], reg['y'], reg['w'], reg['h']
                if rx <= click_x <= rx + rw and ry <= click_y <= ry + rh:
                    if reg.get("is_correct"):
                        acierto = True
                        reasoning += "El clic coincide con una zona correcta."
                        break
            
            if acierto:
                score = 1.0
            else:
                score = 0.0
                reasoning += "El clic no coincide con ninguna zona válida."
        else:
            score = 0.0
            reasoning = "No se detectaron coordenadas válidas en la respuesta."

    # --- TIPO: SUBJETIVO / V_Y_D (LLM) ---
    else:
        context_parts = [task.get("task_text", "")]
        images = task.get("task_image", [])
        if isinstance(images, str): images = [images]
        for img in images:
            context_parts.append(extract_text_from_image(img))
        
        final_context = "\n\n".join(filter(None, context_parts))
        rubric = task.get("rubric", "Evaluar calidad general.")
        rubric_text = json.dumps(rubric, ensure_ascii=False) if isinstance(rubric, dict) else str(rubric)

        prompt_tmpl = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt_tmpl | llm_evaluador
        try:
            llm_result = chain.invoke({
                "task_text": final_context,
                "rubrica_text": rubric_text,
                "task_prompt": task.get("task_prompt", ""),
                "student_answer": student_answer
            })
            score, reasoning = round(llm_result.score, 2), llm_result.reasoning
        except Exception as e:
            print(f"Error LLM: {e}"); return None

    return {
        "score": score,
        "reasoning": reasoning,
        "feedback_shown": select_feedback(task, score)
    }