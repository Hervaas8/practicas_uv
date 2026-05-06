import json
import yaml
import sys
from configuracion_llm import get_llm_evaluador
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    PROMPT_FEEDBACK_FINAL = prompts["eval_strategy_feedback_prompt"]
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

    # --- TIPO: SUBJETIVO / V_Y_D ---
    else:
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

    # --- CÁLCULO DE MAX_REF SEGURO ---
    max_ref = 1.0
    if task_type not in ["conectar", "objetivo", "interactivo"]:
        rubric = task.get("rubric", {})
        if isinstance(rubric, dict):
            puntos = sum(v for v in rubric.values() if isinstance(v, (int, float)))
            if puntos > 0:
                max_ref = float(puntos)

    porcentaje_acierto = score / max_ref if max_ref > 0 else 0.0

    return {
        "score": score,
        "reasoning": reasoning,
        "feedback_shown": select_feedback(task, porcentaje_acierto)
    }

# --- 1. DICCIONARIO DE SUB-ESTRATEGIAS ---
MAPA_SUBESTRATEGIAS = {
    "N1": "Identificar la estructura del texto online",
    "N2": "Determinar relevancia de la info",
    "I1": "Combinar información complementaria",
    "I2": "Enfrentarse al conflicto intertextual",
    "E1": "Analizar expertise del autor",
    "E2": "Analizar benevolencia"
}

def generate_strategy_feedback(strategy_name: str, level: str, history: list) -> str:
    """Genera un feedback cualitativo usando el LLM basado en el historial y sub-estrategias."""
    def get_val(item, key, default=""):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    strat_history = [
        h for h in history 
        if get_val(h, 'strategy') == strategy_name and not get_val(h, 'is_test_task', False)
    ]
    
    if not strat_history:
        return "No hay suficientes datos en esta sesión para realizar un análisis detallado."

    sub_vistas = set()
    historial_texto = ""
    
    for i, h in enumerate(strat_history):
        sub_code = get_val(h, 'sub_strategy', "General")
        significado = MAPA_SUBESTRATEGIAS.get(sub_code, "Habilidad general")
        
        if sub_code != "General" and sub_code is not None:
            sub_vistas.add(f"- {sub_code}: {significado}")

        historial_texto += f"Tarea {i+1} (Habilidad específica: {sub_code}):\n"
        historial_texto += f"- Pregunta: {get_val(h, 'task_prompt')}\n"
        historial_texto += f"- Respuesta del alumno: {get_val(h, 'student_response')}\n"
        historial_texto += f"- Nota: {get_val(h, 'score', 0)}\n"
        historial_texto += f"- Motivo de la nota: {get_val(h, 'reasoning')}\n\n"

    sub_info_ayuda = "\n".join(sub_vistas)

    prompt_tmpl = PromptTemplate.from_template(PROMPT_FEEDBACK_FINAL)
    chain = prompt_tmpl | llm_evaluador 
    
    try:
        resultado = chain.invoke({
            "estrategia": strategy_name.capitalize(),
            "nivel": level,
            "sub_info": sub_info_ayuda,
            "historial_texto": historial_texto.strip()
        })
        return resultado.reasoning
    except Exception as e:
        print(f"Error generando feedback final para {strategy_name}: {e}")
        return "Ha habido un problema al generar tus consideraciones detalladas."