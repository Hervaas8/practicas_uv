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
    # Fallback silencioso si no se encuentra el módulo, para no romper la ejecución
    def extract_text_from_image(path): return "[ERROR: Módulo OCR no encontrado]"

# --- CARGAR CONFIGURACIÓN GLOBAL (Prompts y Rúbricas Generales) ---
try:
    # Cargamos los prompts del sistema
    with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_TEMPLATE = prompts["eval_subjective_prompt"]
    
    
    # Inicializamos el LLM una sola vez
    llm_evaluador = get_llm_evaluador()
    
except Exception as e:
    print(f"Error crítico al cargar configuración en motor_evaluacion: {e}")
    sys.exit(1)

# --- FUNCIÓN AUXILIAR: SELECCIONAR FEEDBACK ---

def select_feedback(task: dict, level: str, score: float):
    """
    Selecciona el mensaje predefinido del JSON de la tarea basándose en:
    1. El Nivel del alumno (nivel_1, nivel_2, nivel_3).
    2. El Resultado (Success si score >= 5, Failure si score < 5).
    """
    is_success = score >= 0.5
    key = "success" if is_success else "failure"
    
    feedbacks = task.get("feedback_by_level", {})
    level_data = feedbacks.get(level)
    
    if not level_data:
        return f"Error: No hay feedback configurado en el curriculum para el nivel '{level}'."
    
    # Caso 1: Feedback simple (solo texto, sin distinción éxito/fracaso)
    if isinstance(level_data, str):
        return level_data
        
    # Caso 2: Feedback estructurado (diccionario con keys success/failure)
    return level_data.get(key, "Feedback no configurado para este resultado.")

# --- FUNCIÓN PRINCIPAL DE EVALUACIÓN ---

def evaluate_task(task: dict, student_answer: any, student_level: str):
    """
    Evalúa una tarea específica.
    
    Entradas:
        task (dict): El objeto de la tarea completo (del curriculum).
        student_answer (any): La respuesta del alumno (texto o lista).
        student_level (str): El nivel del alumno para esta estrategia (ej. "nivel_1").
        
    Retorna:
        dict: {
            "score": float, 
            "reasoning": str, 
            "feedback_shown": str
        }
    """
    task_id = task.get('task_id', 'Desconocida')
    print(f"\n[Motor Evaluación] Task: {task_id} | Type: {task.get('task_type')} | Level: {student_level}")
    
    correct_answer = task.get("correct_answer")
    score = 0.0
    reasoning = ""
    
    # ==============================================================================
    # RUTA B: EVALUACIÓN OBJETIVA (Python Puro)
    # Se activa si hay una 'correct_answer' definida o el tipo es 'objetivo'
    # ==============================================================================
    if task.get("task_type") == "objetivo" or correct_answer:
        print("   -> Modo: Objetivo (Comparación directa)")
        
        # 1. Normalizar respuestas a listas para comparación uniforme
        if isinstance(correct_answer, list):
            correct_list = [str(x).strip().lower() for x in correct_answer]
        else:
            correct_list = [str(correct_answer).strip().lower()]
            
        if isinstance(student_answer, list):
            student_list = [str(x).strip().lower() for x in student_answer]
        else:
            student_list = [str(student_answer).strip().lower()]
            
        # 2. Calcular Aciertos
        total_items = len(correct_list)
        matches = 0
        
        # Iteramos sobre las respuestas esperadas
        for i in range(total_items):
            # Verificamos si el alumno dio respuesta para este índice
            if i < len(student_list):
                if student_list[i] == correct_list[i]:
                    matches += 1
        
        # 3. Calcular Nota Proporcional (0-10)
        if total_items > 0:
            score = (matches / total_items) 
        else:
            score = 0.0
            
        score = round(score, 2)
        reasoning = f"Auto-corregido por Python. Aciertos: {matches}/{total_items}. Nota proporcional."
    elif task.get("task_type") == "interactivo":
        # Extraemos la nota del texto de respuesta automático
        # Esperamos formato: "Aciertos puros: X/Y"
        import re
        match = re.search(r"Aciertos puros: (\d+)/(\d+)", str(student_answer))
        if match:
            hits = int(match.group(1))
            total = int(match.group(2))
            score = hits / total if total > 0 else 0
            reasoning = f"Auto-corregido por Python. Aciertos: {hits}/{total}. Nota proporcional."
        else:
            score = 0.0
            reasoning = "No se pudo extraer la puntuación de la respuesta automática."
    # ==============================================================================
    # RUTA A: EVALUACIÓN SUBJETIVA (LLM + OCR)
    # Se activa para tareas abiertas, resúmenes, etc.
    # ==============================================================================
    else:
        print("   -> Modo: Subjetivo (LLM)")
        
        # 1. Construcción del Contexto (Texto + Imagen/OCR)
        context_parts = []
        
        # A) Texto oficial de la tarea
        raw_text = task.get("task_text")
        if raw_text:
            context_parts.append(raw_text)
        
        # B) Imagen (OCR)
        images = task.get("task_image")
        if isinstance(images, str):
            images = [images]
        if images:
            for image_path in images:
                print(f"      [OCR] Procesando imagen: {image_path}")
                ocr_text = extract_text_from_image(image_path)
                context_parts.append(ocr_text)
        
        # Unificar contexto
        final_task_context = "\n\n".join(context_parts) if context_parts else "No se proporcionó texto ni imagen de contexto."

        # 2. Obtener la Rúbrica
        # Prioridad: Rúbrica específica en la tarea > Rúbrica general del mapa
        raw_rubric = task.get("rubric")
        if raw_rubric:
            # Si hay rúbrica específica en el curriculum
            if isinstance(raw_rubric, (dict, list)):
                rubrica_text = json.dumps(raw_rubric, indent=2, ensure_ascii=False)
            else:
                rubrica_text = str(raw_rubric)
        else:

            sub_strategy = task.get("sub_strategy", "general")
            rubrica_text = "Evaluar la coherencia y calidad general de la respuesta."

        # 3. Seleccionar el Enunciado (Prompt) específico por nivel
        prompts_dict = task.get("task_prompts", {})
        # Si no existe prompt para ese nivel, usa el genérico 'task_prompt'
        specific_prompt = prompts_dict.get(student_level, task.get("task_prompt", ""))
        
        # 4. Invocación al LLM
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm_evaluador
        
        try:
            print("      [LLM] Evaluando respuesta...")
            llm_result = chain.invoke({
                "task_text": final_task_context,
                "rubrica_text": rubrica_text,
                "task_prompt": specific_prompt,
                "student_answer": student_answer
            })
            score = round(llm_result.score, 2)
            reasoning = llm_result.reasoning
            
        except Exception as e:
            print(f"Error invocando al LLM: {e}")
            return None

    # ==============================================================================
    # SELECCIÓN DE FEEDBACK (Común)
    # ==============================================================================
    final_feedback = select_feedback(task, student_level, score)
    
    return {
        "score": score,
        "reasoning": reasoning,
        "feedback_shown": final_feedback
    }