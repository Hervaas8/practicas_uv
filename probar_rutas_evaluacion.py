# **Archivo: probar_rutas_evaluacion.py (VERSIÓN PERFIL DE ALUMNO)**

import json
import yaml
import sys
from configuracion_llm import get_llm_evaluador
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate

# Importamos la utilidad de OCR
try:
    from utils_ocr import extract_text_from_image
except ImportError:
    print("Advertencia: No se encontró 'utils_ocr.py'.")
    def extract_text_from_image(path): return "[ERROR: Módulo OCR no encontrado]"

# --- CARGAR CONFIGURACIÓN ---
try:
    print("Cargando archivos de configuración...")
    with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_TEMPLATE = prompts["eval_subjective_prompt"]
    
    # Nota: Ya no usamos rubricas_map.json porque están en el curriculum
    
    with open("tareas.json", "r", encoding="utf-8-sig") as f:
        curriculum = json.load(f)
        
    llm_evaluador = get_llm_evaluador()
    print("--- Configuración Cargada Correctamente ---")

except Exception as e:
    print(f"Error crítico al cargar configuración: {e}")
    sys.exit(1)

# --- FUNCIÓN AUXILIAR: SELECCIONAR FEEDBACK ---
def select_feedback(task: dict, level: str, score: float):
    is_success = score >= 0.5
    key = "success" if is_success else "failure"
    feedbacks = task.get("feedback_by_level", {})
    level_data = feedbacks.get(level)
    
    if not level_data:
        return f"Error: No hay feedback configurado para el nivel '{level}'."
    
    if isinstance(level_data, str):
        return level_data
        
    return level_data.get(key, "Feedback no configurado.")

# --- FUNCIÓN DE EVALUACIÓN UNIVERSAL ---
def evaluate_task(task: dict, student_answer: any, student_level: str):
    """
    Evalúa la tarea usando el nivel específico proporcionado.
    """
    task_id = task.get('task_id', 'Desconocida')
    strategy = task.get('strategy', 'General')
    
    print(f"\nEvaluating Task: {task_id}")
    print(f"   -> Estrategia Detectada: {strategy.upper()}")
    print(f"   -> Nivel Aplicado al Alumno: {student_level.upper()}")
    
    correct_answer = task.get("correct_answer")
    score = 0.0
    reasoning = ""
    
    # --- RUTA B: EVALUACIÓN OBJETIVA (Python Puro) ---
    if task.get("task_type") == "objetivo" or correct_answer:
        print("-> Modo: Objetivo (Python - Proporcional)")
        
        # 1. Normalizar respuestas a listas para comparación uniforme
        # Si correct_answer es "a", lo convertimos a ["a"]
        if isinstance(correct_answer, list):
            correct_list = [str(x).strip().lower() for x in correct_answer]
        else:
            correct_list = [str(correct_answer).strip().lower()]
            
        # Igual para el alumno
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
        
        # 3. Calcular Nota Proporcional (Regla de tres)
        if total_items > 0:
            score = (matches / total_items) * 10.0
        else:
            score = 0.0
            
        score = round(score, 2) # Redondeo a 2 decimales
        reasoning = f"Auto-corregido por Python. Aciertos: {matches}/{total_items}. Nota proporcional."
    # --- RUTA A: EVALUACIÓN SUBJETIVA (LLM) ---
    else:
        print("   -> Modo: Subjetivo (LLM)")
        context_parts = []
        
        # Contexto Texto + Imagen
        raw_text = task.get("task_text")
        if raw_text: context_parts.append(f"--- TEXTO DEL ENUNCIADO ---\n{raw_text}")
        
        image_path = task.get("task_image")
        if image_path:
            print(f"      [OCR] Procesando imagen: {image_path}")
            ocr_text = extract_text_from_image(image_path)
            context_parts.append(f"--- CONTENIDO VISUAL (OCR) ---\n{ocr_text}")
        
        final_task_context = "\n\n".join(context_parts) if context_parts else "No hay contexto."

        # Rúbrica desde Curriculum
        raw_rubric = task.get("rubric")
        rubrica_text = json.dumps(raw_rubric, indent=2, ensure_ascii=False) if isinstance(raw_rubric, (dict, list)) else str(raw_rubric)

        # Enunciado Específico por Nivel
        # Aquí usamos el student_level que nos han pasado
        prompts_dict = task.get("task_prompts", {})
        specific_prompt = prompts_dict.get(student_level, task.get("task_prompt", ""))
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm_evaluador
        
        try:
            print("      [LLM] Invocando evaluación...")
            llm_result = chain.invoke({
                "task_text": final_task_context,
                "rubrica_text": rubrica_text,
                "task_prompt": specific_prompt,
                "student_answer": student_answer
            })
            score = llm_result.score
            reasoning = llm_result.reasoning
        except Exception as e:
            print(f"Error invocando al LLM: {e}")
            return None

    # --- SELECCIÓN DE FEEDBACK ---
    final_feedback = select_feedback(task, student_level, score)
    
    return {
        "score": score,
        "reasoning": reasoning,
        "feedback_shown": final_feedback
    }

# --- MAIN DE PRUEBA (SIMULACIÓN DEL AGENTE) ---

if __name__ == "__main__":
    try:
        tasks = curriculum["ciencias"]["sesion1"]
    except KeyError:
        print("Error: No se encontró 'ciencias -> sesion1'")
        sys.exit(1)
    
    # Perfil simulado del alumno
    student_profile = {
        "id": "alumno_test",
        "competency_levels": {
            "navegar": "nivel_1",
            "integrar": "nivel_2",
            "evaluar": "nivel_2"
        }
    }
    
    print("\n========================================")
    print(" INICIANDO BATERÍA DE PRUEBAS (Tareas 2, 3 y 4)")
    print("========================================")

    # --- PRUEBA 1: Tarea 2 (Resumen - Subjetiva) - BIEN HECHO ---
    print("\n--- CASO 1: Tarea 2 (Resumen) - Respuesta Correcta ---")
    try:
        t2 = next(t for t in tasks if t["task_id"] == "cien_s1_t02")
        print(f"Tarea: {t2['task_id']} ({t2['description']})")
        
        # Simulamos un buen resumen que cubre los 4 puntos de la rúbrica
        resp_bien = (
            "Los textos tratan sobre la polémica construcción de un centro comercial en la Sierra Calderona. "
            "Como ventajas, el Ayuntamiento destaca la creación de empleo y el turismo. "
            "Como desventajas, los ecologistas señalan el daño al ecosistema y el tráfico. "
            "En conclusión, ambas noticias reflejan el conflicto entre desarrollo económico y protección ambiental en la zona."
        )
        
        # Usamos el nivel de competencia de 'integrar'
        nivel_alumno = student_profile["competency_levels"]["integrar"]
        
        res2 = evaluate_task(t2, resp_bien, nivel_alumno)
        if res2:
            print(f"   -> Score: {res2['score']}")
            print(f"   -> Razón LLM: {res2['reasoning']}")
            print(f"   -> Feedback ({nivel_alumno}): {res2['feedback_shown']}")
    except StopIteration:
        print("Tarea 2 no encontrada.")

    # --- PRUEBA 2: Tarea 3 (Objetiva) - CORRECTA ---
    print("\n--- CASO 2: Tarea 3 (Navegar) - Respuesta Correcta ---")
    try:
        t3 = next(t for t in tasks if t["task_id"] == "cien_s1_t03")
        print(f"Tarea: {t3['task_id']} ({t3['description']})")
        
        # La respuesta correcta es 'a' (@NaturaViva)
        resp_correcta = "a"
        
        nivel_alumno = student_profile["competency_levels"]["navegar"]
        
        res3 = evaluate_task(t3, resp_correcta, nivel_alumno)
        print(f"   -> Score: {res3['score']}")
        print(f"   -> Feedback ({nivel_alumno}): {res3['feedback_shown']}")
    except StopIteration:
        print("Tarea 3 no encontrada.")

    # --- PRUEBA 3: Tarea 4 (Objetiva) - INCORRECTA ---
    print("\n--- CASO 3: Tarea 4 (Integrar) - Respuesta Incorrecta ---")
    try:
        t4 = next(t for t in tasks if t["task_id"] == "cien_s1_t04")
        print(f"Tarea: {t4['task_id']} ({t4['description']})")
        
        # La correcta es 'h', enviamos 'e' para fallar
        resp_incorrecta = "e" 
        
        nivel_alumno = student_profile["competency_levels"]["integrar"]
        
        res4 = evaluate_task(t4, resp_incorrecta, nivel_alumno)
        print(f"   -> Score: {res4['score']}")
        print(f"   -> Razón: {res4['reasoning']}")
        print(f"   -> Feedback ({nivel_alumno}): {res4['feedback_shown']}")
    except StopIteration:
        print("Tarea 4 no encontrada.")