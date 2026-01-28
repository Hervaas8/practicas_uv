# **Archivo: probar_rutas_evaluacion.py (SIMPLIFICADO)**

import json
from configuracion_llm import (
    get_llm_evaluador, 
    get_retriever # <-- Ya no importamos get_retriever_sourcing
)
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- 1. Definir la Ruta A: Evaluador Subjetivo (RAG Simple + LLM) ---

# El prompt sigue siendo el mismo. Recibe el contexto y la rúbrica.
EVALUATION_PROMPT_TEMPLATE = """
Eres un profesor experto y un evaluador justo. Tu tarea es evaluar la respuesta de un alumno que esta cursando educación secundaria.Para ello, se te da información de la asignatura que usaras como fuente de conocimineto, una rúbrica de que criterios a tener en cuenta para este tipo de tarea, el enunciado de la tarea y la respuesta del alumno

**Contexto de la Asignatura:**
---
{contexto_asignatura}
---

**Criterios de la Rúbrica:**
---
{criterios_rubrica}
---

**Pregunta de la Tarea:**
{task}

**Respuesta del Alumno:**
{answer}

**Instrucciones de Salida:**
Debes generar una salida JSON estructurada (con el formato LLMEvaluationOutput) que contenga TRES campos:

1.  **score:** Una puntuación numérica final de 0 a 10.
2.  **reasoning_for_professor:** Tu justificación técnica de la nota para el profesor, 
    explicando cómo aplicaste los criterios de la rúbrica.
3.  **student_feedback:** El feedback redactado PARA EL ALUMNO. Debe ser un texto
    completo, amigable y listo para mostrar. En este texto, explícale qué
    hizo bien y qué falló, basándote en los **Criterios de la Rúbrica**.
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def evaluate_subjective_task(task_details: dict, student_answer: str, config: dict):
    """
    Ejecuta la cadena de evaluación (RAG Simple + LLM).
    La rúbrica ahora viene del config["mapa_rubricas"].
    """
    print(f"\n--- [Ruta A: Evaluador Subjetivo (LLM) - Lógica Simplificada] ---")
    
    # 1. Definir la cadena de RAG (Solo para contexto)
    chain_contexto = (
        (lambda x: x["answer"] + " " + x["task"])
        | config["retriever_contexto"] 
        | format_docs
    )
    
    # 2. Obtener la rúbrica (¡Es un lookup directo!)
    rubrica_texto = config["mapa_rubricas"].get(task_details['task_type'])
    if not rubrica_texto:
        # Fallback por si acaso
        rubrica_texto = "Evaluar basándose en la precisión y claridad general."
        print(f"Advertencia: No se encontró rúbrica para {task_details['task_type']}")

    # 3. Definir el prompt y la cadena completa
    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
    
    cadena_completa = (
        RunnablePassthrough.assign(
            contexto_asignatura=chain_contexto,
            # Asignamos el texto de la rúbrica directamente desde el JSON
            criterios_rubrica=lambda x: rubrica_texto 
        )
        | prompt
        | config["llm_evaluador"]
    )
    
    print(f"Invocando RAG (Contexto) para tarea: {task_details['task_type']}...")
    try:
        resultado = cadena_completa.invoke({
            "answer": student_answer,
            "task": task_details['task_prompt'],
            "task_type": task_details['task_type'],
        })
        return resultado
    except Exception as e:
        print(f"Error invocando LLM: {e}")
        return None

# --- 2. Definir la Ruta B: Evaluador Automático (Python) ---
# Esta función (evaluate_autogradable_task) se queda EXACTAMENTE IGUAL.
# ... (copiar la función 'evaluate_autogradable_task' de la respuesta anterior) ...
def evaluate_autogradable_task(task_details: dict, student_answer: any):
    print(f"\n--- [Ruta B: Evaluador Automático (Python)] ---")
    
    correct_answer = task_details.get("correct_answer")
    if correct_answer is None:
        return LLMEvaluationOutput(
            score=0, 
            reasoning_for_professor="Error: Tarea auto-corregible sin 'correct_answer' en curriculum.json",
            student_feedback="Ha ocurrido un error al evaluar tu tarea. Por favor, contacta al profesor."
        )

    score = 0.0
    reasoning = ""
    feedback = ""
    
    if task_details['task_type'] == "v_o_f":
        if isinstance(student_answer, list) and isinstance(correct_answer, list):
            total = len(correct_answer)
            aciertos = 0
            errores = []
            for i, ans in enumerate(student_answer):
                if i < total and str(ans).upper() == str(correct_answer[i]).upper():
                    aciertos += 1
                else:
                    errores.append(f"la afirmación #{i+1}")
            
            score = (aciertos / total) * 10
            reasoning = f"Aciertos: {aciertos} de {total}. Tarea auto-corregida."
            if aciertos == total:
                feedback = f"¡Respuesta perfecta! Has acertado las {total} afirmaciones."
            else:
                feedback = f"¡Buen intento! Has acertado {aciertos} de {total}. Revisa {', '.join(errores)}."
        else:
            reasoning = "La respuesta o la clave no están en formato de lista."
            feedback = "Tu respuesta no estaba en el formato esperado."

    return LLMEvaluationOutput(
        score=score,
        reasoning_for_professor=reasoning,
        student_feedback=feedback
    )

# --- 3. Simulación del Agente (Router y Pruebas) ---
# Esta función 'main' se queda EXACTAMENTE IGUAL,
# pero ahora cargará menos cosas en el 'config'.
def main():
    print("--- Cargando todos los componentes (Lógica Simplificada)... ---")
    
    try:
        with open("curriculum.json", "r", encoding="utf-8-sig") as f:
            curriculum = json.load(f)
        with open("rubricas_map.json", "r", encoding="utf-8-sig") as f:
            mapa_rubricas = json.load(f)
        
        config = {
            "llm_evaluador": get_llm_evaluador(),
            "retriever_contexto": get_retriever("ciencias", "sesion1"),
            # 'retriever_rubrica' ya no existe
            "mapa_rubricas": mapa_rubricas
        }
    except Exception as e:
        print(f"Error fatal al cargar configuración: {e}")
        return

    # --- PRUEBA 1: Tarea Subjetiva (Resumen) ---
    print("\n\n--- INICIANDO PRUEBA 1: TAREA SUBJETIVA ('resumen') ---")
    task_resumen = curriculum["ciencias"]["sesion1"][0] 
    respuesta_resumen = ("El texto dice textualmente 'La tubería forzada es de acero "
                      "y puede tener 9 metros o más de diámetro'. "
                      "Yo creo que eso es una locura de ingeniería y muy impresionante.")
    
    # Lógica del Router (sin cambios)
    if task_resumen['task_type'] in mapa_rubricas:
        resultado_1 = evaluate_subjective_task(task_resumen, respuesta_resumen, config)
    else:
        resultado_1 = evaluate_autogradable_task(task_resumen, respuesta_resumen)
    print("\n**Resultado Final (Prueba 1):**")
    print(resultado_1)

    print("\n\n--- INICIANDO PRUEBA 1: 'respuesta_corta' (BUENA) ---")
    
    # Buscamos la tarea en nuestro curriculum.json
    task_respuesta_corta = next(t for t in curriculum["ciencias"]["sesion1"] if t["task_type"] == "respuesta_corta")
    
    respuesta_buena = ("El agua almacenada pasa por la presa, su fuerza mueve "
                       "una turbina y esa turbina activa un generador que "
                       "crea la electricidad.")
    
    try:
        resultado_1 = evaluate_subjective_task(task_respuesta_corta, respuesta_buena, config)
        print("\n**Resultado Final (Prueba 1):**")
        print(resultado_1)
        
    except Exception as e:
        print(f"\n--- ERROR EN PRUEBA 1 ---")
        print(f"Asegúrate de que 'poligpt' funciona. Error: {e}")


""""
    # --- PRUEBA 2: Tarea Auto-Corregible (V/F) ---
    print("\n\n--- INICIANDO PRUEBA 2: TAREA AUTO-CORREGIBLE ('v_o_f') ---")
    task_vof = curriculum["ciencias"]["sesion1"][2] 
    respuesta_vof = ["F", "V", "V","F","V","V","V"] 
    
    if task_vof['task_type'] in mapa_rubricas:
        resultado_2 = evaluate_subjective_task(task_vof, respuesta_vof, config)
    else:
        resultado_2 = evaluate_autogradable_task(task_vof, respuesta_vof)
    print("\n**Resultado Final (Prueba 2):**")
    print(resultado_2)

    # --- PRUEBA 3: Tarea Auto-Corregible (V/F) con error ---
    print("\n\n--- INICIANDO PRUEBA 3: AUTO-CORREGIBLE (con error) ---")
    respuesta_vof_error = ["V", "V", "V","F","F","V","F"] # Respuesta con error
    
    if task_vof['task_type'] in mapa_rubricas:
        resultado_3 = evaluate_subjective_task(task_vof, respuesta_vof_error, config)
    else:
        resultado_3 = evaluate_autogradable_task(task_vof, respuesta_vof_error)
        
    print("\n**Resultado Final (Prueba 3):**")
    print(resultado_3)
"""
if __name__ == "__main__":
    main()