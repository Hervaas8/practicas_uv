# **Archivo: probar_rutas_evaluacion.py (VERSIÓN UNIFICADA)**

import json
from configuracion_llm import (
    get_llm_evaluador, 
    get_retriever
)
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- 1. Definir el NUEVO Prompt Universal ---

EVALUATION_PROMPT_TEMPLATE = """
Eres un profesor experto y un evaluador justo. Tu tarea es evaluar la respuesta de un alumno que esta en educación secundaria.

**Contexto de la Asignatura (La Verdad Fáctica):**
---
{contexto_asignatura}
---

**Criterios de la Rúbrica (Cómo Evaluar):**
---
{criterios_rubrica}
---

**Pregunta de la Tarea:**
{task}

**Respuesta del Alumno:**
{answer}

**Clave de Respuestas Correctas (si aplica):**
{correct_answer}

**Instrucciones de Evaluación:**
1.  **Evaluar:** Compara la "Respuesta del Alumno" con el "Contexto" y la "Rúbrica".
2.  **Calcular Puntuación (Score):**
    * **Si se proporciona una 'Clave de Respuestas'**:Compara la "Respuesta del Alumno" con la "Clave de Respuestas". Tu puntuación DEBE ser la proporción de aciertos (aciertos / total) * 10.
    * **Si NO se proporciona una 'Clave de Respuestas' (tarea subjetiva)**: Asigna una puntuación de 0 a 10 basándote en la rúbrica.
3.  **Generar Feedback para el Alumno:**
    * Redacta un feedback amigable y constructivo.
    * **Para tareas con clave (V/F, opcion multiple, etc.)**: Explica SÓLO las respuestas incorrectas, usando el "Contexto" para justificar por qué la respuesta correcta es la correcta.
    * **Para tareas subjetivas (Resumen, etc.)**: Evalúa el rendimiento general contra la "Rúbrica".

**Instrucciones de Salida:**
Devuelve un JSON estructurado (LLMEvaluationOutput) con 'score', 'reasoning_for_professor' y 'student_feedback'.
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def evaluate_task_llm(task_details: dict, student_answer: any, config: dict):
    """
    Ejecuta la cadena de evaluación universal (RAG + LLM) para TODAS las tareas.
    """
    print(f"\n--- [Ruta Universal: Evaluador LLM] ---")
    
    # 1. Definir la cadena de RAG (Solo para contexto)
    chain_contexto = (
        (lambda x: str(x["answer"]) + " " + x["task"])
        | config["retriever_contexto"] 
        | format_docs
    )
    
    # 2. Obtener la rúbrica del mapa
    rubrica_texto = config["mapa_rubricas"].get(task_details['task_type'], "Evaluar por precisión y claridad.")
    
    # 3. Definir el prompt y la cadena completa
    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
    
    cadena_completa = (
        RunnablePassthrough.assign(
            contexto_asignatura=chain_contexto,
            criterios_rubrica=lambda x: rubrica_texto,
            # Pasamos la clave de respuestas al prompt
            correct_answer=lambda x: str(x.get("correct_answer", "N/A (Subjetivo)"))
        )
        | prompt
        | config["llm_evaluador"]
    )
    
    print(f"Invocando LLM para tarea: {task_details['task_type']}...")
    try:
        resultado = cadena_completa.invoke({
            "answer": student_answer,
            "task": task_details['task_prompt'],
            "task_type": task_details['task_type'],
            "correct_answer": task_details.get("correct_answer") # Pasa la clave
        })
        print(student_answer)
        print(task_details.get("correct_answer"))
        return resultado
    except Exception as e:
        print(f"Error invocando LLM: {e}")
        return None



# --- 3. Simulación del Agente (Router Unificado) ---

def main():
    print("--- Cargando todos los componentes (Lógica UNIFICADA)... ---")
    
    try:
        with open("curriculum.json", "r", encoding="utf-8-sig") as f:
            curriculum = json.load(f)
        with open("map_json_todos.json", "r", encoding="utf-8-sig") as f:
            mapa_rubricas = json.load(f)
        
        config = {
            "llm_evaluador": get_llm_evaluador(),
            "retriever_contexto": get_retriever("ciencias", "sesion1"),
            "mapa_rubricas": mapa_rubricas
        }
    except Exception as e:
        print(f"Error fatal al cargar configuración: {e}")
        return
    """
    # --- PRUEBA 1: Tarea Subjetiva (Resumen) ---
    print("\n\n--- INICIANDO PRUEBA 1: TAREA SUBJETIVA ('resumen') ---")
    task_resumen = curriculum["ciencias"]["sesion1"][0] # Tarea de Resumen
    respuesta_resumen = ("El texto dice textualmente 'La tubería forzada es de acero'. "
                         "Yo creo que es impresionante.")
    
    # El Router ahora es universal
    if task_resumen['task_type'] in mapa_rubricas:
        resultado_1 = evaluate_task_llm(task_resumen, respuesta_resumen, config)
    else:
        print(f"ERROR: Tarea '{task_resumen['task_type']}' no encontrada en rubricas_map.json")
        resultado_1 = None
        
    print("\n**Resultado Final (Prueba 1):**")
    print(resultado_1)
    """

    # --- PRUEBA 2: Tarea 'Auto-Corregible' (V/F) con error ---
    print("\n\n--- INICIANDO PRUEBA 2: TAREA ('v_o_f') con error ---")
    task_vof = curriculum["ciencias"]["sesion1"][2] # Tarea de V/F
    respuesta_vof_error = ["V", "V", "V","F", "F", "V", "V"] # Respuesta con 2 errores
    
    if task_vof['task_type'] in mapa_rubricas:
        resultado_2 = evaluate_task_llm(task_vof, respuesta_vof_error, config)
    else:
        print(f"ERROR: Tarea '{task_vof['task_type']}' no encontrada en rubricas_map.json")
        resultado_2 = None
        
    print("\n**Resultado Final (Prueba 2):**")
    print(resultado_2)

if __name__ == "__main__":
    main()