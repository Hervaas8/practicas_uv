# **Archivo: probar_rutas_evaluacion.py (VERSIÓN HÍBRIDA)**

import json
import yaml
import sys
from configuracion_llm import (
    get_llm_evaluador, 
    get_llm, # <-- Importamos el LLM estándar
    get_retriever
)
from esquemas import LLMEvaluationOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    if not prompts or "eval_subjective_prompt" not in prompts or "eval_remedial_prompt" not in prompts:
        print("\n--- ERROR CRÍTICO ---")
        print("prompts.yaml está malformado o faltan claves ('eval_subjective_prompt', 'eval_remedial_prompt').")
        print("Asegúrate de que el archivo YAML está bien formateado.")
        sys.exit(1) # Detiene el script
        
    EVALUATION_PROMPT_TEMPLATE = prompts["eval_subjective_prompt"]
    REMEDIAL_FEEDBACK_PROMPT = prompts["eval_remedial_prompt2"]
    
    print("Prompts cargados desde prompts.yaml")

except FileNotFoundError:
    print("\n--- ERROR CRÍTICO ---")
    print("No se encontró el archivo 'prompts.yaml'. Asegúrate de que el archivo existe.")
    sys.exit(1) # Detiene el script
except yaml.YAMLError as e:
    print("\n--- ERROR CRÍTICO EN prompts.yaml ---")
    print("El archivo YAML tiene un error de formato (revisa la indentación y los caracteres especiales).")
    print(f"Detalle del error: {e}")
    sys.exit(1) # Detiene el script
except Exception as e:
    print(f"Error inesperado al cargar prompts: {e}")
    sys.exit(1)

def format_docs(docs: list[Document]) -> str:
    """
    Convierte la lista de documentos RAG en un string limpio y legible.
    """
    if not docs:
        return "--- SIN CONTEXTO RECUPERADO ---"
    
    clean_chunks = []
    for doc in docs:
        # --- PASO DE LIMPIEZA DEL TEXTO ---
        dirty_text = doc.page_content
        text_no_weird_spaces = dirty_text.replace('\xa0', ' ')
        text_no_newlines = text_no_weird_spaces.replace('\n', ' ')
        words = text_no_newlines.split()
        clean_text = ' '.join(words)
        # --- FIN DEL PASO DE LIMPIEZA ---
        
        clean_chunks.append(clean_text)
        
    # Devolvemos todos los chunks limpios, separados por un doble salto de línea
    return "\n\n".join(clean_chunks)
# --- 1. RUTA A: Evaluador Subjetivo (LLM lo hace todo) ---
# (Esta función es la misma que la de tu versión simplificada)

def evaluate_subjective_task(task_details: dict, student_answer: str, config: dict):
    print(f"\n--- [Ruta A: Evaluador Subjetivo (LLM)] ---")
    
    chain_contexto = (
        (lambda x: str(x["answer"]) + " " + x["task"])
        | config["retriever_contexto"] 
        | format_docs
    )
    
    rubrica_texto = config["mapa_rubricas"].get(task_details['task_type'], "Evaluar por precisión y claridad.")
    
    prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
    
    cadena_completa = (
        RunnablePassthrough.assign(
            contexto_asignatura=chain_contexto,
            criterios_rubrica=lambda x: rubrica_texto 
        )
        | prompt
        | config["llm_evaluador"] # <-- El LLM ESTRUCTURADO
    )
    
    print(f"Invocando LLM para tarea: {task_details['task_type']}...")
    try:
        return cadena_completa.invoke({
            "answer": student_answer,
            "task": task_details['task_prompt'],
            "task_type": task_details['task_type'],
        })
    except Exception as e:
        print(f"Error invocando LLM: {e}")
        return None


def evaluate_autogradable_task(task_details: dict, student_answer: any, student_group: str, config: dict):
    """
    (RUTA B) Usa Python puro para el score. NO llama al LLM.
    Genera un feedback TÉCNICO para el informe final del profesor.
    """
    print(f"\n--- [Ruta B: Evaluador Automático (Solo Python) | Grupo: {student_group}] ---")
    
    correct_answer = task_details.get("correct_answer")
    task_type = task_details.get("task_type")
    
    if correct_answer is None:
        return LLMEvaluationOutput(
            score=0, 
            reasoning_for_professor=f"Error: Tarea '{task_type}' sin 'correct_answer'",
            student_feedback="Error de configuración de tarea."
        )

    # --- Paso 1: Cálculo de Score (Python Puro) ---
    print("Paso 1: Calculando score con Python...")
    
    student_ans_list = student_answer if isinstance(student_answer, list) else [student_answer]
    correct_ans_list = correct_answer if isinstance(correct_answer, list) else [correct_answer]
    
    total = len(correct_ans_list)
    if total == 0:
        return LLMEvaluationOutput(score=0, reasoning_for_professor="Clave de respuestas vacía.", student_feedback="Error de tarea.")

    aciertos = 0
    errores_detectados = []
    
    for i in range(total):
        is_correct = False
        if i < len(student_ans_list) and str(student_ans_list[i]).upper() == str(correct_ans_list[i]).upper():
            aciertos += 1
            is_correct = True
        
        if not is_correct:
            # Preparamos el error para el informe del profesor
            respuesta_dada = student_ans_list[i] if i < len(student_ans_list) else "N/A"
            errores_detectados.append(f"Q#{i+1} (Respuesta: '{respuesta_dada}', Correcta: '{correct_ans_list[i]}')")

    score_bruto = (aciertos / total) * 10
    score = round(score_bruto, 2)
    reasoning = f"Aciertos: {aciertos} de {total}. Tarea auto-corregida por Python."
    print(f"Paso 1 (Python) completado. Score: {score}")

    # --- Paso 2: Generación de Feedback (Python Puro - TÉCNICO) ---
    print("Paso 2: Generando feedback técnico (Python).")
    
    if aciertos == total:
        feedback = f"Respuesta perfecta. ({aciertos}/{total} aciertos)."
    else:
        feedback = f"Respuesta incorrecta. ({aciertos}/{total} aciertos). Errores: {', '.join(errores_detectados)}"

    # Devolvemos el objeto de evaluación estándar
    return LLMEvaluationOutput(
        score=score,
        reasoning_for_professor=reasoning,
        student_feedback=feedback # Este es el feedback técnico para el informe final
    )

# --- 3. Simulación del Agente (El Router Híbrido) ---

def main():
    print("--- Cargando todos los componentes (Lógica HÍBRIDA)... ---")
    
    try:
        with open("curriculum.json", "r", encoding="utf-8-sig") as f:
            curriculum = json.load(f)
        with open("rubricas_map.json", "r", encoding="utf-8-sig") as f:
            mapa_rubricas = json.load(f)
        
        config = {
            "llm": get_llm(), # LLM Estándar
            "llm_evaluador": get_llm_evaluador(), # LLM Estructurado (Ruta A)
            "retriever_contexto": get_retriever("ciencias", "sesion1"),
            "mapa_rubricas": mapa_rubricas
        }
    except Exception as e:
        print(f"Error fatal al cargar configuración: {e}")
        return


    # --- PRUEBA 1: Tarea Subjetiva (Resumen) ---
    #print("\n\n--- INICIANDO PRUEBA 1: TAREA SUBJETIVA ('resumen') ---")
    #task_resumen = curriculum["ciencias"]["sesion1"][0]
    #respuesta_resumen = "Las presas son muros. El agua pasa por un tubo."
    
    # Lógica del Router
    #if task_resumen['task_type'] in mapa_rubricas:
    #   resultado_1 = evaluate_subjective_task(task_resumen, respuesta_resumen, config)
    #else:
   #     resultado_1 = evaluate_autogradable_task(task_resumen, respuesta_resumen, config)
   # print("\n**Resultado Final (Prueba 1):**")
   # print(resultado_1)


    # --- PRUEBA 2: Tarea Objetiva (V/F) con error ---
    
    print("\n\n--- INICIANDO PRUEBA 2: TAREA OBJETIVA ('v_o_f') con error ---")
    task_vof = curriculum["ciencias"]["sesion1"][2] # Tarea de V/F (7 preguntas)
    respuesta_vof_error = ["V", "V", "V", "F", "F", "V", "V"] # 2 errores (#1 y #5)
    
    if task_vof['task_type'] in mapa_rubricas:
        resultado_2 = evaluate_subjective_task(task_vof, respuesta_vof_error, config)
    else:
        resultado_2 = evaluate_autogradable_task(task_vof, respuesta_vof_error, config)
    print("\n**Resultado Final (Prueba 2):**")
    print(resultado_2)
    
    """
    # --- PRUEBA 3: Tarea Objetiva (V/F) perfecta ---
    print("\n\n--- INICIANDO PRUEBA 3: TAREA OBJETIVA ('v_o_f') perfecta ---")
    task_vof_perfecta = curriculum["ciencias"]["sesion1"][2]
    respuesta_vof_perfecta = ["F", "V", "V", "F", "V", "V", "V"] # Respuesta perfecta
    
    if task_vof_perfecta['task_type'] in mapa_rubricas:
        resultado_3 = evaluate_subjective_task(task_vof_perfecta, respuesta_vof_perfecta, config)
    else:
        resultado_3 = evaluate_autogradable_task(task_vof_perfecta, respuesta_vof_perfecta, config)
    print("\n**Resultado Final (Prueba 3):**")
    print(resultado_3)

    """

    """
    # --- PRUEBA 4: Tarea Subjetiva (ventajas_desventajas) - Respuesta BUENA ---
    print("\n\n--- INICIANDO PRUEBA 4: TAREA SUBJETIVA ('ventajas_desventajas') BUENA ---")
    
    try:
        # Busca la tarea de ventajas/desventajas en el currículum
        task_ventajas = next(t for t in curriculum["ciencias"]["sesion1"] if t["task_type"] == "v_y_d")
        
        # Simulamos una respuesta BUENA que usa conceptos del RAG
        respuesta_ventajas_buena = """
        #Ventajas:Las presas duran entre 50 y 100 años.\\nNo contamina el medio ambiente.\\nNo es una fuente renovable.
        #Desventajas: Puede alterar los ecosistemas acuáticos.\\nLas presas reducen la calidad del oxígeno en el agua\\nLas turbinas son muy grandes.
    """
        
        # El Router la enviará a la Ruta A (Subjetiva)
        if task_ventajas['task_type'] in mapa_rubricas:
            resultado_4 = evaluate_subjective_task(task_ventajas, respuesta_ventajas_buena, config)
        else:
            resultado_4 = evaluate_autogradable_task(task_ventajas, respuesta_ventajas_buena, config) # No debería pasar
        
        print("\n**Resultado Final (Prueba 4):**")
        print(resultado_4)
    
    except StopIteration:
        print("\n**Error en Prueba 4: Tarea 'ventajas_desventajas' no encontrada en curriculum.json.**")
    except Exception as e:
        print(f"\n**Error en Prueba 4:** {e}")
    """
    """
    # --- PRUEBA 5: Tarea Subjetiva (ventajas_desventajas) - Respuesta MALA ---
    print("\n\n--- INICIANDO PRUEBA 5: TAREA SUBJETIVA ('ventajas_desventajas') MALA ---")
    
    try:
        task_ventajas = next(t for t in curriculum["ciencias"]["sesion1"] if t["task_type"] == "v_y_d")
        
        # Simulamos una respuesta MALA (inventada, con opiniones)
        respuesta_ventajas_mala = """
        #Ventajas: Es muy barata y se puede poner en cualquier río.
        #Desventajas: Yo creo que es fea y hace mucho ruido.
    """
        
        if task_ventajas['task_type'] in mapa_rubricas:
            resultado_5 = evaluate_subjective_task(task_ventajas, respuesta_ventajas_mala, config)
        else:
            resultado_5 = evaluate_autogradable_task(task_ventajas, respuesta_ventajas_mala, config)
        
        print("\n**Resultado Final (Prueba 5):**")
        print(resultado_5)
    
    except StopIteration:
         print("\n**Error en Prueba 5: Tarea 'ventajas_desventajas' no encontrada en curriculum.json.**")
    except Exception as e:
        print(f"\n**Error en Prueba 5:** {e}")
    """
if __name__ == "__main__":
    main()