import json
import time
import os
import pandas as pd
from sistema.modulos.motor_evaluacion import evaluate_task

from sistema.config import JSON_DIR, RESULTS_DIR

ARCHIVO_JSON = JSON_DIR / "evaluacion_1.json"
ARCHIVO_CSV_SALIDA = RESULTS_DIR / "resultados_benchmark.csv"

# Modelos disponibles en la UPV que vamos a comparar
MODELOS_A_EVALUAR = [
    "deepseek-r1:32b",
    "llama3.1:8b",
    "llama3.2:3b",
    "phi4-mini:3.8b",
    "qwen3-vl:30b",
    "poligpt-code",
    "poligpt"
]

MAX_REINTENTOS = 3
TIEMPO_ESPERA = 5 

def cargar_progreso_previo():
    """Lee el CSV si existe y devuelve un Set con las evaluaciones ya completadas y la lista de datos."""
    completados = set()
    resultados_globales = []
    
    if os.path.exists(ARCHIVO_CSV_SALIDA):
        try:
            df = pd.read_csv(ARCHIVO_CSV_SALIDA)
            # Rellenar los NaN con None para trabajar cómodos en diccionarios
            df = df.where(pd.notnull(df), None)
            resultados_globales = df.to_dict('records')
            
            for fila in resultados_globales:
                # Solo consideramos completadas las que tienen una Nota Global (no dieron error)
                if fila.get("Nota_Global") is not None:
                    clave = (fila["Task_ID"], fila["Perfil_Alumno"], fila["Modelo_Evaluador"])
                    completados.add(clave)
            print(f"Cargadas {len(completados)} evaluaciones exitosas previas.")
        except Exception as e:
            print(f"No se pudo cargar el CSV previo: {e}")
            
    return completados, resultados_globales

def procesar_benchmark():
    try:
        with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ARCHIVO_JSON}'.")
        return

    # Cargamos el historial para no repetir
    completados, resultados_globales = cargar_progreso_previo()

    for task_id, task_data in datos.items():
        print(f"\n--- Evaluando Tarea: {task_id} ---")
        
        task_mock = {
            "task_id": task_id,
            "task_type": task_data.get("tipo_tarea", "subjetivo"),
            "task_prompt": task_data.get("task_prompt", ""),
            "rubric": task_data.get("rubric", {}),
            "task_image": task_data.get("task_image", [])
        }
        
        respuestas = task_data.get("respuestas", [])

        for respuesta_obj in respuestas:
            perfil = respuesta_obj.get("perfil", "desconocido")
            respuesta_alumno = respuesta_obj.get("respuesta", "")

            for modelo in MODELOS_A_EVALUAR:
                # 1. Comprobar si ya lo hemos hecho
                if (task_id, perfil, modelo) in completados:
                    print(f"   Saltando perfil '{perfil}' con modelo '{modelo}' (Ya evaluado).")
                    continue
                
                print(f"   Analizando perfil '{perfil}' con modelo '{modelo}'...")
                
                resultado = None
                tiempo_transcurrido = 0.0

                for intento in range(MAX_REINTENTOS):
                    inicio_tiempo = time.time()
                    
                    resultado = evaluate_task(
                        task=task_mock, 
                        student_answer=respuesta_alumno, 
                        student_level="nivel_medio", 
                        override_model=modelo
                    )

                    fin_tiempo = time.time()
                    tiempo_transcurrido = round(fin_tiempo - inicio_tiempo, 2)

                    if resultado is not None:
                        break 
                    else:
                        print(f"      Fallo en el intento {intento + 1}/{MAX_REINTENTOS}. Reintentando en {TIEMPO_ESPERA}s...")
                        time.sleep(TIEMPO_ESPERA)

                # Buscar si ya existía una fila de un error previo para sobreescribirla, o crear una nueva
                fila_existente = next((f for f in resultados_globales if f["Task_ID"] == task_id and f["Perfil_Alumno"] == perfil and f["Modelo_Evaluador"] == modelo), None)
                
                if fila_existente is None:
                    fila_existente = {
                        "Task_ID": task_id,
                        "Perfil_Alumno": perfil,
                        "Modelo_Evaluador": modelo
                    }
                    resultados_globales.append(fila_existente)

                # Actualizar los datos de la fila
                fila_existente["Tiempo_Respuesta_s"] = tiempo_transcurrido
                
                if resultado:
                    fila_existente["Nota_Global"] = resultado["score"]
                    fila_existente["Max_Score"] = resultado["max_score"]
                    fila_existente["Razonamiento_LLM"] = resultado["reasoning"]
                    
                    scores_dict = resultado.get("scores_por_item", {})
                    if scores_dict:
                        valores_items = list(scores_dict.values())
                        fila_existente["Nota_Item_1"] = valores_items[0] if len(valores_items) > 0 else None
                        fila_existente["Nota_Item_2"] = valores_items[1] if len(valores_items) > 1 else None
                    completados.add((task_id, perfil, modelo))
                else:
                    fila_existente["Nota_Global"] = None
                    fila_existente["Razonamiento_LLM"] = f"Error técnico tras {MAX_REINTENTOS} intentos."

                # Guardado incremental: Actualizamos el CSV después de cada respuesta
                df = pd.DataFrame(resultados_globales)
                df.to_csv(ARCHIVO_CSV_SALIDA, index=False, encoding="utf-8-sig")

    print(f"\nResultados guardados en '{ARCHIVO_CSV_SALIDA}'.")

if __name__ == "__main__":
    procesar_benchmark()