import json
import time
import pandas as pd
from motor_evaluacion import evaluate_task

# 1. Parámetros de configuración
ARCHIVO_JSON = "evaluacion_1.json" 
ARCHIVO_CSV_SALIDA = "resultados_benchmark.csv"

# Modelos disponibles en la UPV que vamos a comparar
MODELOS_A_EVALUAR = [
    "qwen3-vl:30b",
    "deepseek-r1:32b",
    "llama3.1:8b",
    "phi4-mini:3.8b",
    "poligpt-vision",
    "poligpt"
]

# Configuración de reintentos
MAX_REINTENTOS = 3
TIEMPO_ESPERA = 5 # Segundos de pausa antes de volver a intentarlo

def procesar_benchmark():
    # Cargar el dataset
    try:
        with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ARCHIVO_JSON}'.")
        return

    resultados_globales = []

    for task_id, task_data in datos.items():
        print(f"\n--- Evaluando Tarea: {task_id} ---")
        
        # Simular el objeto task como lo espera evaluate_task
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
                print(f"   Analizando perfil '{perfil}' con modelo '{modelo}'...")
                
                resultado = None
                tiempo_transcurrido = 0.0

                # --- BUCLE DE REINTENTOS PARA LA API ---
                for intento in range(MAX_REINTENTOS):
                    inicio_tiempo = time.time()
                    
                    # Ejecutar la evaluación
                    resultado = evaluate_task(
                        task=task_mock, 
                        student_answer=respuesta_alumno, 
                        student_level="nivel_medio", 
                        override_model=modelo
                    )

                    fin_tiempo = time.time()
                    tiempo_transcurrido = round(fin_tiempo - inicio_tiempo, 2)

                    if resultado is not None:
                        # Si ha ido bien, salimos del bucle de reintentos
                        break 
                    else:
                        print(f"      ⚠️ Fallo en el intento {intento + 1}/{MAX_REINTENTOS}. Reintentando en {TIEMPO_ESPERA}s...")
                        time.sleep(TIEMPO_ESPERA)
                # ---------------------------------------

                # Inicializar fila para el dataset
                fila = {
                    "Task_ID": task_id,
                    "Perfil_Alumno": perfil,
                    "Modelo_Evaluador": modelo,
                    "Tiempo_Respuesta_s": tiempo_transcurrido,
                    "Nota_Global": None,
                    "Max_Score": None,
                    "Nota_Item_1": None,
                    "Nota_Item_2": None,
                    "Razonamiento_LLM": ""
                }

                if resultado:
                    fila["Nota_Global"] = resultado["score"]
                    fila["Max_Score"] = resultado["max_score"]
                    fila["Razonamiento_LLM"] = resultado["reasoning"]
                    
                    # Desglosar los ítems del diccionario
                    scores_dict = resultado.get("scores_por_item", {})
                    if scores_dict:
                        valores_items = list(scores_dict.values())
                        if len(valores_items) > 0:
                            fila["Nota_Item_1"] = valores_items[0]
                        if len(valores_items) > 1:
                            fila["Nota_Item_2"] = valores_items[1]
                else:
                    # Si fracasó incluso tras los reintentos, dejamos constancia
                    fila["Razonamiento_LLM"] = f"Error técnico tras {MAX_REINTENTOS} intentos."
                
                resultados_globales.append(fila)

    # Convertir a DataFrame y guardar
    df = pd.DataFrame(resultados_globales)
    df.to_csv(ARCHIVO_CSV_SALIDA, index=False, encoding="utf-8-sig")
    print(f"\n✅ Benchmark completado con éxito. Resultados guardados en '{ARCHIVO_CSV_SALIDA}'.")

if __name__ == "__main__":
    procesar_benchmark()