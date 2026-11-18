"""
evaluar_sintetico.py
--------------------
Evalúa el detector de plagio sobre el conjunto de datos sintético.
Compara múltiples LLMs en la Fase 2 (sin necesidad de argumentos en consola).
Guarda el progreso iterativamente, permite reanudar y mide el tiempo de respuesta.
"""

import json
import csv
import random
import os
import time  # <-- AÑADIDO: Módulo para medir el tiempo
from collections import defaultdict
from sistema.modulos.detector_plagio import (
    limpiar_texto,
    calcular_trigramas,
    calcular_levenshtein_palabras,
    verificar_plagio_llm,
    plagiarism_prediction,
)

from sistema.config import JSON_DIR, RESULTS_DIR, PLAGIARISM_THRESHOLD, PLAGIARISM_MIN_WORDS

ARCHIVO_VARIANTES = JSON_DIR / "respuestas_plagio.json"
ARCHIVO_SALIDA = RESULTS_DIR / "resultados_comparativa_llm.csv"

UMBRAL = PLAGIARISM_THRESHOLD
MIN_PALABRAS = PLAGIARISM_MIN_WORDS
SEED = 42              
RATIO_DEV = 0.60       

NIVELES_PLAGIO = {"nivel_0", "nivel_1", "nivel_2", "nivel_3"}
NIVELES_NO_PLAGIO = {"nivel_4", "nivel_5"}

# Modelos que se utilizarán para la Fase 2
MODELOS_LLM = [
    "deepseek-r1:32b",
    "llama3.1:8b",
    "llama3.2:3b",
    "phi4-mini:3.8b",
    "qwen3-vl:30b",
    "poligpt-code",
    "poligpt"
]

# ==========================================
# CARGA Y PREPARACIÓN DE DATOS
# ==========================================

def cargar_variantes(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def construir_pares(variantes: dict) -> list[dict]:
    pares = []
    for task_id, perfiles in variantes.items():
        for perfil, datos in perfiles.items():
            original = datos.get("respuesta_original", "")
            vars_dict = datos.get("variantes") or {}
            for nivel, texto_variante in vars_dict.items():
                if not texto_variante:
                    continue  
                etiqueta = "plagio" if nivel in NIVELES_PLAGIO else "no_plagio"
                pares.append({
                    "task_id": task_id,
                    "perfil": perfil,
                    "nivel": nivel,
                    "respuesta_original": original,
                    "respuesta_variante": texto_variante,
                    "etiqueta_real": etiqueta,
                })
    return pares

def split_dev(pares, seed=SEED, ratio_dev=RATIO_DEV):
    """
    Devuelve la partición de dev (60%) de forma ESTRATIFICADA por 'perfil'.
    Garantiza la misma proporción de respuestas de cada nivel en test y dev.
    """
    tareas_por_perfil = defaultdict(set)
    for p in pares:
        tareas_por_perfil[p["perfil"]].add(p["task_id"])

    random.seed(seed)
    
    unidades_test = set()
    for perfil in sorted(tareas_por_perfil.keys()):
        tareas = sorted(list(tareas_por_perfil[perfil]))
        random.shuffle(tareas)
        n_dev = int(len(tareas) * ratio_dev)
        for task_id in tareas[n_dev:]:
            unidades_test.add((task_id, perfil))

    return [p for p in pares if (p["task_id"], p["perfil"]) in unidades_test]

# ==========================================
# LECTURA DE PROGRESO PREVIO
# ==========================================

def obtener_procesados(ruta_csv: str) -> set:
    """
    Lee el archivo CSV si existe y devuelve un conjunto con los IDs únicos
    (task_id, perfil, nivel) que ya han sido procesados.
    """
    procesados = set()
    if os.path.exists(ruta_csv):
        with open(ruta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_unico = (row["task_id"], row["perfil"], row["nivel"])
                procesados.add(id_unico)
    return procesados

# ==========================================
# ANÁLISIS DE CADA PAR
# ==========================================

def analizar_par(par: dict, umbral: float, min_palabras: int, modelos: list) -> dict:
    original = par["respuesta_original"]
    variante = par["respuesta_variante"]

    palabras_orig = limpiar_texto(original)
    palabras_var = limpiar_texto(variante)

    n_orig = len(palabras_orig)
    n_var = len(palabras_var)
    es_par_corto = n_orig < min_palabras or n_var < min_palabras

    resultado = {
        **par,
        "palabras_original": n_orig,
        "palabras_variante": n_var,
        "ruta": "llm_directo" if es_par_corto else "filtro_matematico",
        "pasa_filtro_matematico": "N/A" if es_par_corto else None,
        "sim_trigramas": None,
        "sim_levenshtein": None,
        "sim_maxima": None,
    }

    # Inicializamos los campos de cada modelo, incluyendo el tiempo
    for modelo in modelos:
        resultado[f"veredicto_{modelo}"] = None
        resultado[f"justificacion_{modelo}"] = None
        resultado[f"prediccion_{modelo}"] = "no_plagio" 
        resultado[f"tiempo_{modelo}"] = None  # <-- AÑADIDO: Campo para el tiempo

    sim_max = 0
    if not es_par_corto:
        sim_tri = calcular_trigramas(palabras_orig, palabras_var)
        sim_lev = calcular_levenshtein_palabras(palabras_orig, palabras_var)
        sim_max = max(sim_tri, sim_lev)

        resultado["sim_trigramas"] = round(sim_tri * 100, 1)
        resultado["sim_levenshtein"] = round(sim_lev * 100, 1)
        resultado["sim_maxima"] = round(sim_max * 100, 1)
        resultado["pasa_filtro_matematico"] = bool(sim_max >= umbral)

    if es_par_corto or sim_max >= umbral:
        for modelo in modelos:
            # <-- AÑADIDO: Iniciar cronómetro
            inicio = time.time()
            
            analisis = verificar_plagio_llm("(evaluación)", original, variante, modelo=modelo)
            
            # <-- AÑADIDO: Detener cronómetro y calcular duración en segundos
            fin = time.time()
            tiempo_segundos = round(fin - inicio, 2)
            
            veredicto = analisis.get("veredicto")
            resultado[f"veredicto_{modelo}"] = veredicto if analisis.get("status") != "error" else "Error técnico"
            resultado[f"justificacion_{modelo}"] = analisis.get("justificacion")
            resultado[f"prediccion_{modelo}"] = plagiarism_prediction(analisis)
            resultado[f"tiempo_{modelo}"] = tiempo_segundos  # <-- AÑADIDO: Guardar en el diccionario

    return resultado

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print("Cargando variantes...")
    variantes = cargar_variantes(ARCHIVO_VARIANTES)
    pares = construir_pares(variantes)
    print(f"  Total pares construidos: {len(pares)}")

    pares_eval = split_dev(pares) 
    print(f"  Pares a evaluar en total: {len(pares_eval)}")

    procesados = obtener_procesados(ARCHIVO_SALIDA)
    if procesados:
        print(f"  Encontrados {len(procesados)} pares ya procesados en el CSV. Se omitirán.")

    print(f"\nAnalizando con umbral={UMBRAL}, min_palabras={MIN_PALABRAS}...")
    print(f"Modelos configurados: {', '.join(MODELOS_LLM)}\n")
    
    campos_base = [
        "task_id", "perfil", "nivel", "etiqueta_real", 
        "palabras_original", "palabras_variante", "ruta",
        "sim_trigramas", "sim_levenshtein", "sim_maxima", "pasa_filtro_matematico"
    ]
    
    # AÑADIDO: Incluir la columna de tiempo en el CSV para cada modelo
    campos_modelos = []
    for modelo in MODELOS_LLM:
        campos_modelos.extend([
            f"prediccion_{modelo}", 
            f"veredicto_{modelo}", 
            f"tiempo_{modelo}",  # <-- AÑADIDO: Nueva columna
            f"justificacion_{modelo}"
        ])
    
    campos_finales = campos_base + campos_modelos + ["respuesta_original", "respuesta_variante"]

    archivo_existe = os.path.exists(ARCHIVO_SALIDA)
    
    with open(ARCHIVO_SALIDA, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_finales)
        
        if not archivo_existe or os.path.getsize(ARCHIVO_SALIDA) == 0:
            writer.writeheader()

        for i, par in enumerate(pares_eval, 1):
            id_unico = (par["task_id"], par["perfil"], par["nivel"])
            
            if id_unico in procesados:
                continue
            
            print(f"  [{i}/{len(pares_eval)}] {par['task_id']} / {par['perfil']} / {par['nivel']}...", end=" ", flush=True)
            
            res = analizar_par(par, UMBRAL, MIN_PALABRAS, MODELOS_LLM)
            
            writer.writerow(res)
            f.flush()  
            
            print("✅ Analizado y guardado")

    print(f"\n✅ Proceso completado con éxito. Resultados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()
