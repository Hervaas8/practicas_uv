"""
evaluar_sintetico.py
--------------------
Evalúa el detector de plagio sobre el conjunto de datos sintético.
Compara múltiples LLMs en la Fase 2 (sin necesidad de argumentos en consola).
"""

import json
import csv
import random
from collections import defaultdict
from detector_plagio import (
    limpiar_texto,
    calcular_trigramas,
    calcular_levenshtein_palabras,
    verificar_plagio_llm,
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

ARCHIVO_VARIANTES = "respuestas_plagio.json"  # Tu archivo de entrada
ARCHIVO_SALIDA = "resultados_comparativa_llm.csv"  # Tu archivo de salida

UMBRAL = 0.25          # Umbral óptimo de la Fase 1
MIN_PALABRAS = 15
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

def split_test(pares: list[dict], seed: int = SEED, ratio_dev: float = RATIO_DEV) -> list[dict]:
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

    # Inicializamos el diccionario base
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

    # Inicializamos los campos dinámicos para los modelos
    for modelo in modelos:
        resultado[f"veredicto_{modelo}"] = None
        resultado[f"justificacion_{modelo}"] = None
        resultado[f"prediccion_{modelo}"] = "no_plagio" # Por defecto

    sim_max = 0
    if not es_par_corto:
        sim_tri = calcular_trigramas(palabras_orig, palabras_var)
        sim_lev = calcular_levenshtein_palabras(palabras_orig, palabras_var)
        sim_max = max(sim_tri, sim_lev)

        resultado["sim_trigramas"] = round(sim_tri * 100, 1)
        resultado["sim_levenshtein"] = round(sim_lev * 100, 1)
        resultado["sim_maxima"] = round(sim_max * 100, 1)
        resultado["pasa_filtro_matematico"] = bool(sim_max >= umbral)

    # Si es corto o pasa el filtro, evaluamos con TODOS los modelos indicados en la lista
    if es_par_corto or sim_max >= umbral:
        for modelo in modelos:
            analisis = verificar_plagio_llm("(evaluación)", original, variante, modelo=modelo)
            veredicto = analisis.get("veredicto")
            resultado[f"veredicto_{modelo}"] = veredicto
            resultado[f"justificacion_{modelo}"] = analisis.get("justificacion")
            resultado[f"prediccion_{modelo}"] = "plagio" if veredicto == "Posible Plagio" else "no_plagio"

    return resultado

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print("Cargando variantes...")
    variantes = cargar_variantes(ARCHIVO_VARIANTES)
    pares = construir_pares(variantes)
    print(f"  Total pares construidos: {len(pares)}")

    # NOTA: Para hacer pruebas comparativas en fase de diseño usa split_dev (60%).
    # Si este ya es el paso final, usa split_test (40%). 
    pares_eval = split_test(pares) 
    print(f"  Pares a evaluar: {len(pares_eval)}")

    print(f"\nAnalizando con umbral={UMBRAL}, min_palabras={MIN_PALABRAS}...")
    print(f"Modelos configurados: {', '.join(MODELOS_LLM)}\n")
    
    resultados = []
    for i, par in enumerate(pares_eval, 1):
        print(f"  [{i}/{len(pares_eval)}] {par['task_id']} / {par['perfil']} / {par['nivel']}...", end=" ", flush=True)
        res = analizar_par(par, UMBRAL, MIN_PALABRAS, MODELOS_LLM)
        resultados.append(res)
        print("✅ Analizado")

    # Construir cabeceras CSV dinámicamente según la lista interna
    campos_base = [
        "task_id", "perfil", "nivel", "etiqueta_real", 
        "palabras_original", "palabras_variante", "ruta",
        "sim_trigramas", "sim_levenshtein", "sim_maxima", "pasa_filtro_matematico"
    ]
    
    campos_modelos = []
    for modelo in MODELOS_LLM:
        campos_modelos.extend([f"prediccion_{modelo}", f"veredicto_{modelo}", f"justificacion_{modelo}"])
    
    campos_finales = campos_base + campos_modelos + ["respuesta_original", "respuesta_variante"]

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos_finales)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n✅ Resultados guardados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()