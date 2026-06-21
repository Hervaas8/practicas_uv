"""
evaluar_clough.py
-----------------
Evalúa el detector de plagio sobre el corpus de Clough & Stevenson (CS11).
Compara cada documento contra el original y genera un CSV "ancho"
donde cada fila contiene los resultados de TODOS los modelos evaluados.
"""

import csv
import time
from pathlib import Path
from sistema.modulos.detector_plagio import (
    limpiar_texto,
    calcular_trigramas,
    calcular_levenshtein_palabras,
    verificar_plagio_llm,
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

from sistema.config import DATA_DIR, RESULTS_DIR

CORPUS_DIR = DATA_DIR / "clough"
OUTPUT_FILE = RESULTS_DIR / "resultados_clough_ancho.csv"

UMBRAL = 0.25
MIN_PALABRAS = 15

CATEGORIAS_PLAGIO = {"cut", "light"}
CATEGORIAS_NO_PLAGIO = {"non", "heavy"}

MODELOS_LLM = [
    "deepseek-r1:32b",
    "llama3.1:8b",
    "llama3.2:3b",
    "phi4-mini:3.8b",
    "qwen3-vl:30b",
    "poligpt-code",
    "poligpt"
]

# Definición exacta de las columnas solicitadas
CAMPOS_CSV = [
    "task_id", "perfil", "nivel", "etiqueta_real", 
    "palabras_original", "palabras_variante", "ruta", 
    "sim_trigramas", "sim_levenshtein", "sim_maxima", "pasa_filtro_matematico"
]
# Añadir columnas dinámicas por cada modelo
for m in MODELOS_LLM:
    CAMPOS_CSV.extend([f"prediccion_{m}", f"veredicto_{m}", f"tiempo_{m}", f"justificacion_{m}"])
    
CAMPOS_CSV.extend(["respuesta_original", "respuesta_variante"])

# ==========================================
# CARGA DEL CORPUS Y UTILIDADES
# ==========================================

def cargar_corpus(corpus_dir: str) -> list[dict]:
    corpus_path = Path(corpus_dir)
    info_path = corpus_path / "file_information.csv"

    documentos = []
    with open(info_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documentos.append({
                "filename": row["File"],
                "tarea": row["Task"],
                "categoria": row["Category"],
            })

    originales = {
        d["tarea"]: corpus_path / d["filename"]
        for d in documentos if d["categoria"] == "orig"
    }

    pares = []
    for doc in documentos:
        if doc["categoria"] == "orig":
            continue

        tarea = doc["tarea"]
        ruta_doc = corpus_path / doc["filename"]
        ruta_orig = originales.get(tarea)

        if not ruta_orig or not ruta_orig.exists() or not ruta_doc.exists():
            continue

        texto_orig = ruta_orig.read_text(encoding="utf-8", errors="ignore").strip()
        texto_doc = ruta_doc.read_text(encoding="utf-8", errors="ignore").strip()

        etiqueta = "plagio" if doc["categoria"] in CATEGORIAS_PLAGIO else "no_plagio"

        pares.append({
            "task_id": doc["filename"],
            "perfil": tarea,
            "nivel": doc["categoria"],
            "etiqueta_real": etiqueta,
            "texto_original": texto_orig,
            "texto_documento": texto_doc,
        })

    return pares

# --- NUEVO: Regla estricta de validación ---
def modelo_es_valido(fila_previa: dict, modelo: str) -> bool:
    """
    Devuelve False (necesita reprocesar) si:
    - La casilla está vacía.
    - Contiene la palabra "error" Y el nombre del modelo.
    """
    if not fila_previa:
        return False
        
    veredicto = str(fila_previa.get(f"veredicto_{modelo}", "")).strip().lower()
    justificacion = str(fila_previa.get(f"justificacion_{modelo}", "")).strip().lower()
    
    # Condición 1: Estar vacío
    if not veredicto:
        return False
        
    # Unimos ambos textos para buscar en cualquier lugar de las respuestas
    texto_completo = veredicto + " " + justificacion
    
    # Condición 2: Contener "error" Y el nombre del modelo
    if "error" in texto_completo:
        # Extraemos la base del modelo (ej: de 'qwen3-vl:30b' saca 'qwen3' o 'qwen')
        # Esto permite que coincida tanto si pusiste "error qwen3-vl:30b" como "error qwen"
        modelo_base = modelo.split("-")[0].split(":")[0].lower()
        
        if modelo.lower() in texto_completo or modelo_base in texto_completo:
            return False # Encontramos "error" y el nombre del modelo, así que hay que reprocesar
            
    return True

# ==========================================
# ANÁLISIS MULTIMODELO POR DOCUMENTO
# ==========================================

def analizar_documento_completo(par: dict, umbral: float, min_palabras: int, fila_previa: dict = None) -> dict:
    if fila_previa is None:
        fila_previa = {}
        
    original = par["texto_original"]
    variante = par["texto_documento"]

    palabras_orig = limpiar_texto(original)
    palabras_var = limpiar_texto(variante)

    n_orig = len(palabras_orig)
    n_var = len(palabras_var)
    es_par_corto = n_orig < min_palabras or n_var < min_palabras

    # Diccionario base con los metadatos y análisis matemático
    resultado = {
        "task_id": par["task_id"],
        "perfil": par["perfil"],
        "nivel": par["nivel"],
        "etiqueta_real": par["etiqueta_real"],
        "palabras_original": n_orig,
        "palabras_variante": n_var,
        "respuesta_original": original,
        "respuesta_variante": variante,
    }

    debe_llamar_llm = False

    if es_par_corto:
        resultado["ruta"] = "llm_directo"
        resultado["sim_trigramas"] = "N/A"
        resultado["sim_levenshtein"] = "N/A"
        resultado["sim_maxima"] = "N/A"
        resultado["pasa_filtro_matematico"] = "N/A"
        debe_llamar_llm = True
    else:
        sim_tri = calcular_trigramas(palabras_orig, palabras_var)
        sim_lev = calcular_levenshtein_palabras(palabras_orig, palabras_var)
        sim_max = max(sim_tri, sim_lev)

        resultado["ruta"] = "filtro_matematico"
        resultado["sim_trigramas"] = round(sim_tri * 100, 1)
        resultado["sim_levenshtein"] = round(sim_lev * 100, 1)
        resultado["sim_maxima"] = round(sim_max * 100, 1)

        if sim_max >= umbral:
            resultado["pasa_filtro_matematico"] = True
            debe_llamar_llm = True
        else:
            resultado["pasa_filtro_matematico"] = False
            debe_llamar_llm = False

    # Ejecutar todos los modelos para este documento
    for modelo in MODELOS_LLM:
        if debe_llamar_llm:
            
            if modelo_es_valido(fila_previa, modelo):
                print(f"\n    -> {modelo} ya ejecutado con éxito. Rescatando datos...", end="")
                resultado[f"prediccion_{modelo}"] = fila_previa.get(f"prediccion_{modelo}")
                resultado[f"veredicto_{modelo}"] = fila_previa.get(f"veredicto_{modelo}")
                resultado[f"tiempo_{modelo}"] = fila_previa.get(f"tiempo_{modelo}")
                resultado[f"justificacion_{modelo}"] = fila_previa.get(f"justificacion_{modelo}")
                continue
            
            print(f"\n    -> Consultando {modelo}...", end=" ", flush=True)
            inicio = time.time()
            
            try:
                analisis = verificar_plagio_llm("(Clough corpus)", original, variante, modelo=modelo)
                tiempo_total = time.time() - inicio
                
                veredicto = analisis.get("veredicto")
                prediccion = "plagio" if veredicto == "Posible Plagio" else "no_plagio"
                justificacion = analisis.get("justificacion")
                
            except Exception as e:
                tiempo_total = time.time() - inicio
                # MODIFICACIÓN: Forzamos a que guarde la palabra "Error" y el modelo
                veredicto = f"Error {modelo}" 
                prediccion = "Error"
                justificacion = f"Fallo en ejecución: Error {modelo} - {str(e)}"
                print("[!] Fallo.", end=" ")

            resultado[f"prediccion_{modelo}"] = prediccion
            resultado[f"veredicto_{modelo}"] = veredicto
            resultado[f"tiempo_{modelo}"] = round(tiempo_total, 2)
            resultado[f"justificacion_{modelo}"] = justificacion
            print(f"[{round(tiempo_total, 1)}s]")
            
        else:
            resultado[f"prediccion_{modelo}"] = "no_plagio"
            resultado[f"veredicto_{modelo}"] = "N/A"
            resultado[f"tiempo_{modelo}"] = 0.0
            resultado[f"justificacion_{modelo}"] = "Descartado por filtro matemático"

    return resultado

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print("Cargando corpus Clough & Stevenson...")
    pares = cargar_corpus(CORPUS_DIR)
    print(f"  Pares totales: {len(pares)}")

    documentos_procesados = {}
    archivo_csv = Path(OUTPUT_FILE)
    if archivo_csv.exists():
        with open(archivo_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                documentos_procesados[row["task_id"]] = row
        print(f"  Historial cargado: {len(documentos_procesados)} documentos encontrados.")

    print(f"\nAnalizando con umbral={UMBRAL}, min_palabras={MIN_PALABRAS}...")
    
    temp_file = Path(OUTPUT_FILE + ".tmp")
    casos_evaluados_hoy = 0

    with open(temp_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        writer.writeheader()

        for i, par in enumerate(pares, 1):
            task_id = par['task_id']
            fila_previa = documentos_procesados.get(task_id, {})
            
            necesita_procesar = False
            if not fila_previa:
                necesita_procesar = True
            else:
                for modelo in MODELOS_LLM:
                    if not modelo_es_valido(fila_previa, modelo):
                        necesita_procesar = True
                        break

            if not necesita_procesar:
                print(f"  [{i}/{len(pares)}] {task_id} totalmente evaluado. Saltando...")
                writer.writerow(fila_previa)
                continue
                
            print(f"  [{i}/{len(pares)}] Procesando/Reintentando {task_id} ({par['nivel']})...", end="", flush=True)
            
            res = analizar_documento_completo(par, UMBRAL, MIN_PALABRAS, fila_previa)
            
            writer.writerow(res)
            f.flush()
            
            print(f"\n  ✅ {task_id} guardado en disco.\n")
            casos_evaluados_hoy += 1

    temp_file.replace(OUTPUT_FILE)

    print(f"\n✅ Ejecución finalizada. Resultados guardados en: {OUTPUT_FILE}")
    print(f"   Documentos analizados o re-procesados en esta sesión: {casos_evaluados_hoy}")

if __name__ == "__main__":
    main()