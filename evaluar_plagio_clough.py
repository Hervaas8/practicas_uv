"""
evaluar_clough.py
-----------------
Evalúa el detector de plagio sobre el corpus de Clough & Stevenson (CS11).
Compara cada documento contra el original de su tarea y genera un CSV
con todos los casos analizados. Guarda los resultados incrementalmente
para poder reanudar la ejecución si se interrumpe.
"""

import csv
import time
from pathlib import Path
from detector_plagio import (
    limpiar_texto,
    calcular_trigramas,
    calcular_levenshtein_palabras,
    verificar_plagio_llm,
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Rutas de entrada y salida predeterminadas
CORPUS_DIR = "./data/clough"
OUTPUT_FILE = "resultados_clough.csv"

UMBRAL = 0.25       # Cambiar al valor óptimo encontrado en el dev set
MIN_PALABRAS = 15

# Categorías que tu sistema DEBE detectar (plagio léxico)
CATEGORIAS_PLAGIO = {"cut", "light"}
# Categorías que tu sistema NO debe detectar
CATEGORIAS_NO_PLAGIO = {"non", "heavy"}

# ==========================================
# CARGA DEL CORPUS
# ==========================================

def cargar_corpus(corpus_dir: str) -> list[dict]:
    """
    Lee file_information.csv y carga los textos correspondientes.
    Devuelve lista de pares (original, documento, categoría, tarea).
    """
    corpus_path = Path(corpus_dir)
    info_path = corpus_path / "file_information.csv"

    # Cargar metadatos
    documentos = []
    with open(info_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documentos.append({
                "filename": row["File"],
                "tarea": row["Task"],
                "categoria": row["Category"],
            })

    # Separar originales de comparados
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
            print(f"⚠️  Archivo no encontrado: {doc['filename']} o su original")
            continue

        texto_orig = ruta_orig.read_text(encoding="utf-8", errors="ignore").strip()
        texto_doc = ruta_doc.read_text(encoding="utf-8", errors="ignore").strip()

        etiqueta = "plagio" if doc["categoria"] in CATEGORIAS_PLAGIO else "no_plagio"

        pares.append({
            "filename": doc["filename"],
            "tarea": tarea,
            "categoria_clough": doc["categoria"],
            "etiqueta_real": etiqueta,
            "texto_original": texto_orig,
            "texto_documento": texto_doc,
        })

    return pares

# ==========================================
# ANÁLISIS DE CADA PAR
# ==========================================

def analizar_par(par: dict, umbral: float, min_palabras: int) -> dict:
    """Ejecuta el pipeline completo sobre un par del corpus Clough."""
    original = par["texto_original"]
    documento = par["texto_documento"]

    palabras_orig = limpiar_texto(original)
    palabras_doc = limpiar_texto(documento)

    n_orig = len(palabras_orig)
    n_doc = len(palabras_doc)
    es_par_corto = n_orig < min_palabras or n_doc < min_palabras

    resultado = {
        "filename": par["filename"],
        "tarea": par["tarea"],
        "categoria_clough": par["categoria_clough"],
        "etiqueta_real": par["etiqueta_real"],
        "palabras_original": n_orig,
        "palabras_documento": n_doc,
        "ruta": None,
        "sim_trigramas": None,
        "sim_levenshtein": None,
        "sim_maxima": None,
        "pasa_filtro_matematico": None,
        "veredicto_llm": None,
        "justificacion_llm": None,
        "frases_identicas": None,
        "tiempo_llm_segundos": 0.0, # <-- Nuevo campo de tiempo
        "prediccion_final": None,
        "texto_original": original,
        "texto_documento": documento,
    }

    if es_par_corto:
        resultado["ruta"] = "llm_directo"
        resultado["pasa_filtro_matematico"] = "N/A"
        
        # Medimos el tiempo de respuesta del LLM
        inicio = time.time()
        analisis = verificar_plagio_llm("(Clough corpus)", original, documento)
        tiempo_total = time.time() - inicio
        
        resultado["tiempo_llm_segundos"] = round(tiempo_total, 2)
        resultado["veredicto_llm"] = analisis.get("veredicto")
        resultado["justificacion_llm"] = analisis.get("justificacion")
        resultado["frases_identicas"] = str(analisis.get("frases_identicas", []))
        resultado["prediccion_final"] = (
            "plagio" if resultado["veredicto_llm"] == "Posible Plagio" else "no_plagio"
        )
    else:
        sim_tri = calcular_trigramas(palabras_orig, palabras_doc)
        sim_lev = calcular_levenshtein_palabras(palabras_orig, palabras_doc)
        sim_max = max(sim_tri, sim_lev)

        resultado["ruta"] = "filtro_matematico"
        resultado["sim_trigramas"] = round(sim_tri * 100, 1)
        resultado["sim_levenshtein"] = round(sim_lev * 100, 1)
        resultado["sim_maxima"] = round(sim_max * 100, 1)

        if sim_max >= umbral:
            resultado["pasa_filtro_matematico"] = True
            
            # Medimos el tiempo de respuesta del LLM
            inicio = time.time()
            analisis = verificar_plagio_llm("(Clough corpus)", original, documento)
            tiempo_total = time.time() - inicio
            
            resultado["tiempo_llm_segundos"] = round(tiempo_total, 2)
            resultado["veredicto_llm"] = analisis.get("veredicto")
            resultado["justificacion_llm"] = analisis.get("justificacion")
            resultado["frases_identicas"] = str(analisis.get("frases_identicas", []))
            resultado["prediccion_final"] = (
                "plagio" if resultado["veredicto_llm"] == "Posible Plagio" else "no_plagio"
            )
        else:
            resultado["pasa_filtro_matematico"] = False
            resultado["prediccion_final"] = "no_plagio"
            # tiempo_llm_segundos se queda en 0.0 porque no se llamó al LLM

    return resultado

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print("Cargando corpus Clough & Stevenson...")
    pares = cargar_corpus(CORPUS_DIR)
    print(f"  Pares totales: {len(pares)}")

    # 1. Leer los archivos ya procesados para no repetir
    archivos_procesados = set()
    archivo_csv = Path(OUTPUT_FILE)
    if archivo_csv.exists():
        with open(archivo_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                archivos_procesados.add(row["filename"])
        print(f"  Archivos ya procesados encontrados: {len(archivos_procesados)}")

    print(f"\nAnalizando con umbral={UMBRAL}, min_palabras={MIN_PALABRAS}...")
    
    campos = [
        "filename", "tarea", "categoria_clough", "etiqueta_real", "prediccion_final",
        "palabras_original", "palabras_documento", "ruta",
        "sim_trigramas", "sim_levenshtein", "sim_maxima",
        "pasa_filtro_matematico", "veredicto_llm", "justificacion_llm",
        "frases_identicas", "tiempo_llm_segundos", "texto_original", "texto_documento",
    ]

    # 2. Abrir el archivo en modo "append" (o "write" si no existe)
    modo = "a" if archivos_procesados else "w"
    
    with open(OUTPUT_FILE, modo, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        
        # Solo escribir la cabecera si el archivo es nuevo
        if not archivos_procesados:
            writer.writeheader()

        casos_evaluados_hoy = 0

        for i, par in enumerate(pares, 1):
            if par['filename'] in archivos_procesados:
                # Nos saltamos este archivo si ya está en el CSV
                print(f"  [{i}/{len(pares)}] {par['filename']} ya evaluado. Saltando...")
                continue
                
            print(f"  [{i}/{len(pares)}] {par['filename']} ({par['categoria_clough']})...",
                  end=" ", flush=True)
            
            res = analizar_par(par, UMBRAL, MIN_PALABRAS)
            
            # 3. Guardar en el CSV inmediatamente después de analizar
            writer.writerow(res)
            f.flush() # Forzar escritura en disco para no perder datos si se cancela
            
            print(f"→ {res['prediccion_final']} (real: {res['etiqueta_real']}) [T: {res['tiempo_llm_segundos']}s]")
            casos_evaluados_hoy += 1

    print(f"\n✅ Ejecución finalizada. Resultados guardados en: {OUTPUT_FILE}")
    print(f"   Casos analizados en esta sesión: {casos_evaluados_hoy}")


if __name__ == "__main__":
    main()