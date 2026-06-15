"""
evaluar_clough.py
-----------------
Evalúa el detector de plagio sobre el corpus de Clough & Stevenson (CS11).
Compara cada documento contra el original de su tarea y genera un CSV
con todos los casos analizados.

Estructura esperada del corpus:
    clough_corpus/
        orig_taska.txt
        orig_taskb.txt
        ...
        g0pA_taska.txt   (categoría: non / cut / light / heavy)
        ...
        file_information.csv

Uso:
    python evaluar_clough.py \
        --corpus_dir ./clough_corpus \
        --output resultados_clough.csv

Requiere que detector_plagio.py esté en el mismo directorio.

Nota sobre etiquetas:
    - cut / light  → plagio léxico  → etiqueta: plagio
    - heavy        → plagio semántico → etiqueta: no_plagio (tu sistema NO debería detectarlo)
    - non          → sin plagio     → etiqueta: no_plagio
"""

import csv
import argparse
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
        "prediccion_final": None,
        "texto_original": original,
        "texto_documento": documento,
    }

    if es_par_corto:
        resultado["ruta"] = "llm_directo"
        resultado["pasa_filtro_matematico"] = "N/A"
        analisis = verificar_plagio_llm("(Clough corpus)", original, documento)
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
            analisis = verificar_plagio_llm("(Clough corpus)", original, documento)
            resultado["veredicto_llm"] = analisis.get("veredicto")
            resultado["justificacion_llm"] = analisis.get("justificacion")
            resultado["frases_identicas"] = str(analisis.get("frases_identicas", []))
            resultado["prediccion_final"] = (
                "plagio" if resultado["veredicto_llm"] == "Posible Plagio" else "no_plagio"
            )
        else:
            resultado["pasa_filtro_matematico"] = False
            resultado["prediccion_final"] = "no_plagio"

    return resultado

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", required=True,
                        help="Directorio raíz del corpus Clough")
    parser.add_argument("--output", default="resultados_clough.csv")
    args = parser.parse_args()

    print("Cargando corpus Clough & Stevenson...")
    pares = cargar_corpus(args.corpus_dir)
    print(f"  Pares cargados: {len(pares)}")
    print(f"  Plagios léxicos (cut+light): {sum(1 for p in pares if p['etiqueta_real'] == 'plagio')}")
    print(f"  No plagio (non+heavy): {sum(1 for p in pares if p['etiqueta_real'] == 'no_plagio')}")

    print(f"\nAnalizando con umbral={UMBRAL}, min_palabras={MIN_PALABRAS}...")
    resultados = []
    for i, par in enumerate(pares, 1):
        print(f"  [{i}/{len(pares)}] {par['filename']} ({par['categoria_clough']})...",
              end=" ", flush=True)
        res = analizar_par(par, UMBRAL, MIN_PALABRAS)
        resultados.append(res)
        print(f"→ {res['prediccion_final']} (real: {res['etiqueta_real']})")

    # Guardar CSV
    campos = [
        "filename", "tarea", "categoria_clough", "etiqueta_real", "prediccion_final",
        "palabras_original", "palabras_documento", "ruta",
        "sim_trigramas", "sim_levenshtein", "sim_maxima",
        "pasa_filtro_matematico", "veredicto_llm", "justificacion_llm",
        "frases_identicas", "texto_original", "texto_documento",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n✅ Resultados guardados en: {args.output}")
    print(f"   Total casos analizados: {len(resultados)}")


if __name__ == "__main__":
    main()