"""
Evaluación Cuantitativa: Similitud Semántica mediante Embeddings.
Compara las generaciones de los modelos contra el Ground Truth esperado.
"""

import csv
import os
import math
import time
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from sistema.config import RESULTS_DIR, resolve_project_path

# --- CONFIGURACIÓN DE LA API ---
POLIGPT_API_BASE = "https://api.poligpt.upv.es"
POLIGPT_API_KEY = "[CLAVE_ELIMINADA]" # En producción, usa os.getenv("POLIGPT_API_KEY")

# Caché en memoria para no repetir peticiones a la API del mismo texto
# Clave: Texto (str), Valor: Vector (List[float])
EMBEDDING_CACHE: Dict[str, List[float]] = {}

def obtener_modelo_embeddings():
    """Inicializa el cliente de Langchain para embeddings."""
    return OpenAIEmbeddings(
        model="bge-m3",
        openai_api_key=POLIGPT_API_KEY,
        openai_api_base=POLIGPT_API_BASE
    )

def get_embedding(texto: str, embedder) -> List[float]:
    """Obtiene el embedding de un texto usando caché para optimizar llamadas."""
    if not texto or not texto.strip():
        return []
    
    texto_limpio = texto.strip()
    if texto_limpio in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[texto_limpio]
    
    try:
        vector = embedder.embed_query(texto_limpio)
        EMBEDDING_CACHE[texto_limpio] = vector
        return vector
    except Exception as e:
        print(f"Error obteniendo embedding para texto: '{texto_limpio[:30]}...' -> {e}")
        return []

def similitud_coseno(vec1: List[float], vec2: List[float]) -> float:
    """Calcula la similitud del coseno entre dos vectores (0.0 a 1.0)."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def evaluar_similitud_csv(
    input_csv=RESULTS_DIR / "resultados_generador_enunciados.csv",
    output_csv=RESULTS_DIR / "resultados_similitud.csv"
):
    input_csv = resolve_project_path(input_csv)
    output_csv = resolve_project_path(output_csv)
    embedder = obtener_modelo_embeddings()
    filas = []
    
    # 1. Leer el CSV generado previamente
    if not os.path.exists(input_csv):
        print(f"No se encuentra el archivo {input_csv}.")
        return
        
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filas.append(row)

    # 2. Preparar el nuevo CSV con las columnas de métricas
    columnas_extra = [
        "sim_pista_1", 
        "sim_pista_2", 
        "sim_enunciado_medio", 
        "sim_enunciado_bajo",
        "sim_cambio_base" # Para medir cuánto se ha adaptado respecto al original
    ]
    fieldnames = list(filas[0].keys()) + columnas_extra
    
    print(f"Iniciando evaluación semántica para {len(filas)} filas...")

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(filas):
            # Saltamos si hubo error de generación previa
            if row.get("error") or not row.get("pista_1_generada"):
                for col in columnas_extra:
                    row[col] = ""
                writer.writerow(row)
                continue
                
            print(f"[{i+1}/{len(filas)}] Calculando similitudes para tarea {row['id']} ({row['modelo_usado']})")

            # --- OBTENER VECTORES ESPERADOS (Usa caché) ---
            vec_p1_ref = get_embedding(row.get("pista_1", ""), embedder)
            vec_p2_ref = get_embedding(row.get("pista_2", ""), embedder)
            vec_med_ref = get_embedding(row.get("enunciado_medio", ""), embedder)
            vec_baj_ref = get_embedding(row.get("enunciado_bajo", ""), embedder)
            vec_base_ref = get_embedding(row.get("enunciado_base", ""), embedder)

            # --- OBTENER VECTORES GENERADOS ---
            vec_p1_gen = get_embedding(row.get("pista_1_generada", ""), embedder)
            vec_p2_gen = get_embedding(row.get("pista_2_generada", ""), embedder)
            vec_med_gen = get_embedding(row.get("enunciado_medio_generado", ""), embedder)
            vec_baj_gen = get_embedding(row.get("enunciado_bajo_generado", ""), embedder)

            # --- CALCULAR SIMILITUDES ---
            # Guardamos con 4 decimales para que el CSV sea más limpio
            row["sim_pista_1"] = round(similitud_coseno(vec_p1_ref, vec_p1_gen), 4)
            row["sim_pista_2"] = round(similitud_coseno(vec_p2_ref, vec_p2_gen), 4)
            row["sim_enunciado_medio"] = round(similitud_coseno(vec_med_ref, vec_med_gen), 4)
            row["sim_enunciado_bajo"] = round(similitud_coseno(vec_baj_ref, vec_baj_gen), 4)
            
            # Métrica especial: Base Original vs Enunciado Bajo Generado
            # Si esta similitud es muy alta (ej. > 0.98), significa que el modelo apenas ha añadido las pistas o no ha integrado bien.
            row["sim_cambio_base"] = round(similitud_coseno(vec_base_ref, vec_baj_gen), 4)

            writer.writerow(row)
            f.flush()

    print(f"\n¡Evaluación semántica finalizada! Resultados guardados en {output_csv}")

if __name__ == "__main__":
    evaluar_similitud_csv()