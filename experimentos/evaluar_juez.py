"""
Fase Final v2: Evaluación Cualitativa con LLM-as-a-Judge (Llama 3.3 70B)
VERSIÓN CORREGIDA: Rúbrica con anclajes intermedios para forzar escala 1-5.
"""

import csv
import os
import time
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sistema.config import RESULTS_DIR, resolve_project_path

# --- 1. CONFIGURACIÓN API ---
POLIGPT_API_BASE = "https://api.poligpt.upv.es"
POLIGPT_API_KEY = "[CLAVE_ELIMINADA]"  # Idealmente usa variables de entorno

# --- 2. ESQUEMA DEL MEGA-JUEZ ---
class EvaluacionJuez(BaseModel):
    p1_andamiaje: int = Field(description="Nota del 1 al 5 siguiendo estrictamente la rúbrica de Andamiaje.")
    p1_coherencia: int = Field(description="Nota del 1 al 5 siguiendo estrictamente la rúbrica de Coherencia.")
    p2_andamiaje: int = Field(description="Nota del 1 al 5 siguiendo estrictamente la rúbrica de Andamiaje.")
    p2_coherencia: int = Field(description="Nota del 1 al 5 siguiendo estrictamente la rúbrica de Coherencia.")
    medio_integracion: int = Field(description="Nota del 1 al 5 siguiendo la rúbrica de Integración.")
    bajo_integracion: int = Field(description="Nota del 1 al 5 siguiendo la rúbrica de Integración.")
    fidelidad_global: int = Field(description="Nota del 1 al 5 siguiendo la rúbrica de Fidelidad.")
    justificacion_global: str = Field(description="Resumen breve (3 líneas) justificando por qué has asignado las notas intermedias o extremas.")

# --- 3. EL NUEVO PROMPT (RÚBRICA DETALLADA) ---
PROMPT_RUBRICA = """Eres un experto en diseño instruccional evaluando adaptaciones de ejercicios generadas por una IA.
Debes usar TODA LA ESCALA DEL 1 AL 5. No te limites a 1 y 5. Usa los valores 2, 3 y 4 cuando la respuesta no sea ni perfecta ni un desastre.

=== REGLAS DEL SISTEMA ===
- Enunciado Medio = Enunciado Base + Pista 1
- Enunciado Bajo = Enunciado Base + Pista 1 + Pista 2

=== CONTEXTO ORIGINAL ===
- Enunciado Base: {enunciado_base}
- Estrategia Evaluada: {estrategia} / {subestrategia}
- Pista 1 Esperada (Gold Standard): {pista_1_ref}
- Pista 2 Esperada (Gold Standard): {pista_2_ref}

=== GENERACIONES DE LA IA A EVALUAR ===
- Pista 1 IA: {pista_1_gen}
- Pista 2 IA: {pista_2_gen}
- Enunciado Medio IA: {enunciado_medio_gen}
- Enunciado Bajo IA: {enunciado_bajo_gen}

=== RÚBRICA ESTRICTA Y ESCALONADA (1 a 5) ===

1. ANDAMIAJE (Para P1 y P2):
[1] Pésimo: Da la solución final directamente, arruinando el ejercicio.
[2] Deficiente: No da la solución exacta, pero da demasiada información haciendo el ejercicio trivial.
[3] Aceptable: Da una guía genérica. Ni buena ni mala, cumple su función básica.
[4] Bueno: Buena guía socrática, hace pensar al alumno, casi al nivel del profesor.
[5] Excelente: Andamiaje perfecto. Guía magistralmente sin dar la respuesta.

2. COHERENCIA (Para P1 y P2):
[1] Incoherente: Alucina cosas que no están en el enunciado base o no tiene sentido.
[3] Parcial: Relacionado con el tema, pero el enfoque pedagógico es un poco extraño.
[5] Perfecto: Totalmente alineado con el enunciado y la estrategia evaluada.

3. INTEGRACIÓN (Para Enunciados Medio y Bajo):
[1] Pésimo: Ha hecho un 'copia-pega' bruto de las pistas al final del enunciado, rompiendo la gramática o siendo redundante.
[2] Deficiente: Integrado, pero suena forzado o repetitivo.
[3] Aceptable: Se entiende bien, pero claramente se nota dónde se ha añadido el texto nuevo.
[4] Bueno: Buena cohesión narrativa, suena bastante natural.
[5] Excelente: Fluidez perfecta. Parece que el enunciado se escribió así desde el principio.

4. FIDELIDAD GLOBAL (Al integrar):
[1] Pésimo: Cambió completamente los datos críticos del problema original.
[3] Aceptable: Modificó algún detalle menor que no afecta, pero el objetivo es el mismo.
[5] Perfecto: Mantuvo exactamente la integridad de la tarea base.
"""

def get_juez_llm():
    llm = ChatOpenAI(
        model="llama3.3:70b",
        api_key=POLIGPT_API_KEY,
        base_url=POLIGPT_API_BASE,
        temperature=0.0, # Mantenemos 0 para que aplique la rúbrica sin "creatividad"
        max_tokens=1500,
    )
    return llm.with_structured_output(EvaluacionJuez)


def evaluar_con_juez(
    input_csv=RESULTS_DIR / "resultados_similitud.csv",
    output_csv=RESULTS_DIR / "resultados_finales_v2.csv"
):
    input_csv = resolve_project_path(input_csv)
    output_csv = resolve_project_path(output_csv)

    if not input_csv.exists():
        print(f"ERROR: No se encuentra {input_csv}.")
        return

    juez_llm = get_juez_llm()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_RUBRICA)
    
    filas = []
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filas.append(row)

    columnas_juez = [
        "juez_p1_andamiaje", "juez_p1_coherencia", 
        "juez_p2_andamiaje", "juez_p2_coherencia",
        "juez_medio_integracion", "juez_bajo_integracion", 
        "juez_fidelidad", "juez_justificacion"
    ]
    fieldnames = list(filas[0].keys())
    
    for col in columnas_juez:
        if col not in fieldnames:
            fieldnames.append(col)

    procesados = 0
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            procesados = sum(1 for row in reader) - 1
            if procesados < 0: procesados = 0

    print(f"Iniciando Re-Evaluación v2. Evaluando {len(filas)} registros (Empezando desde el {procesados + 1})...")

    modo_apertura = "a" if procesados > 0 else "w"
    
    with open(output_csv, modo_apertura, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if procesados == 0:
            writer.writeheader()

        for i, row in enumerate(filas):
            if i < procesados:
                continue 

            if row.get("error", "").strip() or not row.get("pista_1_generada", "").strip():
                for col in columnas_juez:
                    row[col] = ""
                writer.writerow(row)
                continue
                
            print(f"[{i+1}/{len(filas)}] Re-evaluando tarea {row['id']} de {row['modelo_usado']}...")

            variables_prompt = {
                "enunciado_base": row.get("enunciado_base", ""),
                "estrategia": row.get("estrategia", ""),
                "subestrategia": row.get("subestrategia", ""),
                "pista_1_ref": row.get("pista_1", ""),
                "pista_2_ref": row.get("pista_2", ""),
                "pista_1_gen": row.get("pista_1_generada", ""),
                "pista_2_gen": row.get("pista_2_generada", ""),
                "enunciado_medio_gen": row.get("enunciado_medio_generado", ""),
                "enunciado_bajo_gen": row.get("enunciado_bajo_generado", "")
            }

            try:
                prompt_formateado = prompt_template.invoke(variables_prompt)
                evaluacion = juez_llm.invoke(prompt_formateado)
                
                row["juez_p1_andamiaje"] = evaluacion.p1_andamiaje
                row["juez_p1_coherencia"] = evaluacion.p1_coherencia
                row["juez_p2_andamiaje"] = evaluacion.p2_andamiaje
                row["juez_p2_coherencia"] = evaluacion.p2_coherencia
                row["juez_medio_integracion"] = evaluacion.medio_integracion
                row["juez_bajo_integracion"] = evaluacion.bajo_integracion
                row["juez_fidelidad"] = evaluacion.fidelidad_global
                row["juez_justificacion"] = evaluacion.justificacion_global.replace("\n", " ")
                
            except Exception as e:
                print(f"  -> ERROR evaluando fila {i+1}: {e}")
                for col in columnas_juez:
                    row[col] = ""
                row["juez_justificacion"] = f"ERROR_API: {e}"

            writer.writerow(row)
            f.flush()
            time.sleep(1)

    print(f"\n¡Re-Evaluación v2 completada! Guardado en: {output_csv}")

if __name__ == "__main__":
    evaluar_con_juez()