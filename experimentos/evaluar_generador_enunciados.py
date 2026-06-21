"""Benchmark del módulo generador de pistas y enunciados adaptados."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sistema.modulos.generador_enunciados import generar_enunciados_adaptados

from sistema.config import JSON_DIR, RESULTS_DIR, resolve_project_path

MODELOS_A_EVALUAR = [
    "deepseek-r1:32b",
    "llama3.1:8b",
    "llama3.2:3b",
    "phi4-mini:3.8b",
    "qwen3-vl:30b",
    "poligpt-code",
    "poligpt",
]

COLUMNAS_CSV = [
    "id",
    "task_imagen",
    "estrategia",
    "subestrategia",
    "modelo_usado",
    "enunciado_base",
    "pista_1",
    "pista_2",
    "enunciado_medio",
    "enunciado_bajo",
    "pista_1_generada",
    "pista_2_generada",
    "enunciado_medio_generado",
    "enunciado_bajo_generado",
    "tiempo_respuesta",
    "error",
]


def cargar_tareas_validacion(path: str, limite: Optional[int] = None) -> List[Dict[str, Any]]:
    """Carga tareas de validación desde JSON."""
    path = resolve_project_path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo de entrada debe ser una lista de tareas.")

    return data[:limite] if limite is not None else data


def _serializar_imagenes(valor: Any) -> str:
    if valor is None:
        return ""
    if isinstance(valor, list):
        return json.dumps(valor, ensure_ascii=False)
    return str(valor)


def _leer_resultados_existentes_y_limpiar_errores(path: str) -> Set[Tuple[str, str]]:
    """Devuelve pares (id, modelo) exitosos y limpia las filas con error del CSV."""
    procesados: Set[Tuple[str, str]] = set()
    if not os.path.exists(path):
        return procesados

    filas_validas = []
    
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("id", "").strip()
            modelo = row.get("modelo_usado", "").strip()
            error = row.get("error", "").strip()
            
            # Solo guardamos como procesadas las filas que NO tienen error
            if not error:
                if task_id and modelo:
                    procesados.add((task_id, modelo))
                filas_validas.append(row)

    # Sobrescribimos el archivo conservando solo las correctas
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        writer.writeheader()
        writer.writerows(filas_validas)

    return procesados


def _crear_fila_base(tarea: Dict[str, Any], modelo: str) -> Dict[str, Any]:
    return {
        "id": tarea.get("id", tarea.get("task_id", "")),
        "task_imagen": _serializar_imagenes(tarea.get("task_imagen", tarea.get("task_image", []))),
        "estrategia": tarea.get("estrategia", tarea.get("strategy", "")),
        "subestrategia": tarea.get("subestrategia", tarea.get("sub_strategy", "")),
        "modelo_usado": modelo,
        "enunciado_base": tarea.get("enunciado_nivel_alto", ""),
        "pista_1": tarea.get("pista_1", ""),
        "pista_2": tarea.get("pista_2", ""),
        "enunciado_medio": tarea.get("enunciado_nivel_medio", ""),
        "enunciado_bajo": tarea.get("enunciado_nivel_bajo", ""),
        "pista_1_generada": "",
        "pista_2_generada": "",
        "enunciado_medio_generado": "",
        "enunciado_bajo_generado": "",
        "tiempo_respuesta": "",
        "error": "",
    }


def evaluar_modelos_generador(
    input_json: str,
    output_csv: str,
    modelos: Iterable[str] = MODELOS_A_EVALUAR,
    limite: Optional[int] = None,
    reanudar: bool = True,
) -> None:
    """Genera el CSV de resultados del módulo para varios modelos."""
    tareas = cargar_tareas_validacion(input_json, limite=limite)
    modelos = list(modelos)

    if reanudar:
        procesados = _leer_resultados_existentes_y_limpiar_errores(output_csv)
    else:
        procesados = set()

    existe_csv = os.path.exists(output_csv)

    with open(output_csv, "a" if reanudar else "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        
        # Solo escribe cabecera si el archivo es nuevo o forzamos no reanudar
        if not existe_csv or not reanudar or os.path.getsize(output_csv) == 0:
            writer.writeheader()

        total = len(tareas) * len(modelos)
        contador = 0

        for tarea in tareas:
            task_id = str(tarea.get("id", tarea.get("task_id", "")))

            for modelo in modelos:
                contador += 1
                
                if reanudar and (task_id, modelo) in procesados:
                    print(f"[{contador}/{total}] Saltando {task_id} con {modelo}: ya procesado correctamente.")
                    continue

                print(f"[{contador}/{total}] Generando {task_id} con {modelo}...")
                fila = _crear_fila_base(tarea, modelo)

                try:
                    salida = generar_enunciados_adaptados(tarea=tarea, modelo=modelo)
                    fila.update(
                        {
                            "pista_1_generada": salida.get("pista_1", ""),
                            "pista_2_generada": salida.get("pista_2", ""),
                            "enunciado_medio_generado": salida.get("enunciado_nivel_medio", ""),
                            "enunciado_bajo_generado": salida.get("enunciado_nivel_bajo", ""),
                            "tiempo_respuesta": round(float(salida.get("tiempo_respuesta", 0.0)), 4),
                            "error": "",
                        }
                    )
                except Exception as exc: 
                    fila["error"] = str(exc)
                    print(f"ERROR en {task_id} con {modelo}: {exc}")

                writer.writerow(fila)
                f.flush()

    print(f"Resultados guardados en: {output_csv}")


def _parse_modelos(modelos_arg: Optional[str]) -> List[str]:
    if not modelos_arg:
        return MODELOS_A_EVALUAR
    return [m.strip() for m in modelos_arg.split(",") if m.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el generador de enunciados adaptados.")
    parser.add_argument(
    "--input",
    default=str(JSON_DIR / "tareas_enunciados.json"),
    help="JSON de validación."
)

    parser.add_argument(
    "--output",
    default=str(RESULTS_DIR / "resultados_generador_enunciados.csv"),
    help="CSV de salida."
)
    parser.add_argument("--modelos", default=None, help="Modelos separados por coma.")
    parser.add_argument("--limite", type=int, default=None, help="Límite de tareas.")
    parser.add_argument("--no-reanudar", action="store_true", help="Sobrescribe el CSV.")

    args = parser.parse_args()

    evaluar_modelos_generador(
        input_json=args.input,
        output_csv=args.output,
        modelos=_parse_modelos(args.modelos),
        limite=args.limite,
        reanudar=not args.no_reanudar,
    )