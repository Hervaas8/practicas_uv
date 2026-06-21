"""Generación de pistas y enunciados adaptados mediante LLM.

Este módulo incorpora procesamiento híbrido: si el modelo tiene capacidad de visión,
pasa las imágenes en base64. Si es de solo texto, utiliza el módulo OCR para
extraer el texto y adjuntarlo al contexto.
"""

from __future__ import annotations

import json
import re
import time
import yaml
import base64
import os
import io
from PIL import Image
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError
from langchain_core.messages import HumanMessage

# Importamos el esquema centralizado y la configuración de LLM
from sistema.utils.esquemas import LLMScaffoldingOutput
from sistema.utils.configuracion_llm import get_llm, is_vision_model
from sistema.config import PROMPTS_PATH, resolve_project_path



# Importación de utilidades OCR
try:
    from sistema.utils.utils_easy import extract_text_with_llm 
except ImportError:
    def extract_text_with_llm(path): return "[ERROR: Módulo OCR no encontrado]"

# Carga del prompt desde YAML
try:
    with open(PROMPTS_PATH, "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    # Utilizamos el prompt de generador de pistas
    PROMPT_GENERADOR_ENUNCIADOS = prompts.get("generador_pistas_prompt", "")
except Exception as e:
    print(f"Error crítico al cargar prompts.yaml en generador_enunciados: {e}")
    import sys
    sys.exit(1)


def encode_image(image_path, max_size=(800, 800)):
    """Redimensiona, comprime a JPEG y codifica una imagen a Base64 para no saturar los tokens."""
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            # Convertir a RGB por si es un PNG con fondo transparente
            img = img.convert("RGB")
            # Redimensionar manteniendo la proporción
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None


def _get_task_id(tarea: Dict[str, Any]) -> str:
    return str(tarea.get("id") or tarea.get("task_id") or "")


def _get_task_images(tarea: Dict[str, Any]) -> List[str]:
    imagenes = tarea.get("task_imagen", tarea.get("task_image", []))
    if imagenes is None:
        return []
    if isinstance(imagenes, str):
        return [imagenes]
    if isinstance(imagenes, list):
        return [str(img) for img in imagenes]
    return [str(imagenes)]


def _get_enunciado_base(tarea: Dict[str, Any]) -> str:
    if tarea.get("enunciado_nivel_alto"):
        return str(tarea["enunciado_nivel_alto"])

    prompt = tarea.get("task_prompt", "")
    if isinstance(prompt, dict):
        return str(prompt.get("nivel_alto") or prompt.get("nivel_3") or "")
    return str(prompt)


def _get_estrategia(tarea: Dict[str, Any]) -> str:
    return str(tarea.get("estrategia") or tarea.get("strategy") or "general")


def _get_subestrategia(tarea: Dict[str, Any]) -> str:
    value = tarea.get("subestrategia", tarea.get("sub_strategy", ""))
    return "" if value is None else str(value)


def construir_contexto_extraido(tarea: Dict[str, Any]) -> str:
    """Construye un contexto textual simple a partir de la tarea."""
    partes: List[str] = []

    if tarea.get("contexto_extraido"):
        partes.append(str(tarea["contexto_extraido"]))

    if tarea.get("task_text"):
        partes.append(f"Texto de apoyo: {tarea['task_text']}")


    return " | ".join(partes) if partes else "No se proporciona contexto adicional."


def construir_prompt_generador(
    tarea: Dict[str, Any],
    contexto_extraido: Optional[str] = None,
) -> str:
    """Rellena el prompt del generador para una tarea concreta usando replace para evitar fallos con JSON."""
    return PROMPT_GENERADOR_ENUNCIADOS.replace(
        "{contexto_extraido}", contexto_extraido or construir_contexto_extraido(tarea)
    ).replace(
        "{enunciado_base}", _get_enunciado_base(tarea)
    ).replace(
        "{estrategia}", _get_estrategia(tarea)
    ).replace(
        "{subestrategia}", _get_subestrategia(tarea)
    )


def _extraer_json_de_respuesta(texto: str) -> Dict[str, Any]:
    """Extrae y parsea el primer objeto JSON encontrado en una respuesta de texto."""
    texto = texto.strip()

    # Elimina fences tipo ```json ... ``` si aparecen.
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\s*```$", "", texto)
        texto = texto.strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if not match:
        raise ValueError("La respuesta del modelo no contiene un objeto JSON válido.")

    return json.loads(match.group(0))


def _normalizar_salida(raw_output: Any) -> Dict[str, str]:
    """Convierte la respuesta cruda del LLM a un diccionario validado con nuestro esquema."""
    if isinstance(raw_output, LLMScaffoldingOutput):
        return raw_output.model_dump()

    if isinstance(raw_output, dict):
        return LLMScaffoldingOutput(**raw_output).model_dump()

    content = getattr(raw_output, "content", raw_output)
    if not isinstance(content, str):
        raise TypeError(f"Tipo de respuesta no soportado: {type(raw_output)}")

    data = _extraer_json_de_respuesta(content)
    return LLMScaffoldingOutput(**data).model_dump()


# =====================================================================
# 5. MOTOR PRINCIPAL DE GENERACIÓN
# =====================================================================

def generar_enunciados_adaptados(
    tarea: Dict[str, Any],
    modelo: str = "poligpt",
    contexto_extraido: Optional[str] = None,
    usar_salida_estructurada: bool = True,
) -> Dict[str, Any]:
    """Genera pistas y enunciados adaptados para una tarea integrando OCR o Visión."""
    
    # 1. Determinar si usamos visión o OCR
    rutas_imagenes = _get_task_images(tarea)
    es_vision = is_vision_model(modelo)

    # Restricción de seguridad: apagar visión si hay muchas imágenes para modelos como qwen
    if es_vision and len(rutas_imagenes) > 5 and "qwen" in modelo.lower():
        es_vision = False

    # 2. Procesamiento OCR si no es modelo de visión
    texto_ocr_adicional = ""
    if not es_vision and rutas_imagenes:
        partes_ocr = []
        for img_path in rutas_imagenes:
            if img_path: 
                partes_ocr.append(extract_text_with_llm(img_path))
        texto_ocr_adicional = "\n\n[TEXTO EXTRAÍDO DE LAS IMÁGENES MEDIANTE OCR]:\n" + "\n".join(filter(None, partes_ocr))

    # 3. Construir contexto y prompt final
    contexto_final = (contexto_extraido or construir_contexto_extraido(tarea)) + texto_ocr_adicional
    prompt_str = construir_prompt_generador(tarea, contexto_extraido=contexto_final)

    # 4. Estructurar mensaje para el LLM
    content = [{"type": "text", "text": prompt_str}]

    if es_vision and rutas_imagenes:
        for img_path in rutas_imagenes:
            img_b64 = encode_image(img_path)
            if img_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    msg = HumanMessage(content=content)
    llm = get_llm(tipo_tarea="generador", override_model=modelo)

    inicio = time.perf_counter()
    ultimo_error: Optional[Exception] = None

    if usar_salida_estructurada:
        try:
            llm_estructurado = llm.with_structured_output(LLMScaffoldingOutput)
            raw_output = llm_estructurado.invoke([msg])
            resultado = _normalizar_salida(raw_output)
            resultado["tiempo_respuesta"] = time.perf_counter() - inicio
            return resultado
        except Exception as exc: 
            ultimo_error = exc

    # Fallback a invocación estándar si structured_output falla
    try:
        raw_output = llm.invoke([msg])
        resultado = _normalizar_salida(raw_output)
        resultado["tiempo_respuesta"] = time.perf_counter() - inicio
        return resultado
    except (ValidationError, json.JSONDecodeError, ValueError, TypeError) as exc:
        detalle = f"Error al generar/parsear la salida con el modelo {modelo}: {exc}"
        if ultimo_error is not None:
            detalle += f" | Error previo con salida estructurada: {ultimo_error}"
        raise RuntimeError(detalle) from exc


# =====================================================================
# 6. FUNCIONES INTEGRADORAS (WRAPPERS)
# =====================================================================

def generar_task_prompt_adaptado(
    tarea: Dict[str, Any],
    modelo: str = "poligpt",
    contexto_extraido: Optional[str] = None,
) -> Dict[str, Any]:
    """Devuelve una copia de la tarea con `task_prompt` actualizado con los niveles correspondientes."""
    salida = generar_enunciados_adaptados(
        tarea=tarea,
        modelo=modelo,
        contexto_extraido=contexto_extraido,
    )

    tarea_adaptada = dict(tarea)
    enunciado_alto = _get_enunciado_base(tarea)
    tarea_adaptada["task_prompt"] = {
        "nivel_alto": enunciado_alto,
        "nivel_medio": salida["enunciado_nivel_medio"],
        "nivel_bajo": salida["enunciado_nivel_bajo"],
    }
    tarea_adaptada["pistas_generadas"] = {
        "pista_1": salida["pista_1"],
        "pista_2": salida["pista_2"],
        "modelo": modelo,
        "tiempo_respuesta": salida["tiempo_respuesta"],
    }
    return tarea_adaptada


def generar_enunciados_para_sesion(
    tareas: Iterable[Dict[str, Any]],
    modelo: str = "poligpt",
) -> List[Dict[str, Any]]:
    """Aplica la generación de enunciados adaptados a una lista completa de tareas."""
    return [generar_task_prompt_adaptado(tarea, modelo=modelo) for tarea in tareas]


# =====================================================================
# 7. EJECUCIÓN DIRECTA (MODO SCRIPT)
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera pistas y enunciados adaptados para la primera tarea de un JSON.")
    parser.add_argument("input_json", help="Ruta del JSON con tareas.")
    parser.add_argument("modelo", nargs="?", default="poligpt", help="Modelo a utilizar.")
    args = parser.parse_args()
    input_path = resolve_project_path(args.input_json)

    with open(input_path, "r", encoding="utf-8-sig") as f:
        tareas = json.load(f)

    if not isinstance(tareas, list) or not tareas:
        raise ValueError("El JSON de entrada debe ser una lista de tareas no vacía.")

    salida = generar_enunciados_adaptados(tareas[0], modelo=args.modelo)
    print(json.dumps(salida, ensure_ascii=False, indent=2))