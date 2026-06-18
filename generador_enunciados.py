"""Generación de pistas y enunciados adaptados mediante LLM.

Este módulo está pensado para dos usos:
1. Evaluar el módulo de generación de andamiaje sobre un conjunto de tareas.
2. Integrarlo más adelante en el flujo del tutor para generar los niveles medio/bajo
   cuando se incorpore una nueva sesión.

No contiene credenciales ni configuración propia: reutiliza `get_llm` de
`configuracion_llm.py`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError



class LLMGeneradorEnunciadosOutput(BaseModel):
    """Salida esperada del generador de andamiaje."""

    pista_1: str = Field(
        description=(
            "Primera pista. Describe el formato, el contexto digital o la forma "
            "en que está organizada la información, sin resolver la tarea."
        )
    )
    pista_2: str = Field(
        description=(
            "Segunda pista. Añade una ayuda más concreta sobre qué observar, "
            "buscar o comparar, sin resolver la tarea."
        )
    )
    enunciado_nivel_medio: str = Field(
        description="Enunciado adaptado de nivel medio. Debe incluir solo la pista 1."
    )
    enunciado_nivel_bajo: str = Field(
        description="Enunciado adaptado de nivel bajo. Debe incluir la pista 1 y la pista 2."
    )


PROMPT_GENERADOR_ENUNCIADOS = """
Eres un experto en pedagogía digital y en diseño de andamiaje cognitivo. Tu objetivo es generar pistas para estudiantes que están resolviendo tareas de competencia digital, adaptando un enunciado de nivel alto a versiones con más apoyo.

Aquí tienes los datos de la tarea:
- Contexto de la tarea, si existe: {contexto_extraido}
- Enunciado base, nivel alto, sin ayudas: {enunciado_base}
- Estrategia evaluada: {estrategia}
- Subestrategia evaluada: {subestrategia}

Tu tarea es generar dos pistas acumulativas y construir los enunciados adaptados de nivel medio y nivel bajo.

La pista 1 debe describir el formato, el contexto digital o el tipo de recurso que el estudiante debe interpretar. Debe ayudar a entender cómo está organizada la información, pero sin indicar directamente la respuesta.

La pista 2 debe añadir una ayuda más concreta sobre qué debe observar, buscar o comparar. Puede incluir ejemplos orientativos, pero nunca debe resolver directamente la tarea.

El enunciado de nivel medio debe incorporar únicamente la pista 1 junto al enunciado base.

El enunciado de nivel bajo debe incorporar la pista 1 y la pista 2 junto al enunciado base.

Las pistas deben colocarse antes de la pregunta o instrucción principal de la tarea. Si el enunciado contiene una introducción, un diálogo o una explicación inicial, conserva esa parte y añade las pistas justo antes de la pregunta o instrucción que debe resolver el estudiante. Si no se puede separar claramente la introducción de la pregunta, coloca las pistas antes del enunciado base.

REGLAS SEGÚN LA ESTRATEGIA:

Si la estrategia es Navegación y la subestrategia es Localizar estructura/menús:
- Pista 1: explica que las páginas web, redes sociales o plataformas digitales suelen organizar la información mediante menús, pestañas, secciones, enlaces o respuestas.
- Pista 2: sugiere el tipo de enlace, pestaña o sección que podría buscarse, usando expresiones como “yo buscaría enlaces como...”, “fíjate en apartados relacionados con...” o “busca opciones similares a...”.

Si la estrategia es Navegación y la subestrategia es Generar términos de búsqueda:
- Pista 1: describe de forma general cómo funcionan los buscadores o chats para encontrar información.
- Pista 2: sugiere usar palabras clave concretas relacionadas con el objetivo de la tarea, sin dar la respuesta final.

Si la estrategia es Evaluación y la subestrategia está relacionada con la experiencia o fiabilidad de las fuentes:
- Pista 1: recuerda que en internet no toda la información es fiable y que cualquier persona puede publicar contenido.
- Pista 2: explica que conviene fijarse en si la fuente es oficial, experta, institucional, verificable o escrita por personas con conocimiento del tema.

Si la estrategia es Evaluación y la subestrategia está relacionada con identificar fuentes:
- Pista 1: explica que en periódicos, páginas web o publicaciones digitales la autoría suele aparecer en zonas visibles del texto o de la página.
- Pista 2: aclara que a veces el autor puede ser una institución, un organismo oficial o una entidad, no solo una persona individual.

Si la estrategia es Evaluación y la subestrategia está relacionada con benevolencia o intención:
- Pista 1: recuerda que en internet algunas fuentes no buscan informar, sino convencer, vender o manipular.
- Pista 2: aconseja fijarse en la intención del autor o de la página: informar, opinar, persuadir, vender o engañar.

Si la estrategia es Integración y la subestrategia es Multitexto:
- Pista 1: explica que en internet podemos encontrar mucha información en distintos textos, y que esta puede ser similar, complementaria o contradictoria.
- Pista 2: indica que hay que leer ambos textos con atención y comparar sus ideas, aunque estén expresadas con palabras diferentes.

Si la estrategia es Integración y la subestrategia es Multimodal:
- Pista 1: explica que la información digital puede aparecer en distintos formatos, como textos, imágenes, vídeos, comentarios o conversaciones.
- Pista 2: indica que cada formato puede aportar información distinta y que conviene relacionar lo visual con lo escrito.

Si la estrategia es Integración y la subestrategia es Comparar/Corroborar:
- Pista 1: explica que la información debe contrastarse con otras fuentes.
- Pista 2: indica que hay que buscar coincidencias y diferencias entre documentos para valorar si una información es fiable.

Si la tarea combina dos estrategias, por ejemplo Integración y Navegación:
- Combina ambas ayudas de forma natural.
- La pista 1 debe explicar el formato general: por ejemplo, que la información puede estar organizada en pestañas, secciones o varios textos.
- La pista 2 debe orientar al estudiante sobre qué debe buscar, leer o comparar, sin darle la respuesta.

REGLAS DE TONO Y ESTILO:
- Dirígete al estudiante de forma clara, cercana y directa.
- No uses viñetas ni numeración dentro del texto de las pistas.
- No resuelvas la tarea.
- No menciones la respuesta correcta.
- No copies literalmente el enunciado base dentro de las pistas.
- No añadas información que no esté relacionada con la tarea.
- Mantén cada pista en una o dos frases como máximo.
- Conserva el sentido original del enunciado base.
- El enunciado de nivel medio y el de nivel bajo deben sonar naturales, como si hubieran sido redactados directamente para el estudiante.

FORMATO DE SALIDA:
Devuelve únicamente un JSON válido con esta estructura exacta:

{{
  "pista_1": "...",
  "pista_2": "...",
  "enunciado_nivel_medio": "...",
  "enunciado_nivel_bajo": "..."
}}

EJEMPLOS:

Ejemplo 1:
Enunciado base: "Observa la página web y haz clic en el apartado adecuado para encontrar información sobre las actividades del museo."
Estrategia: navegar
Subestrategia: localizar estructura/menús
Salida:
{{
  "pista_1": "Las páginas web suelen organizar la información mediante menús, pestañas o secciones para que sea más fácil encontrar lo que buscamos.",
  "pista_2": "Fíjate en apartados que puedan estar relacionados con actividades, visitas, programación o información práctica.",
  "enunciado_nivel_medio": "Las páginas web suelen organizar la información mediante menús, pestañas o secciones para que sea más fácil encontrar lo que buscamos. Observa la página web y haz clic en el apartado adecuado para encontrar información sobre las actividades del museo.",
  "enunciado_nivel_bajo": "Las páginas web suelen organizar la información mediante menús, pestañas o secciones para que sea más fácil encontrar lo que buscamos. Fíjate en apartados que puedan estar relacionados con actividades, visitas, programación o información práctica. Observa la página web y haz clic en el apartado adecuado para encontrar información sobre las actividades del museo."
}}

Ejemplo 2:
Enunciado base: "¿Qué página escogerías para informarte sobre este tema? Justifica tu respuesta."
Estrategia: evaluar
Subestrategia: experiencia de las fuentes
Salida:
{{
  "pista_1": "En internet no toda la información tiene la misma calidad, porque cualquier persona puede publicar contenido.",
  "pista_2": "Para elegir una fuente fiable, fíjate en si pertenece a una institución oficial, a expertos en el tema o si ofrece información verificable.",
  "enunciado_nivel_medio": "En internet no toda la información tiene la misma calidad, porque cualquier persona puede publicar contenido. ¿Qué página escogerías para informarte sobre este tema? Justifica tu respuesta.",
  "enunciado_nivel_bajo": "En internet no toda la información tiene la misma calidad, porque cualquier persona puede publicar contenido. Para elegir una fuente fiable, fíjate en si pertenece a una institución oficial, a expertos en el tema o si ofrece información verificable. ¿Qué página escogerías para informarte sobre este tema? Justifica tu respuesta."
}}

Ejemplo 3:
Enunciado base: "Lee los dos textos y explica si ofrecen la misma información o si presentan ideas diferentes."
Estrategia: integrar
Subestrategia: multitexto
Salida:
{{
  "pista_1": "En internet podemos encontrar mucha información en distintos textos, y esta información puede ser similar, complementaria o contradictoria.",
  "pista_2": "Lee ambos textos con atención y compara sus ideas principales, aunque estén escritas con palabras diferentes.",
  "enunciado_nivel_medio": "En internet podemos encontrar mucha información en distintos textos, y esta información puede ser similar, complementaria o contradictoria. Lee los dos textos y explica si ofrecen la misma información o si presentan ideas diferentes.",
  "enunciado_nivel_bajo": "En internet podemos encontrar mucha información en distintos textos, y esta información puede ser similar, complementaria o contradictoria. Lee ambos textos con atención y compara sus ideas principales, aunque estén escritas con palabras diferentes. Lee los dos textos y explica si ofrecen la misma información o si presentan ideas diferentes."
}}
""".strip()


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
    """Construye un contexto textual simple a partir de la tarea.

    De momento no hace OCR ni procesa imágenes. Si en el futuro tienes OCR o una
    descripción de la imagen, pásala en `contexto_extraido` dentro de la tarea o
    directamente a `generar_enunciados_adaptados`.
    """
    partes: List[str] = []

    if tarea.get("contexto_extraido"):
        partes.append(str(tarea["contexto_extraido"]))

    if tarea.get("task_text"):
        partes.append(f"Texto de apoyo: {tarea['task_text']}")

    imagenes = _get_task_images(tarea)
    if imagenes:
        partes.append("Imágenes asociadas: " + ", ".join(imagenes))

    if tarea.get("description"):
        partes.append(f"Descripción interna de la tarea: {tarea['description']}")

    return " | ".join(partes) if partes else "No se proporciona contexto adicional."


def construir_prompt_generador(
    tarea: Dict[str, Any],
    contexto_extraido: Optional[str] = None,
) -> str:
    """Rellena el prompt del generador para una tarea concreta."""
    return PROMPT_GENERADOR_ENUNCIADOS.format(
        contexto_extraido=contexto_extraido or construir_contexto_extraido(tarea),
        enunciado_base=_get_enunciado_base(tarea),
        estrategia=_get_estrategia(tarea),
        subestrategia=_get_subestrategia(tarea),
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
    """Convierte la respuesta del LLM a un diccionario validado."""
    if isinstance(raw_output, LLMGeneradorEnunciadosOutput):
        return raw_output.model_dump()

    if isinstance(raw_output, dict):
        return LLMGeneradorEnunciadosOutput(**raw_output).model_dump()

    content = getattr(raw_output, "content", raw_output)
    if not isinstance(content, str):
        raise TypeError(f"Tipo de respuesta no soportado: {type(raw_output)}")

    data = _extraer_json_de_respuesta(content)
    return LLMGeneradorEnunciadosOutput(**data).model_dump()


def generar_enunciados_adaptados(
    tarea: Dict[str, Any],
    modelo: str = "poligpt",
    contexto_extraido: Optional[str] = None,
    usar_salida_estructurada: bool = True,
) -> Dict[str, Any]:
    """Genera pistas y enunciados adaptados para una tarea.

    Parameters
    ----------
    tarea:
        Diccionario de tarea. Puede tener el formato original de `tareas2.json` o
        el formato de validación con `enunciado_nivel_alto`.
    modelo:
        Nombre del modelo que se pasará a `get_llm(..., override_model=modelo)`.
    contexto_extraido:
        Texto adicional opcional: OCR, descripción de imágenes o información ya
        procesada del material de la tarea.
    usar_salida_estructurada:
        Si es True, intenta usar `with_structured_output`. Si el modelo/backend no
        lo soporta, hace fallback automático a salida JSON normal.

    Returns
    -------
    Dict[str, Any]
        Diccionario con `pista_1`, `pista_2`, `enunciado_nivel_medio`,
        `enunciado_nivel_bajo` y `tiempo_respuesta`.
    """
    prompt = construir_prompt_generador(tarea, contexto_extraido=contexto_extraido)
    # Importación diferida para que el módulo pueda cargarse aunque el entorno de
    # evaluación no tenga instaladas todavía las dependencias de LangChain.
    from configuracion_llm import get_llm

    llm = get_llm(tipo_tarea="generador", override_model=modelo)

    inicio = time.perf_counter()
    ultimo_error: Optional[Exception] = None

    if usar_salida_estructurada:
        try:
            llm_estructurado = llm.with_structured_output(LLMGeneradorEnunciadosOutput)
            raw_output = llm_estructurado.invoke(prompt)
            resultado = _normalizar_salida(raw_output)
            resultado["tiempo_respuesta"] = time.perf_counter() - inicio
            return resultado
        except Exception as exc:  # noqa: BLE001 - queremos fallback robusto para benchmarking
            ultimo_error = exc

    try:
        raw_output = llm.invoke(prompt)
        resultado = _normalizar_salida(raw_output)
        resultado["tiempo_respuesta"] = time.perf_counter() - inicio
        return resultado
    except (ValidationError, json.JSONDecodeError, ValueError, TypeError) as exc:
        detalle = f"Error al generar/parsear la salida con el modelo {modelo}: {exc}"
        if ultimo_error is not None:
            detalle += f" | Error previo con salida estructurada: {ultimo_error}"
        raise RuntimeError(detalle) from exc


def generar_task_prompt_adaptado(
    tarea: Dict[str, Any],
    modelo: str = "poligpt",
    contexto_extraido: Optional[str] = None,
) -> Dict[str, Any]:
    """Devuelve una copia de la tarea con `task_prompt` actualizado.

    Esta función es la que más adelante se puede utilizar al cargar una sesión nueva:
    conserva el nivel alto original y sustituye/genera nivel medio y bajo.
    """
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
    """Genera enunciados adaptados para una lista de tareas."""
    return [generar_task_prompt_adaptado(tarea, modelo=modelo) for tarea in tareas]


if __name__ == "__main__":
    # Prueba rápida manual:
    # python generador_enunciados.py tareas_seleccionadas_pistas_enunciados.json poligpt
    import argparse

    parser = argparse.ArgumentParser(description="Genera pistas y enunciados adaptados para la primera tarea de un JSON.")
    parser.add_argument("input_json", help="Ruta del JSON con tareas.")
    parser.add_argument("modelo", nargs="?", default="poligpt", help="Modelo a utilizar.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8-sig") as f:
        tareas = json.load(f)

    if not isinstance(tareas, list) or not tareas:
        raise ValueError("El JSON de entrada debe ser una lista de tareas no vacía.")

    salida = generar_enunciados_adaptados(tareas[0], modelo=args.modelo)
    print(json.dumps(salida, ensure_ascii=False, indent=2))
