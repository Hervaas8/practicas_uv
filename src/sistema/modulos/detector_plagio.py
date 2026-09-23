import re
import unicodedata
import itertools
import yaml
import sys
from sistema.utils.configuracion_llm import get_llm_estructurado
from langchain_core.prompts import ChatPromptTemplate
from sistema.config import PROMPTS_PATH

# ==========================================
# --- CARGAR CONFIGURACIÓN DE PROMPTS ---
# ==========================================
try:
    with open(PROMPTS_PATH, "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_PLAGIO = prompts["plagio_prompt"]
except Exception as e:
    print(f"Error crítico al cargar prompts.yaml en detector_plagio: {e}")
    sys.exit(1)

# ==========================================
# 1. MOTOR MATEMÁTICO Y PREPROCESAMIENTO
# ==========================================

def limpiar_texto(texto: str) -> list:
    """
    Convierte a minúsculas, elimina tildes y diacríticos, quita puntuación
    y devuelve lista de palabras normalizada.
    """
    if not isinstance(texto, str):
        return []
    # Normalizar tildes y diacríticos (á→a, é→e, ñ→n, ü→u, etc.)
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    # Minúsculas y eliminar puntuación
    texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
    return texto_limpio.split()

def calcular_trigramas(lista_a: list, lista_b: list) -> float:
    """Calcula la similitud usando el Índice de Jaccard sobre bloques de 3 palabras."""
    if len(lista_a) < 3 or len(lista_b) < 3:
        return 0.0

    tri_a = set([" ".join(lista_a[i:i+3]) for i in range(len(lista_a)-2)])
    tri_b = set([" ".join(lista_b[i:i+3]) for i in range(len(lista_b)-2)])

    interseccion = len(tri_a & tri_b)
    union = len(tri_a | tri_b)

    return interseccion / union if union > 0 else 0.0

def calcular_levenshtein_palabras(lista_a: list, lista_b: list) -> float:
    """Calcula similitud basada en la distancia de edición por palabras enteras."""
    if not lista_a or not lista_b:
        return 0.0

    filas = len(lista_a) + 1
    cols = len(lista_b) + 1
    dist = [[0 for _ in range(cols)] for _ in range(filas)]

    for i in range(1, filas): dist[i][0] = i
    for j in range(1, cols): dist[0][j] = j

    for col in range(1, cols):
        for fila in range(1, filas):
            coste = 0 if lista_a[fila-1] == lista_b[col-1] else 1
            dist[fila][col] = min(
                dist[fila-1][col] + 1,
                dist[fila][col-1] + 1,
                dist[fila-1][col-1] + coste
            )

    distancia_final = dist[filas-1][cols-1]
    max_longitud = max(len(lista_a), len(lista_b))

    return 1.0 - (distancia_final / max_longitud)

# ==========================================
# 2. VERIFICACIÓN CON LLM (El Árbitro)
# ==========================================

def verificar_plagio_llm(pregunta: str, resp_a: str, resp_b: str, modelo: str = None):
    """Pasa los casos sospechosos a PoliGPT para obtener el veredicto estructurado."""
    # Añadimos override_model=modelo para que cambie de LLM
    llm_detective = get_llm_estructurado("plagio", override_model=modelo)
    
    # Preparamos el template usando LangChain
    prompt_tmpl = ChatPromptTemplate.from_template(PROMPT_PLAGIO)
    
    # Conectamos el prompt con el LLM estructurado
    chain = prompt_tmpl | llm_detective
    
    try:
        # Invocamos la cadena pasándole las variables necesarias
        resultado = chain.invoke({
            "pregunta": pregunta,
            "resp_a": resp_a,
            "resp_b": resp_b
        })
        return resultado.dict() 
    except Exception as e:
        print(f"Error en LLM Plagio ({modelo}): {e}")
        return {
            "veredicto": "Sospechoso", 
            "justificacion": f"Alta similitud matemática, pero el LLM {modelo} no pudo verificarlo por un error técnico.", 
            "frases_identicas": []
        }

# ==========================================
# 3. ORQUESTADOR PRINCIPAL
# ==========================================

def auditar_sesion_completa(
    session_data: list,
    umbral: float = 0.25,
    min_palabras: int = 15,
    valid_task_ids: list = None  # <-- PARÁMETRO AÑADIDO
) -> dict:
    """
    Analiza todas las respuestas abiertas de una sesión y devuelve un reporte estructurado.

    Lógica de detección:
      - Respuestas con menos de min_palabras (default 15): van directamente al LLM,
        ya que los cálculos matemáticos son poco fiables con tan pocos tokens.
      - Respuestas con min_palabras o más: primero pasan por el filtro matemático
        (trigramas + Levenshtein). Solo si superan el umbral se llama al LLM.

    Formato de salida:
      { "task_id": { "prompt": "...", "alertas": [ {alerta1}, ... ] } }
    """
    tareas_abiertas = {}

    # 1. Extraer y agrupar respuestas por tarea
    for estudiante in session_data:
        nombre = estudiante.get('alumno_name', 'Desconocido')
        historial = estudiante.get('task_history', [])

        for tarea in historial:
            t_id     = tarea.get('task_id')     if isinstance(tarea, dict) else getattr(tarea, 'task_id', None)
            
            # --- FILTRO DE TAREAS SUBJETIVAS ---
            # Si nos pasan una lista de valid_task_ids y la tarea no está en ella, la ignoramos.
            if valid_task_ids is not None and t_id not in valid_task_ids:
                continue
            # -----------------------------------

            t_prompt = tarea.get('task_prompt') if isinstance(tarea, dict) else getattr(tarea, 'task_prompt', '')
            t_resp   = tarea.get('student_response') if isinstance(tarea, dict) else getattr(tarea, 'student_response', '')

            if isinstance(t_resp, str):
                palabras = limpiar_texto(t_resp)
                # Incluimos la respuesta independientemente de su longitud;
                # el enrutamiento se decide en la fase de comparación.
                if len(palabras) > 0:
                    if t_id not in tareas_abiertas:
                        tareas_abiertas[t_id] = {"prompt": t_prompt, "respuestas": []}

                    tareas_abiertas[t_id]["respuestas"].append({
                        "alumno": nombre,
                        "texto": t_resp,         # texto original (sin limpiar) → para el LLM
                        "palabras": palabras,    # tokens normalizados → para cálculos matemáticos
                    })

    # 2. Comparar alumnos para cada tarea
    reporte_final = {}

    for t_id, data in tareas_abiertas.items():
        respuestas = data["respuestas"]
        prompt = data["prompt"]
        alertas = []

        for r1, r2 in itertools.combinations(respuestas, 2):
            n1 = len(r1["palabras"])
            n2 = len(r2["palabras"])
            es_par_corto = n1 < min_palabras or n2 < min_palabras

            if es_par_corto:
                # --- RUTA DIRECTA AL LLM ---
                # Los cálculos matemáticos son poco fiables con respuestas muy cortas.
                # Delegamos la decisión completa al LLM.
                analisis_llm = verificar_plagio_llm(prompt, r1["texto"], r2["texto"])
                alertas.append({
                    "alumno_1": r1["alumno"],
                    "alumno_2": r2["alumno"],
                    "similitud_matematica": None,   # no aplica
                    "ruta": "llm_directo",
                    "respuesta_1": r1["texto"],
                    "respuesta_2": r2["texto"],
                    "analisis": analisis_llm
                })

            else:
                # --- RUTA NORMAL: filtro matemático → LLM si supera umbral ---
                sim_trigramas   = calcular_trigramas(r1["palabras"], r2["palabras"])
                sim_levenshtein = calcular_levenshtein_palabras(r1["palabras"], r2["palabras"])
                sim_maxima      = max(sim_trigramas, sim_levenshtein)

                if sim_maxima >= umbral:
                    analisis_llm = verificar_plagio_llm(prompt, r1["texto"], r2["texto"])
                    alertas.append({
                        "alumno_1": r1["alumno"],
                        "alumno_2": r2["alumno"],
                        "similitud_matematica": round(sim_maxima * 100, 1),
                        "ruta": "filtro_matematico",
                        "respuesta_1": r1["texto"],
                        "respuesta_2": r2["texto"],
                        "analisis": analisis_llm
                    })

        if alertas:
            reporte_final[t_id] = {
                "prompt": prompt,
                "alertas": sorted(
                    alertas,
                    key=lambda x: x["similitud_matematica"] if x["similitud_matematica"] is not None else 101,
                    reverse=True
                )
            }

    return reporte_final