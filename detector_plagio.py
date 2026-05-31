import re
import itertools
import yaml
import sys
from configuracion_llm import get_llm_estructurado
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# --- CARGAR CONFIGURACIÓN DE PROMPTS ---
# ==========================================
try:
    with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_PLAGIO = prompts["plagio_prompt"]
except Exception as e:
    print(f"Error crítico al cargar prompts.yaml en detector_plagio: {e}")
    sys.exit(1)

# ==========================================
# 1. MOTOR MATEMÁTICO Y PREPROCESAMIENTO
# ==========================================

def limpiar_texto(texto: str) -> list:
    """Convierte a minúsculas, quita puntuación y devuelve lista de palabras."""
    if not isinstance(texto, str):
        return []
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

def verificar_plagio_llm(pregunta: str, resp_a: str, resp_b: str):
    """Pasa los casos sospechosos a PoliGPT para obtener el veredicto estructurado."""
    llm_detective = get_llm_estructurado("plagio")
    
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
        print(f"Error en LLM Plagio: {e}")
        return {
            "veredicto": "Sospechoso", 
            "justificacion": "Alta similitud matemática, pero el LLM no pudo verificarlo por un error técnico.", 
            "frases_identicas": []
        }

# ==========================================
# 3. ORQUESTADOR PRINCIPAL
# ==========================================

def auditar_sesion_completa(session_data: list, umbral: float = 0.70, min_palabras: int = 8) -> dict:
    """
    Analiza todas las respuestas abiertas de una sesión y devuelve un reporte estructurado.
    Formato de salida: { "task_id": { "prompt": "...", "alertas": [ {alerta1}, ... ] } }
    """
    tareas_abiertas = {}

    # 1. Extraer y agrupar respuestas por tarea
    for estudiante in session_data:
        nombre = estudiante.get('alumno_name', 'Desconocido')
        historial = estudiante.get('task_history', [])
        
        for tarea in historial:
            t_id = tarea.get('task_id') if isinstance(tarea, dict) else getattr(tarea, 'task_id', None)
            t_prompt = tarea.get('task_prompt') if isinstance(tarea, dict) else getattr(tarea, 'task_prompt', '')
            t_resp = tarea.get('student_response') if isinstance(tarea, dict) else getattr(tarea, 'student_response', '')
            
            if isinstance(t_resp, str):
                palabras = limpiar_texto(t_resp)
                if len(palabras) >= min_palabras:
                    if t_id not in tareas_abiertas:
                        tareas_abiertas[t_id] = {"prompt": t_prompt, "respuestas": []}
                    
                    tareas_abiertas[t_id]["respuestas"].append({
                        "alumno": nombre,
                        "texto": t_resp,
                        "palabras": palabras
                    })

    # 2. Comparar alumnos para cada tarea
    reporte_final = {}
    
    for t_id, data in tareas_abiertas.items():
        respuestas = data["respuestas"]
        prompt = data["prompt"]
        alertas = []
        
        for r1, r2 in itertools.combinations(respuestas, 2):
            sim_trigramas = calcular_trigramas(r1["palabras"], r2["palabras"])
            sim_levenshtein = calcular_levenshtein_palabras(r1["palabras"], r2["palabras"])
            
            sim_maxima = max(sim_trigramas, sim_levenshtein)
            
            if sim_maxima >= umbral:
                analisis_llm = verificar_plagio_llm(prompt, r1["texto"], r2["texto"])
                
                alertas.append({
                        "alumno_1": r1["alumno"],
                        "alumno_2": r2["alumno"],
                        "similitud_matematica": round(sim_maxima * 100, 1),
                        "respuesta_1": r1["texto"],
                        "respuesta_2": r2["texto"],
                        "analisis": analisis_llm
                    })
        
        if alertas:
            reporte_final[t_id] = {
                "prompt": prompt,
                "alertas": sorted(alertas, key=lambda x: x["similitud_matematica"], reverse=True)
            }

    return reporte_final