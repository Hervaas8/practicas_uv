import json
import yaml
import sys
import base64
import os
from PIL import Image
import io
from sistema.utils.configuracion_llm import get_llm_evaluador, is_vision_model
from langchain_core.messages import HumanMessage
from sistema.config import PROMPTS_PATH, resolve_project_path

# --- IMPORTACIÓN DE UTILIDADES OCR ---
try:
    from sistema.utils.utils_easy import extract_text_with_llm 
except ImportError:
    def extract_text_with_llm(path): return "[ERROR: Módulo OCR no encontrado]"

# --- CARGAR CONFIGURACIÓN GLOBAL ---
try:
    with open(PROMPTS_PATH, "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    PROMPT_TEMPLATE = prompts["eval_subjective_prompt"]
    PROMPT_FEEDBACK_FINAL = prompts["eval_strategy_feedback_prompt"]
except Exception as e:
    print(f"Error crítico al cargar configuración en motor_evaluacion: {e}")
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
            
            # Guardar en memoria como JPEG con algo de compresión
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None

def select_feedback(task: dict, normalized_score: float):
    """Selecciona feedback basado en el porcentaje de éxito (0.0 a 1.0)."""
    is_success = normalized_score >= 0.5
    key = "success" if is_success else "failure"
    feedback_dict = task.get("feedback", {})
    return feedback_dict.get(key, "Feedback no configurado.")

def evaluate_task(task: dict, student_answer: any, student_level: str, override_model: str = None):
    task_id = task.get('task_id', 'Desconocida')
    task_type = task.get('task_type')
    
    # Resolver el prompt dependiendo de si es diccionario
    prompt_data = task.get("task_prompt", "")
    if isinstance(prompt_data, dict):
        nivel_map = {"nivel_1": "nivel_bajo", "nivel_2": "nivel_medio", "nivel_3": "nivel_alto"}
        task_prompt_str = prompt_data.get(nivel_map.get(student_level, "nivel_bajo"), "")
    else:
        task_prompt_str = str(prompt_data)
    
    score = 0.0
    max_score = 1.0 
    reasoning = ""
    scores_por_item = {}

    # --- TIPO: CONECTAR ---
    if task_type == "conectar":
        correct_mapping = task.get("correct_mapping", {})
        matches = 0
        total_required = sum(len(defs) if isinstance(defs, list) else 1 for defs in correct_mapping.values())
        for concept, correct_defs in correct_mapping.items():
            st_defs = student_answer.get(concept, [])
            if isinstance(st_defs, str): st_defs = [st_defs]
            for d in st_defs:
                if d in (correct_defs if isinstance(correct_defs, list) else [correct_defs]):
                    matches += 1
        score = matches / total_required if total_required > 0 else 0
        reasoning = f"Auto-corregido: {matches} de {total_required} conexiones."

    # --- TIPO: OBJETIVO / NAVEGACION_TEST ---
    elif task_type in ["objetivo", "navegacion_test"]:
        correct_answer = task.get("correct_answer")
        def normalize(v): return [str(x).strip().lower() for x in (v if isinstance(v, list) else [v])]
        correct_list = normalize(correct_answer)
        student_list = normalize(student_answer)
        matches = sum(1 for i, v in enumerate(correct_list) if i < len(student_list) and student_list[i] == v)
        score = matches / len(correct_list) if correct_list else 0
        reasoning = f"Auto-corregido: {matches}/{len(correct_list)} aciertos."

    # --- TIPO: INTERACTIVO ---
    elif task_type == "interactivo":
        import re
        match = re.search(r"CLICK: (\d+), (\d+)", str(student_answer))
        if match:
            click_x, click_y = int(match.group(1)), int(match.group(2))
            regions = task.get("regions", [])
            acierto = False
            for reg in regions:
                rx, ry, rw, rh = reg['x'], reg['y'], reg['w'], reg['h']
                if rx <= click_x <= rx + rw and ry <= click_y <= ry + rh:
                    if reg.get("is_correct"):
                        acierto = True; break
            score = 1.0 if acierto else 0.0
            reasoning = f"Clic en ({click_x}, {click_y}). {'Correcto' if acierto else 'Fuera de zona'}."
        else:
            score, reasoning = 0.0, "Coordenadas no válidas."

    # --- TIPO: TABLA GRID ---
    elif task_type == "tabla_grid":
        filas = task.get("filas", [])
        matches = 0
        for fila in filas:
            motivo = fila.get("motivo", "")
            respuesta_esperada = str(fila.get("correcta", "")).strip().upper()
            if isinstance(student_answer, dict):
                respuesta_alumno = str(student_answer.get(motivo, "")).strip().upper()
                if respuesta_esperada == respuesta_alumno:
                    matches += 1
        raw_score = matches / len(filas) if filas else 0
        score = 1.0 if raw_score > 0.5 else 0.0
        reasoning = f"Auto-corregido (Tabla Grid): {matches} de {len(filas)} aciertos. Nota ajustada a {score}."

# --- TIPO: SUBJETIVO / V_Y_D / TABLAS (Evaluación Multi-Objetivo) ---
    else:
        rubric = task.get("rubric", {})
        
        # 1. Contar dinámicamente los apartados a evaluar
        # Al tener todas las rúbricas estandarizadas, solo contamos las claves que son diccionarios
        if isinstance(rubric, dict):
            items_evaluables = [k for k, v in rubric.items() if isinstance(v, dict)]
            max_score = float(len(items_evaluables))
        else:
            max_score = 1.0 # Fallback de seguridad
            items_evaluables = []

        # Lógica Híbrida: Visión vs OCR
        modelo_activo = override_model if override_model else "poligpt-vision"
        es_vision = is_vision_model(modelo_activo)

        rutas_imagenes = task.get("task_image", [])
        if isinstance(rutas_imagenes, str): rutas_imagenes = [rutas_imagenes]

        # --- RESTRICCIÓN ESPECÍFICA: SOLO QWEN SE PASA A OCR SI HAY >5 IMÁGENES ---
        if es_vision and len(rutas_imagenes) > 5 and "qwen" in modelo_activo.lower():
            # Apagamos el modo visión solo para los modelos Qwen en esta tarea
            es_vision = False

        # Cargamos el LLM
        llm_evaluador = get_llm_evaluador(override_model=modelo_activo)

        texto_ocr_adicional = ""
        
        # Si NO es visión (o si lo acabamos de apagar por ser un Qwen sobrecargado)
        if not es_vision:
            partes_ocr = []
            for img_path in rutas_imagenes:
                if img_path: partes_ocr.append(extract_text_with_llm(img_path))
            texto_ocr_adicional = "\n\n[TEXTO EXTRAÍDO DE LAS IMÁGENES MEDIANTE OCR]:\n" + "\n".join(filter(None, partes_ocr))
            
        texto_base = task.get("task_text", "")
        if texto_base is None: texto_base = ""
        contexto_completo = texto_base + texto_ocr_adicional

        # 2. Instrucción dinámica unificada para el LLM
        texto_apartados = "apartado" if max_score == 1.0 else "apartados"
        nota_llm = (
            f"NOTA CRÍTICA: Esta tarea tiene {int(max_score)} {texto_apartados} a evaluar según la rúbrica. "
            f"Evalúa CADA UNO por separado. "
            f"Es ESTRICTAMENTE OBLIGATORIO que tu JSON de salida incluya el campo 'scores_por_item' "
            f"conteniendo un diccionario con la nota asignada a cada apartado (usando las claves exactas de la rúbrica). "
            f"Si uno de los apartados la clave se llama Instrucciones, no es un apartado a evaluar, simplemente es información adicional de la rúbrica."
            f"(Total máximo: {max_score})"
        )

        instrucciones = PROMPT_TEMPLATE.replace("{task_text}", contexto_completo)\
                                       .replace("{rubrica_text}", json.dumps(rubric, ensure_ascii=False) if isinstance(rubric, dict) else str(rubric))\
                                       .replace("{task_prompt}", f"{task_prompt_str}. {nota_llm}")\
                                       .replace("{student_answer}", str(student_answer))

        content = [{"type": "text", "text": instrucciones}]

        if es_vision and rutas_imagenes:
            for img_path in rutas_imagenes:
                img_b64 = encode_image(img_path)
                if img_b64:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        msg = HumanMessage(content=content)

        try:
            resultado = llm_evaluador.invoke([msg])
            scores_por_item = resultado.scores_por_item
            
            # Sumamos las notas de forma segura
            if scores_por_item:
                score = float(sum(scores_por_item.values()))
            else:
                score = round(resultado.score, 2)
                
            reasoning = resultado.reasoning
        except Exception as e:
            print(f"Error LLM: {e}")
            return None

    return {
        "score": score,
        "max_score": max_score,
        "scores_por_item": scores_por_item, 
        "reasoning": reasoning,
        "feedback_shown": select_feedback(task, score / max_score if max_score > 0 else 0)
    }

# --- 1. DICCIONARIO DE SUB-ESTRATEGIAS ---
MAPA_SUBESTRATEGIAS = {
    "N1": "Identificar la estructura del texto online",
    "N2": "Determinar relevancia de la info",
    "I1": "Combinar información complementaria",
    "I2": "Enfrentarse al conflicto intertextual",
    "E1": "Analizar expertise del autor",
    "E2": "Analizar benevolencia"
}

def generate_strategy_feedback(strategy_name: str, level: str, history: list) -> str:
    # Lógica inalterada, pero adaptada si quieres
    def get_val(item, key, default=""):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    strat_history = [h for h in history if get_val(h, 'strategy') == strategy_name and not get_val(h, 'is_test_task', False)]
    if not strat_history: return "No hay suficientes datos."

    sub_vistas = set()
    historial_texto = ""
    for i, h in enumerate(strat_history):
        sub_code = get_val(h, 'sub_strategy', "General")
        if sub_code != "General" and sub_code: sub_vistas.add(f"- {sub_code}: {MAPA_SUBESTRATEGIAS.get(sub_code, 'Habilidad general')}")
        historial_texto += f"Tarea {i+1}: Respuesta: {get_val(h, 'student_response')} | Nota: {get_val(h, 'score', 0)}/{get_val(h, 'max_score', 1)}\n"

    from sistema.utils.configuracion_llm import get_llm_estructurado
    from langchain_core.prompts import ChatPromptTemplate
    prompt_tmpl = ChatPromptTemplate.from_template(PROMPT_FEEDBACK_FINAL)
    llm_feedback = get_llm_estructurado("feedback_estrategia") 
    chain = prompt_tmpl | llm_feedback 
    try:
        res = chain.invoke({"estrategia": strategy_name.capitalize(), "nivel": level, "sub_info": "\n".join(sub_vistas), "historial_texto": historial_texto.strip()})
        return res.texto_feedback
    except Exception: return "Ha habido un problema al generar tus consideraciones."