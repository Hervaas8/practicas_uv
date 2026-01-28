import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid
import os
import re
from typing import List, Union, Tuple

# Importamos tu grafo y esquemas
from agente2 import app
from esquemas import StudentGraphState

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Tutor IA Adaptativo", page_icon="🎓")

st.title("🎓 Tutor IA de Ciencias")
st.markdown("""
Este agente adapta el nivel de dificultad y el andamiaje según tus respuestas.
**Materia:** Ciencias | **Sesión:** 1
""")

# --- GESTIÓN DEL ESTADO ---

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interactive_state" not in st.session_state:
    st.session_state.interactive_state = {
        "active": False,
        "current_step_index": 0,
        "correct_clicks": 0,
        "total_attempts_in_step": 0, # Para saber si falló antes de acertar
        "history_log": [] # Para enviar al agente al final
    }
# --- FUNCIONES DE LIMPIEZA Y TEXTO ---

def normalize_images(image_data: Union[str, List[str], None]) -> List[str]:
    if not image_data: return []
    if isinstance(image_data, str): return [image_data]
    return image_data

def split_text_and_options(text: str) -> Tuple[str, List[str]]:
    """
    Separa el enunciado de las opciones buscando líneas que empiecen por 'a)', 'b.', etc.
    """
    if not text: return "", []
    
    # Regex Multilínea: Busca al inicio de cualquier línea (^) una letra seguida de ) o .
    # (?m) activa el modo multilínea.
    pattern = r"(?m)^\s*([a-zA-Z][\)\.])\s+(.*)"
    
    # Buscamos todas las coincidencias
    matches = list(re.finditer(pattern, text))
    
    if len(matches) >= 2:
        # El texto limpio llega hasta donde empieza la primera opción
        first_match_start = matches[0].start()
        clean_text = text[:first_match_start].strip()
        
        # Reconstruimos las opciones completas (Letra + Texto)
        options = [m.group(0).strip() for m in matches]
        return clean_text, options
    
    return text, []

def render_message(msg, is_last_message=False):
    """
    Pinta el mensaje.
    - is_last_message: Si es True, NO pintamos las opciones como texto (porque saldrán los botones).
    """
    with st.chat_message(msg["role"]):
        content = msg.get("content", "")
        
        # 1. Separamos texto y opciones
        clean_text, options = split_text_and_options(content)
        
        # 2. Pintamos el texto limpio (sin las opciones a,b,c...)
        st.markdown(clean_text)
        
        # 3. GESTIÓN DE OPCIONES VISUALES
        # Si NO es el último mensaje (es historial), pintamos las opciones como lista bonita
        if options and not is_last_message and msg["role"] == "assistant":
            formatted_options = "\n".join([f"- {opt}" for opt in options])
            st.markdown(f"**Opciones:**\n{formatted_options}")
        
        # 4. Contexto Adicional (Texto largo)
        if msg.get("context_text"):
            with st.expander("📄 Ver texto de referencia", expanded=True):
                st.info(msg["context_text"])
                
        # 5. Imágenes
        images = msg.get("image_paths", [])
        if images:
            for i, img_path in enumerate(images):
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"Recurso visual {i+1}")
                elif img_path:
                    st.warning(f"⚠️ No se encontró la imagen: {img_path}")

def handle_user_input(user_response_content):
    """Procesa la respuesta y avanza el grafo."""
    
    user_msg = {"role": "user", "content": user_response_content}
    st.session_state.messages.append(user_msg)
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    app.update_state(config, {"current_student_response": user_response_content})
    
    with st.spinner("El tutor está evaluando tu respuesta..."):
        for event in app.stream(None, config=config): pass
            
    new_state = app.get_state(config).values
    
    # Verificamos si antes estábamos en test y ahora ya no
    is_in_assessment = new_state.get("in_initial_assessment", False)
    
    if "was_in_assessment" not in st.session_state:
        st.session_state.was_in_assessment = False # Valor inicial

    # SI: Estábamos en test (True) Y Ahora ya no (False) -> Significa que acabó
    if st.session_state.was_in_assessment and not is_in_assessment:
        st.session_state.messages = [] # ¡BORRAMOS TODO EL HISTORIAL!
        st.toast("Test finalizado. Iniciando sesión...", icon="🧹")
    
    # Actualizamos el chivato para la próxima vez
    st.session_state.was_in_assessment = is_in_assessment
    # Construir respuesta
    agent_text = ""
    feedback = new_state.get("last_feedback_message")
    if feedback:
        agent_text += f"**Feedback:**\n{feedback}\n\n---\n\n"
        
    current_task = new_state.get("current_task")
    next_prompt = new_state.get("current_task_prompt")
    
    if not current_task:
        agent_text += "🎉 **¡Has completado la sesión!**"
    elif next_prompt:
        agent_text += f"{next_prompt}"
    
    task_text = current_task.get("task_text") if current_task else None
    task_imgs = normalize_images(current_task.get("task_image")) if current_task else []

    if agent_text:
        agent_msg = {
            "role": "assistant",
            "content": agent_text,
            "context_text": task_text,
            "image_paths": task_imgs
        }
        st.session_state.messages.append(agent_msg)
    
    st.rerun()
def check_click(coords, regions):
    """Verifica si el clic (x,y) cae dentro de alguna región definida."""
    if not coords: return None
    
    click_x, click_y = coords['x'], coords['y']
    
    for region in regions:
        # Asumimos formato caja: x, y, w (ancho), h (alto)
        rx, ry, rw, rh = region['x'], region['y'], region['w'], region['h']
        
        if rx <= click_x <= rx + rw and ry <= click_y <= ry + rh:
            return region
            
    return None

def handle_interactive_task(task):
    """Gestiona la lógica de imágenes encadenadas."""
    
    state = st.session_state.interactive_state
    steps = task.get("steps", [])
    
    # Si no hay pasos, error
    if not steps:
        st.error("Error: Tarea interactiva sin pasos definidos.")
        return

    # 1. Obtener el paso actual
    current_step_idx = state["current_step_index"]
    
    # SI HEMOS TERMINADO TODOS LOS PASOS
    if current_step_idx >= len(steps):
        # Calculamos nota final (Proporción de aciertos a la primera)
        total_steps = len(steps)
        score_ratio = state["correct_clicks"] / total_steps
        
        final_response_text = (
            f"Tarea Interactiva Finalizada.\n"
            f"Aciertos puros: {state['correct_clicks']}/{total_steps}\n"
            f"Log: {state['history_log']}"
        )
        
        # Enviamos esto al Agente como si el usuario lo hubiera escrito
        # El agente (evaluate_task) deberá interpretar este texto o la nota.
        # Aquí forzamos un envío automático:
        st.success("¡Tarea completada! Procesando resultados...")
        
        # Reseteamos estado interno para la próxima vez
        st.session_state.interactive_state = {
            "active": False, "current_step_index": 0, 
            "correct_clicks": 0, "total_attempts_in_step": 0, "history_log": []
        }
        
        # Llamamos a la función principal de envío
        handle_user_input(final_response_text)
        return

    # 2. Mostrar la imagen del paso actual
    step_data = steps[current_step_idx]
    image_path = step_data.get("image")

    
    # 3. COMPONENTE INTERACTIVO
    # key es vital para que al cambiar de paso se resetee el componente
    coords = streamlit_image_coordinates(
        image_path, 
        key=f"img_step_{task['task_id']}_{current_step_idx}"
    )
    
    # 4. LÓGICA AL HACER CLIC
    if coords:
        region_hit = check_click(coords, step_data.get("regions", []))
        
        if region_hit:
            if region_hit.get("is_correct"):
                # ACIERTO
                st.toast(region_hit.get("feedback_ok", "¡Correcto!"), icon="✅")
                
                # Solo cuenta como punto si acertó a la primera en este paso
                if state["total_attempts_in_step"] == 0:
                    state["correct_clicks"] += 1
                    state["history_log"].append(f"Paso {current_step_idx}: Acierto")
                else:
                    state["history_log"].append(f"Paso {current_step_idx}: Acierto (con reintentos)")
                
                # Avanzamos al siguiente paso
                state["current_step_index"] += 1
                state["total_attempts_in_step"] = 0 # Reiniciar contador para el nuevo paso
                st.rerun()
                
            else:
                # CLIC EN ZONA DEFINIDA PERO INCORRECTA (Trampas)
                st.toast("Esa no es la zona correcta. Inténtalo de nuevo.", icon="❌")
                state["total_attempts_in_step"] += 1
                
        else:
            # CLIC EN LA NADA
            st.toast("Has clicado fuera de las zonas activas.", icon="⚠️")
            state["total_attempts_in_step"] += 1
# --- INICIALIZACIÓN ---

if "graph_initialized" not in st.session_state:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    app.invoke({"student_id": "alumno_streamlit"}, config=config)
    
    state_values = app.get_state(config).values
    first_task = state_values.get("current_task", {})
    first_prompt = state_values.get("current_task_prompt", "Error cargando tarea.")
    
    initial_msg = {
        "role": "assistant",
        "content": first_prompt,
        "context_text": first_task.get("task_text"),
        "image_paths": normalize_images(first_task.get("task_image"))
    }
    
    st.session_state.messages.append(initial_msg)
    st.session_state.graph_initialized = True

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧠 Estado del Agente")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    current_state = app.get_state(config).values
    
    if current_state:
        st.subheader("📊 Niveles")
        st.json(current_state.get("competency_levels", {}))
        
        st.subheader("📝 Tarea Actual")
        task = current_state.get("current_task", {})
        if task:
            st.write(f"**Tipo:** {task.get('task_type', 'N/A')}")
        
        if st.button("Reiniciar Sesión"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# --- INTERFAZ PRINCIPAL ---

# 1. Dibujar historial
# Calculamos el índice del último mensaje para saber a cuál ocultarle las opciones
last_idx = len(st.session_state.messages) - 1

for i, msg in enumerate(st.session_state.messages):
    # Si es el último mensaje y es del asistente, le pasamos is_last_message=True
    is_last = (i == last_idx) and (msg["role"] == "assistant")
    render_message(msg, is_last_message=is_last)

# 2. Lógica de Input
last_agent_content = ""
for m in reversed(st.session_state.messages):
    if m["role"] == "assistant":
        last_agent_content = m["content"]
        break

# Extraemos opciones para generar los botones
# Usamos la misma función, pero aquí solo nos interesan las opciones
_, detected_options = split_text_and_options(last_agent_content)

current_task = current_state.get("current_task") if current_state else None

if current_task:
    task_type = current_task.get("task_type", "normal") # Leemos el tipo
    
    if task_type == "interactivo":
        # --- MODO INTERACTIVO ---
        handle_interactive_task(current_task)
    elif detected_options:
        # --- MODO SELECCIÓN ---
        st.write("---")
        st.subheader("Selecciona tu respuesta:")
        
        with st.form(key="quiz_form"):
            # Aquí aparecen los botones (las opciones)
            # Como render_message ocultó el texto de opciones arriba, no hay duplicidad
            selected_option = st.radio("Opciones:", detected_options, label_visibility="collapsed")
            submit_button = st.form_submit_button("Enviar Respuesta")
            
        if submit_button:
            # Enviamos solo la letra (ej: "a")
            answer_letter = selected_option.split(" ")[0].replace(")", "").replace(".", "")
            handle_user_input(answer_letter)
            
    else:
        # --- MODO TEXTO ---
        if user_input := st.chat_input("Escribe tu respuesta aquí..."):
            handle_user_input(user_input)
else:
    st.success("Sesión finalizada. Gracias por participar.")