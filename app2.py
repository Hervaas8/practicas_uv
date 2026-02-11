import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid, os, re
from agente2 import app

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor IA - Ciencias", layout="wide")

# Inicialización de estados persistentes
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.current_step = "working" 
    st.session_state.last_feedback = None
    st.session_state.display_task = None  # Tarea que se muestra en pantalla
    st.session_state.display_prompt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

def get_val(obj, key):
    """Acceso seguro a Pydantic o Dict"""
    return getattr(obj, key) if hasattr(obj, key) else obj.get(key)

# Obtener estado del grafo
state_values = app.get_state(config).values
if not state_values:
    app.invoke({"student_id": "alumno_demo"}, config=config)
    state_values = app.get_state(config).values

# Sincronización inicial: Si no hay nada en pantalla, cargamos la tarea actual del grafo
if st.session_state.display_task is None:
    st.session_state.display_task = get_val(state_values, "current_task")
    st.session_state.display_prompt = get_val(state_values, "current_task_prompt")

# --- FUNCIONES DE NAVEGACIÓN ---
def submit_answer(user_input):
    with st.spinner("Evaluando..."):
        app.update_state(config, {"current_student_response": user_input})
        for event in app.stream(None, config=config): pass
        
        # Guardamos el feedback pero NO actualizamos display_task todavía
        new_state = app.get_state(config).values
        st.session_state.last_feedback = get_val(new_state, "last_feedback_message")
        st.session_state.current_step = "reviewed"
    st.rerun()

def handle_next():
    """Aquí es donde realmente pasamos a la siguiente tarea visualmente"""
    new_state = app.get_state(config).values
    st.session_state.display_task = get_val(new_state, "current_task")
    st.session_state.display_prompt = get_val(new_state, "current_task_prompt")
    st.session_state.current_step = "working"
    st.session_state.last_feedback = None
    st.rerun()

# --- INTERFAZ ---
task = st.session_state.display_task

if not task:
    st.success("¡Sesión finalizada con éxito!")
else:
    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        st.markdown(f"### {st.session_state.display_prompt}")
        st.divider()

        if st.session_state.current_step == "working":
            # Normalizamos el tipo de tarea para evitar errores por espacios o mayúsculas
            t_type = str(task.get("task_type", "")).strip().lower()

            # --- A) TAREAS DE OPCIONES ---
            options = re.findall(r"([a-d][\)\.])\s*([^\n]+)", st.session_state.display_prompt)
            if t_type == "objetivo" and options:
                selection = st.radio("Elige tu respuesta:", [f"{o[0]} {o[1]}" for o in options])
                if st.button("Enviar Respuesta"):
                    submit_answer(selection[0])

            # --- B) TAREAS INTERACTIVAS ---
            elif t_type == "interactivo":
                st.info("🎯 Haz clic directamente en la imagen de la derecha para responder.")

            # --- C) TAREA: VENTAJAS Y DESVENTAJAS (v_y_d) ---
            elif t_type == "v_y_d":
                st.info("Completa la tabla con al menos dos ideas por casilla:")
                with st.form("form_vyd_estricto"):
                    # Estructura visual de tabla
                    r1c1, r1c2 = st.columns(2)
                    v1 = r1c1.text_area("Ventajas (Texto 1)", key="v1_input")
                    d1 = r1c2.text_area("Desventajas (Texto 1)", key="d1_input")
                    
                    st.divider()
                    
                    r2c1, r2c2 = st.columns(2)
                    v2 = r2c1.text_area("Ventajas (Texto 2)", key="v2_input")
                    d2 = r2c2.text_area("Desventajas (Texto 2)", key="d2_input")
                    
                    if st.form_submit_button("Enviar Análisis"):
                        # Concatenamos la respuesta para el motor de evaluación
                        resultado_tabla = (
                            f"TEXTO 1 -> Ventajas: {v1} | Desventajas: {d1}\n"
                            f"TEXTO 2 -> Ventajas: {v2} | Desventajas: {d2}"
                        )
                        submit_answer(resultado_tabla)

            # --- D) TAREA: CONECTAR ---
            elif t_type == "conectar":
                user_map = {}
                for concept in task.get("concepts", []):
                    user_map[concept] = st.multiselect(
                        f"Para: **{concept}**", 
                        task.get("definitions", []),
                        key=f"input_{concept}"
                    )
                if st.button("Validar Conexiones"):
                    submit_answer(user_map)
            
            # --- E) RESTO (Texto libre) ---
            else:
                resp = st.text_area("Escribe tu respuesta:", height=150)
                if st.button("Enviar"):
                    submit_answer(resp)
        else:
            # PANTALLA DE REVISIÓN (FEEDBACK)
            st.markdown(f"**Feedback:**\n{st.session_state.last_feedback}")
            if st.button("Siguiente Tarea ➔"):
                handle_next()

    with col_der:
        # 1. PREPARACIÓN DINÁMICA DE PESTAÑAS
        img_list = task.get("task_image", [])
        if isinstance(img_list, str): img_list = [img_list]
        
        text_context = task.get("task_text")
        
        # Validamos si el texto existe y no es una cadena vacía o "None"
        tab_list = []
        has_text = text_context and str(text_context).strip() not in ["None", ""]
        
        if has_text:
            tab_list.append("📄 Info")
        
        for i in range(len(img_list)):
            tab_list.append(f"🖼️ Imagen {i+1}")

        # 2. RENDERIZADO
        if tab_list:
            tabs = st.tabs(tab_list)
            idx = 0
            
            if has_text:
                with tabs[idx]:
                    st.write(text_context)
                idx += 1

            for i, img_path in enumerate(img_list):
                with tabs[idx]:
                    if os.path.exists(img_path):
                        if task.get("task_type") == "interactivo" and st.session_state.current_step == "working":
                            coords = streamlit_image_coordinates(img_path, key=f"click_{task['task_id']}")
                            if coords:
                                # Enviamos el prefijo CLICK seguido de las coordenadas reales
                                 submit_answer(f"CLICK: {coords['x']}, {coords['y']}")
                        else:
                            # CAMBIO CLAVE: usamos use_column_width para compatibilidad con tu versión
                            st.image(img_path, use_column_width=True)
                    else:
                        st.error(f"Archivo no encontrado: {img_path}")
                idx += 1