import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid, os, re
from agente2 import app, load_test_inicial, load_curriculum

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor IA - Ciencias", layout="wide")

# --- CSS HACK PARA CONTROL TOTAL DE IMÁGENES ---
st.markdown(
    """
    <style>
        /* Centramos el contenedor de la imagen */
        div[data-testid="stImage"] {
            display: flex;
            justify_content: center;
            align-items: center;
            width: 100%;
        }

        /* Reglas para la imagen en sí */
        div[data-testid="stImage"] img {
            /* 1. LÍMITE HORIZONTAL: Nunca más ancha que el contenedor */
            max-width: 100% !important;
            
            /* 2. LÍMITE VERTICAL: Nunca más alta que 550px */
            max-height: 550px !important;
            
            /* 3. ADAPTACIÓN: Calcula las dimensiones automáticamente respetando los límites */
            width: auto !important;
            height: auto !important;
            
            /* 4. VISUALIZACIÓN: Asegura que se vea completa (contain) */
            object-fit: contain !important;
        }

        /* Estilo para las métricas del sidebar */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicialización de estados
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.current_step = "setup" 
    st.session_state.last_feedback = None
    st.session_state.display_task = None
    st.session_state.display_prompt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

def get_val(obj, key):
    return getattr(obj, key) if hasattr(obj, key) else obj.get(key)

# --- FUNCIONES DE NAVEGACIÓN ---
def start_agent(do_test):
    """Inicializa el agente con o sin test"""
    if do_test:
        initial_tasks = load_test_inicial()
        is_test_mode = True
        default_levels = {} 
    else:
        initial_tasks = load_curriculum()
        is_test_mode = False
        default_levels = {
            "navegar": "nivel_1", "integrar": "nivel_1", 
            "evaluar": "nivel_1", "contenido": "nivel_1"
        }

    # Invocamos pasando también total_tasks
    app.invoke({
        "student_id": "alumno_demo",
        "remaining_tasks": initial_tasks,
        "total_tasks": len(initial_tasks), # Inicializamos el contador
        "test_mode": is_test_mode,
        "competency_levels": default_levels,
        "current_subject": "ciencias",
        "current_session": "sesion1"
    }, config=config)
    
    new_state = app.get_state(config).values
    st.session_state.display_task = get_val(new_state, "current_task")
    st.session_state.display_prompt = get_val(new_state, "current_task_prompt")
    st.session_state.current_step = "working"
    st.rerun()

def submit_answer(user_input):
    with st.spinner("Evaluando..."):
        app.update_state(config, {"current_student_response": user_input})
        for event in app.stream(None, config=config): pass
        
        new_state = app.get_state(config).values
        st.session_state.last_feedback = get_val(new_state, "last_feedback_message")
        st.session_state.current_step = "reviewed"
    st.rerun()

def handle_next():
    new_state = app.get_state(config).values
    st.session_state.display_task = get_val(new_state, "current_task")
    st.session_state.display_prompt = get_val(new_state, "current_task_prompt")
    st.session_state.current_step = "working"
    st.session_state.last_feedback = None
    st.rerun()

def render_competency_sidebar(levels, is_test):
    """Muestra los niveles en la barra lateral con colores."""
    st.sidebar.title("📊 Tu Progreso")
    
    if is_test:
        st.sidebar.info("Evaluando nivel inicial...")
    else:
        st.sidebar.markdown("### Niveles de Competencia")
        
        # Diccionario de colores para los niveles
        colors = {
            "nivel_1": "🔴 Nivel 1 (Básico)",
            "nivel_2": "🟠 Nivel 2 (Intermedio)",
            "nivel_3": "🟢 Nivel 3 (Avanzado)"
        }
        
        # Mapeo de nombres más amigables para las estrategias
        names = {
            "navegar": "Navegación",
            "integrar": "Integración",
            "evaluar": "Evaluación",
            "contenido": "Contenido"
        }
        
        for strategy, level_key in levels.items():
            if strategy in names:
                clean_name = names[strategy]
                level_text = colors.get(level_key, level_key)
                st.sidebar.metric(label=clean_name, value=level_text.split(" ")[1] + " " + level_text.split(" ")[2])
                st.sidebar.divider()

# --- INTERFAZ ---

if st.session_state.current_step == "setup":
    st.title("🎓 Tutor IA - Ciencias")
    st.markdown("Bienvenido. Antes de empezar, nos gustaría conocerte mejor.")
    st.info("¿Deseas realizar un test inicial para adaptar el nivel de las tareas?")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Sí, realizar test inicial", use_container_width=True):
            start_agent(do_test=True)
    with c2:
        if st.button("❌ No, ir directamente a la sesión", use_container_width=True):
            start_agent(do_test=False)

else:
    # Obtenemos estado actual
    state_vals = app.get_state(config).values
    is_test_mode = get_val(state_vals, "test_mode")
    competencies = get_val(state_vals, "competency_levels")
    
    # 1. RENDERIZAR SIDEBAR CON NIVELES
    render_competency_sidebar(competencies, is_test_mode)
    
    task = st.session_state.display_task

    if not task:
        st.success("¡Sesión finalizada con éxito!")
        # Barra de progreso al 100%
        st.progress(1.0)
        if st.button("Reiniciar"):
            st.session_state.clear()
            st.rerun()
    else:
        if is_test_mode:
            st.warning("📝 MODO TEST INICIAL - Responde lo mejor que puedas")
        
        col_izq, col_der = st.columns([1, 1], gap="large")

        with col_izq:
            full_prompt = st.session_state.display_prompt
            t_type = str(task.get("task_type", "")).strip().lower()
            
            # --- CAMBIO IMPORTANTE: Regex más estricta ---
            # Antes: r"([a-d][\)\.])\s*([^\n]+)"  <- Fallaba con "pregunta."
            # Ahora: r"(?:^|\n)\s*([a-d][\)\.])\s*([^\n]+)" <- Exige que empiece la línea
            options_pattern = r"(?:^|\n)\s*([a-d][\)\.])\s*([^\n]+)"
            found_options = re.findall(options_pattern, full_prompt)

            # --- RENDERIZADO DE TAREAS ---
            if st.session_state.current_step == "working" and t_type == "objetivo" and found_options:
                split_match = re.search(r"(?i)(\n\s*Opciones:?)|(\n\s*[-•]?\s*[a-d][\)\.])", full_prompt)
                clean_text = full_prompt[:split_match.start()].strip() if split_match else full_prompt
                
                st.markdown(f"### {clean_text}")
                st.divider()
                selection = st.radio("Elige tu respuesta:", [f"{o[0]} {o[1]}" for o in found_options])
                if st.button("Enviar Respuesta"): submit_answer(selection[0])
            
            elif st.session_state.current_step == "working" and t_type == "objetivo" and not found_options:
                st.markdown(f"### {full_prompt}")
                st.divider()
                selection = st.radio("Elige una opción:", ["a", "b", "c", "d"])
                if st.button("Enviar Respuesta"): submit_answer(selection)

            else:
                st.markdown(f"### {full_prompt}")
                st.divider()
                if st.session_state.current_step == "working":
                    if t_type == "interactivo":
                        st.info("🎯 Haz clic directamente en la imagen de la derecha para responder.")
                    elif t_type == "v_y_d":
                        st.info("Completa la tabla:")
                        with st.form("form_vyd"):
                            r1c1, r1c2 = st.columns(2)
                            v1 = r1c1.text_area("Ventajas T1")
                            d1 = r1c2.text_area("Desventajas T1")
                            r2c1, r2c2 = st.columns(2)
                            v2 = r2c1.text_area("Ventajas T2")
                            d2 = r2c2.text_area("Desventajas T2")
                            if st.form_submit_button("Enviar"):
                                submit_answer(f"T1 V:{v1} D:{d1} | T2 V:{v2} D:{d2}")
                    elif t_type == "conectar":
                        user_map = {}
                        for concept in task.get("concepts", []):
                            user_map[concept] = st.multiselect(f"Para: **{concept}**", task.get("definitions", []), key=f"k_{concept}")
                        if st.button("Validar"): submit_answer(user_map)
                    else:
                        resp = st.text_area("Escribe tu respuesta:", height=150)
                        if st.button("Enviar"): submit_answer(resp)
            
            if st.session_state.current_step != "working":
                st.markdown(f"**Feedback:**\n{st.session_state.last_feedback}")
                if st.button("Siguiente Tarea ➔"): handle_next()

        with col_der:
            img_list = task.get("task_image", [])
            if isinstance(img_list, str): img_list = [img_list]
            text_context = task.get("task_text")
            
            tab_list = []
            has_text = text_context and str(text_context).strip() not in ["None", ""]
            if has_text: tab_list.append("📄 Info")
            for i in range(len(img_list)): tab_list.append(f"🖼️ Img {i+1}")

            if tab_list:
                tabs = st.tabs(tab_list)
                idx = 0
                if has_text:
                    with tabs[idx]: st.write(text_context)
                    idx += 1
                for i, img_path in enumerate(img_list):
                    with tabs[idx]:
                        if os.path.exists(img_path):
                            if task.get("task_type") == "interactivo" and st.session_state.current_step == "working":
                                coords = streamlit_image_coordinates(img_path, key=f"click_{task['task_id']}")
                                if coords: submit_answer(f"CLICK: {coords['x']}, {coords['y']}")
                            else:
                                st.image(img_path, use_column_width=True)
                        else:
                            st.warning(f"Imagen no encontrada: {img_path}")
                    idx += 1
        
        # --- BARRA DE PROGRESO (ABAJO) ---
        st.divider()
        total = get_val(state_vals, "total_tasks") or 1
        remaining = len(get_val(state_vals, "remaining_tasks"))
        
        # Cálculo del progreso
        completed = total - remaining
        progress_val = max(0.0, min(1.0, completed / total))
        
        st.progress(progress_val, text=f"Progreso de la sesión: {int(progress_val*100)}%")