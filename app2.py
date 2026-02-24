import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid, os, re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agente2 import app, load_test_inicial, load_curriculum

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor IA - Ciencias", layout="wide")

# --- CSS HACK PARA CONTROL TOTAL DE IMÁGENES ---
st.markdown(
    """
    <style>
        div[data-testid="stImage"] {
            display: flex;
            justify_content: center;
            align-items: center;
            width: 100%;
        }
        div[data-testid="stImage"] img {
            max-width: 100% !important;
            max-height: 550px !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
            "navegar": "nivel_1", 
            "integrar": "nivel_1", 
            "evaluar": "nivel_1"
        }

    app.invoke({
        "student_id": "alumno_demo",
        "remaining_tasks": initial_tasks,
        "total_tasks": len(initial_tasks),
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
        
        colors = {
            "nivel_1": "🔴 Nivel 1 (Básico)",
            "nivel_2": "🟠 Nivel 2 (Intermedio)",
            "nivel_3": "🟢 Nivel 3 (Avanzado)"
        }
        
        names = {
            "navegar": "Navegación",
            "integrar": "Integración",
            "evaluar": "Evaluación"
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
    state_vals = app.get_state(config).values
    is_test_mode = get_val(state_vals, "test_mode")
    competencies = get_val(state_vals, "competency_levels") or {}
    
    render_competency_sidebar(competencies, is_test_mode)
    
    task = st.session_state.display_task

    if not task:
        # --- DASHBOARD FINAL DE SESIÓN ---
        st.balloons()
        st.title("🏆 ¡Sesión finalizada con éxito!")
        st.markdown("Aquí tienes el resumen de tu rendimiento en esta sesión. ¡Buen trabajo!")
        st.progress(1.0)
        
        # 1. EXTRACCIÓN DE DATOS GLOBALES
        history = get_val(state_vals, "task_history") or []
        final_levels = competencies
        
        # 2. CÁLCULO DE LA RACHA MÁXIMA
        max_streak = 0
        current_streak = 0
        
        for h in history:
            score = h.get('score', 0) if isinstance(h, dict) else getattr(h, 'score', 0)
            if score >= 0.5: 
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0
                
        if max_streak > 1:
            st.success(f"🔥 **¡Racha Imparable!** Durante esta sesión llegaste a acertar **{max_streak}** tareas seguidas.")
        elif max_streak == 1:
            st.info("Has completado las tareas, ¡la próxima vez seguro que logras encadenar varios aciertos seguidos!")
        elif history:
            st.info("¡Gracias por tu esfuerzo! Sigue practicando para mejorar tus resultados.")

        # 3. ESTRUCTURA DE PESTAÑAS
        tab_estrategias, tab_opinion = st.tabs(["🧠 Mis Habilidades (Estrategias)", "🗣️ Mi Opinión"])
        
        # --- PESTAÑA: ESTRATEGIAS ---
        with tab_estrategias:
            st.markdown("### 🎯 Tus Niveles Finales por Competencia")
            
            strat_names = {"navegar": "Navegación", "integrar": "Integración", "evaluar": "Evaluación"}
            strat_colors = {"nivel_1": "🔴 Básico", "nivel_2": "🟠 Intermedio", "nivel_3": "🟢 Avanzado"}
            
            cols = st.columns(3)
            for i, (strat_key, strat_name) in enumerate(strat_names.items()):
                lvl = final_levels.get(strat_key, "nivel_1")
                with cols[i]:
                    st.metric(label=strat_name, value=strat_colors.get(lvl, lvl))
                    
            st.divider()
            
            # GRÁFICO DE BARRAS Y CONSIDERACIONES
            st.markdown("### 📊 Tu Perfil Competencial (Puntuación Media)")
            col_barras, col_text = st.columns([1, 1], gap="large")
            
            strat_scores = {"navegar": [], "integrar": [], "evaluar": []}
            for h in history:
                is_dict = isinstance(h, dict)
                strat = h.get('strategy', '') if is_dict else getattr(h, 'strategy', '')
                score = h.get('score', 0) if is_dict else getattr(h, 'score', 0)
                
                strat = strat.replace("navegacion", "navegar").replace("integracion", "integrar").replace("evaluacion", "evaluar")
                if strat in strat_scores:
                    strat_scores[strat].append(score)
            
            avg_scores = {}
            for k, v in strat_scores.items():
                avg_scores[k] = sum(v)/len(v) if v else 0.0
                
            with col_barras:
                df_barras = pd.DataFrame({
                    "Estrategia": ["Navegación", "Integración", "Evaluación"],
                    "Puntuación": [avg_scores["navegar"], avg_scores["integrar"], avg_scores["evaluar"]]
                })
                
                fig_barras = px.bar(
                    df_barras, 
                    x="Estrategia", 
                    y="Puntuación", 
                    color="Estrategia",
                    range_y=[0, 1],
                    color_discrete_sequence=["#FF5722", "#2196F3", "#4CAF50"]
                )
                
                fig_barras.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300
                )
                
                # Añadir el texto del valor encima de la barra
                fig_barras.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                
                st.plotly_chart(fig_barras, use_container_width=True)
                
            with col_text:
                st.markdown("#### Consideraciones:")
                for strat_key, strat_name in strat_names.items():
                    lvl = final_levels.get(strat_key, "nivel_1")
                    if lvl == "nivel_3":
                        msg = f"**{strat_name}:** Has **adquirido** una gran soltura y eficacia. ¡Excelente dominio!"
                        st.success(msg)
                    elif lvl == "nivel_2":
                        msg = f"**{strat_name}:** Estás **en proceso de adquirir** las herramientas necesarias. Vas por muy buen camino."
                        st.info(msg)
                    else:
                        msg = f"**{strat_name}:** Aún **estás desarrollando** tus habilidades base. Con un poco más de práctica lo dominarás."
                        st.warning(msg)
                        
            st.divider()
            
            # EVOLUCIÓN HISTÓRICA POR NIVELES
            st.markdown("### 📈 Evolución de tus Niveles")
            st.write("Observa cómo han cambiado tus niveles competenciales a lo largo de las tareas:")
            
            chart_data = []
            
            try:
                # Recuperamos el historial de estados de LangGraph y lo analizamos
                state_snapshots = list(app.get_state_history(config))
                state_snapshots.reverse() # De más antiguo a más reciente
                
                levels_by_task = {}
                
                for snap in state_snapshots:
                    state_dict = snap.values
                    if not state_dict: continue
                        
                    th = state_dict.get("task_history", [])
                    
                    # Contamos cuántas tareas principales se han hecho en este punto
                    main_tasks_completed = sum(1 for x in th if not (x.get('is_test_task', False) if isinstance(x, dict) else getattr(x, 'is_test_task', False)))
                    
                    # Al iterar de antiguo a nuevo, nos quedamos siempre con el estado más reciente de cada momento
                    levels_by_task[main_tasks_completed] = state_dict.get("competency_levels", {})
                    
                # Construimos la tabla final
                for task_num in sorted(levels_by_task.keys()):
                    step_name = "Inicio" if task_num == 0 else f"T{task_num}"
                    levels = levels_by_task[task_num]
                    
                    for s_key, s_name in strat_names.items():
                        lvl_str = levels.get(s_key, "nivel_1")
                        lvl_num = int(lvl_str.split("_")[1]) if isinstance(lvl_str, str) and "_" in lvl_str else 1
                        
                        chart_data.append({
                            "Tarea": step_name,
                            "Estrategia": s_name,
                            "Nivel": lvl_num
                        })
                        
            except Exception as e:
                st.warning(f"No se pudo cargar el historial de niveles: {e}")
                
            if chart_data:
                df_chart = pd.DataFrame(chart_data)
                
                fig_line = px.line(
                    df_chart, 
                    x="Tarea", 
                    y="Nivel", 
                    color="Estrategia", 
                    markers=True,
                    line_shape="hv", # Esto hace que la línea sea "escalonada"
                    color_discrete_sequence=["#FF5722", "#2196F3", "#4CAF50"]
                )
                
                fig_line.update_layout(
                    height=380, 
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[1, 2, 3],
                        ticktext=['Nivel 1<br>(Básico)', 'Nivel 2<br>(Intermedio)', 'Nivel 3<br>(Avanzado)'],
                        range=[0.8, 3.2] # Añade un poco de margen para que no se corte por arriba/abajo
                    )
                )
                
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No hay suficientes datos para mostrar la gráfica de evolución.")

        # --- PESTAÑA: OPINIÓN DEL ALUMNO ---
        with tab_opinion:
            st.markdown("### ¿Qué te ha parecido la sesión de hoy?")
            st.write("Tu opinión nos ayuda a mejorar y a ajustar el nivel de los próximos retos.")
            
            student_rating = st.feedback("faces")
            
            if student_rating is not None:
                if student_rating >= 3:
                    st.success("¡Nos alegra mucho que te haya gustado!")
                elif student_rating == 2:
                    st.info("¡Gracias por tu valoración! Seguiremos mejorando.")
                else:
                    st.warning("Sentimos que haya sido dura. ¡Revisaremos el nivel para la próxima!")

        st.divider()
        if st.button("↻ Terminar y Salir"):
            st.session_state.clear()
            st.rerun()
            
    else:
        # --- LÓGICA DE TAREAS ---
        if is_test_mode:
            st.warning("📝 MODO TEST INICIAL - Responde lo mejor que puedas")
        
        col_izq, col_der = st.columns([1, 1], gap="large")

        with col_izq:
            full_prompt = st.session_state.display_prompt
            t_type = str(task.get("task_type", "")).strip().lower()
            
            options_pattern = r"(?:^|\n)\s*([a-d][\)\.])\s*([^\n]+)"
            found_options = re.findall(options_pattern, full_prompt)

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
        
        st.divider()
        total = get_val(state_vals, "total_tasks") or 1
        remaining = len(get_val(state_vals, "remaining_tasks"))
        
        completed = total - remaining
        progress_val = max(0.0, min(1.0, completed / total))
        
        st.progress(progress_val, text=f"Progreso de la sesión: {int(progress_val*100)}%")