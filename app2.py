import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid, os, re, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- IMPORTACIONES DEL AGENTE Y BASE DE DATOS ---
from agente2 import app, load_test_inicial, load_curriculum
from database import authenticate_user, save_student_results, get_professor_students_data

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor IA - Ciencias", layout="wide")

st.markdown(
    """
    <style>
        div[data-testid="stImage"] { display: flex; justify_content: center; align_items: center; width: 100%; }
        div[data-testid="stImage"] img { max-width: 100% !important; max-height: 550px !important; width: auto !important; height: auto !important; object-fit: contain !important; }
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.session_state.user_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.current_step = "setup" 
    st.session_state.last_feedback = None
    st.session_state.display_task = None
    st.session_state.display_prompt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

def get_val(obj, key):
    return getattr(obj, key) if hasattr(obj, key) else obj.get(key)

def get_available_sessions():
    try:
        with open("tareas2.json", "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            return list(data.get("ciencias", {}).keys())
    except Exception:
        return ["sesion1"]

def start_agent(do_test, session_id):
    if do_test:
        initial_tasks = load_test_inicial()
        is_test_mode = True
        default_levels = {} 
    else:
        initial_tasks = load_curriculum(session_id)
        is_test_mode = False
        default_levels = {"navegar": "nivel_1", "integrar": "nivel_1", "evaluar": "nivel_1"}

    app.invoke({
        "student_id": st.session_state.user_name,
        "remaining_tasks": initial_tasks,
        "total_tasks": len(initial_tasks),
        "test_mode": is_test_mode,
        "competency_levels": default_levels,
        "current_subject": "ciencias",
        "current_session": session_id
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
    st.sidebar.title(f"👤 {st.session_state.user_name}")
    st.sidebar.divider()
    st.sidebar.title("📊 Tu Progreso")
    
    if is_test:
        st.sidebar.info("Evaluando nivel inicial...")
    else:
        st.sidebar.markdown("### Niveles de Competencia")
        colors = {"nivel_1": "🔴 Nivel 1 (Básico)", "nivel_2": "🟠 Nivel 2 (Intermedio)", "nivel_3": "🟢 Nivel 3 (Avanzado)"}
        names = {"navegar": "Navegación", "integrar": "Integración", "evaluar": "Evaluación"}
        for strategy, level_key in levels.items():
            if strategy in names:
                clean_name = names[strategy]
                level_text = colors.get(level_key, level_key)
                st.sidebar.metric(label=clean_name, value=level_text.split(" ")[1] + " " + level_text.split(" ")[2])
                st.sidebar.divider()
    
    if st.sidebar.button("🚪 Cerrar Sesión"):
        st.session_state.clear()
        st.rerun()

# --- FUNCIÓN REUTILIZABLE PARA EL DASHBOARD INDIVIDUAL (ALUMNO Y PROFESOR) ---
def render_student_dashboard(history, final_levels, current_session, enable_evolution_chart=False):
    """Pinta las pestañas de Estrategias y Contenidos de un alumno."""
    
    # Cálculo de racha
    max_streak = 0
    current_streak = 0
    for h in history:
        score = h.get('score', 0) if isinstance(h, dict) else getattr(h, 'score', 0)
        if score >= 0.5: 
            current_streak += 1
            if current_streak > max_streak: max_streak = current_streak
        else:
            current_streak = 0
            
    if max_streak > 1:
        st.success(f"🔥 **¡Racha Imparable!** En esta sesión se lograron acertar **{max_streak}** tareas seguidas.")
    elif max_streak == 1:
        st.info("Tareas completadas, ¡hay margen para intentar encadenar aciertos seguidos en la próxima!")

    tab_estrategias, tab_contenidos = st.tabs(["🧠 Habilidades (Estrategias)", "📚 Conocimientos (Contenidos)"])
    
    # --- PESTAÑA: ESTRATEGIAS ---
    with tab_estrategias:
        st.markdown("### 🎯 Niveles Finales por Competencia")
        strat_names = {"navegar": "Navegación", "integrar": "Integración", "evaluar": "Evaluación"}
        strat_colors = {"nivel_1": "🔴 Básico", "nivel_2": "🟠 Intermedio", "nivel_3": "🟢 Avanzado"}
        
        cols = st.columns(3)
        for i, (strat_key, strat_name) in enumerate(strat_names.items()):
            lvl = final_levels.get(strat_key, "nivel_1")
            with cols[i]:
                st.metric(label=strat_name, value=strat_colors.get(lvl, lvl))
                
        st.divider()
        st.markdown("### 📊 Perfil Competencial (Puntuación Media)")
        col_barras, col_text = st.columns([1, 1], gap="large")
        
        strat_scores = {"navegar": [], "integrar": [], "evaluar": []}
        for h in history:
            is_dict = isinstance(h, dict)
            strat = h.get('strategy', '') if is_dict else getattr(h, 'strategy', '')
            score = h.get('score', 0) if is_dict else getattr(h, 'score', 0)
            strat = strat.replace("navegacion", "navegar").replace("integracion", "integrar").replace("evaluacion", "evaluar")
            if strat in strat_scores:
                strat_scores[strat].append(score)
        
        avg_scores = {k: sum(v)/len(v) if v else 0.0 for k, v in strat_scores.items()}
            
        with col_barras:
            df_barras = pd.DataFrame({
                "Estrategia": ["Navegación", "Integración", "Evaluación"],
                "Puntuación": [avg_scores["navegar"], avg_scores["integrar"], avg_scores["evaluar"]]
            })
            fig_barras = px.bar(
                df_barras, x="Estrategia", y="Puntuación", color="Estrategia",
                range_y=[0, 1], color_discrete_sequence=["#FF5722", "#2196F3", "#4CAF50"]
            )
            fig_barras.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=300)
            fig_barras.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            st.plotly_chart(fig_barras, use_container_width=True)
            
        with col_text:
            st.markdown("#### Consideraciones:")
            for strat_key, strat_name in strat_names.items():
                lvl = final_levels.get(strat_key, "nivel_1")
                if lvl == "nivel_3":
                    st.success(f"**{strat_name}:** Gran soltura y eficacia. ¡Excelente dominio!")
                elif lvl == "nivel_2":
                    st.info(f"**{strat_name}:** En proceso de adquirir las herramientas necesarias. Va por buen camino.")
                else:
                    st.warning(f"**{strat_name}:** Aún desarrollando habilidades base. Requiere más práctica.")
                    
        # --- EVOLUCIÓN DESDE LA BASE DE DATOS ---
        st.divider()
        st.markdown("### 📈 Evolución de Niveles")
        chart_data = []
        task_counter = 0
        
        for h in history:
            is_dict = isinstance(h, dict)
            is_test = h.get('is_test_task', False) if is_dict else getattr(h, 'is_test_task', False)
            
            if not is_test:
                task_counter += 1
                step_name = f"T{task_counter}"
                levels_snapshot = h.get('current_levels', {}) if is_dict else getattr(h, 'current_levels', {})
                
                if levels_snapshot:
                    for s_key, s_name in strat_names.items():
                        lvl_str = levels_snapshot.get(s_key, "nivel_1")
                        lvl_num = int(lvl_str.split("_")[1]) if isinstance(lvl_str, str) and "_" in lvl_str else 1
                        chart_data.append({"Tarea": step_name, "Estrategia": s_name, "Nivel": lvl_num})

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig_line = px.line(df_chart, x="Tarea", y="Nivel", color="Estrategia", markers=True, line_shape="hv", color_discrete_sequence=["#FF5722", "#2196F3", "#4CAF50"])
            fig_line.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Nivel 1<br>(Básico)', 'Nivel 2<br>(Intermedio)', 'Nivel 3<br>(Avanzado)'], range=[0.8, 3.2]))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No hay datos históricos de evolución disponibles (puede ser una sesión antigua guardada antes de esta actualización).")

    # --- PESTAÑA: CONTENIDOS ---
    with tab_contenidos:
        st.markdown("### 📊 Progreso en Contenidos")
        content_history = [h for h in history if (h.get('contenido', False) if isinstance(h, dict) else getattr(h, 'contenido', False))]
        
        if not content_history:
            st.info("En esta sesión no se han evaluado tareas de contenido específico.")
        else:
            scores = [h.get('score', 0) if isinstance(h, dict) else getattr(h, 'score', 0) for h in content_history]
            global_score = sum(scores) / len(scores) if scores else 0
            global_score_10 = global_score * 10 
            
            # ELIMINADO EL GRÁFICO TIPO VELOCÍMETRO, CAMBIADO POR UN MÉTRICO SIMPLE
            col_nota, col_feedback = st.columns([1, 2], gap="large")
            
            with col_nota:
                st.metric(label="Nota Global de Contenidos", value=f"{global_score_10:.1f} / 10")
                
            with col_feedback:
                st.markdown("#### Consideraciones Globales:")
                if global_score_10 >= 8: st.success("¡Excelente trabajo! Sólida base teórica.")
                elif global_score_10 >= 5: st.info("Va por buen camino. Conceptos generales claros.")
                else: st.warning("Se recomienda volver a repasar la teoría base.")
                    
            st.divider()
            st.markdown("### 🧩 Desglose por Conceptos")
            concept_scores = {}
            for h in content_history:
                c_list = h.get('conceptos', []) if isinstance(h, dict) else getattr(h, 'conceptos', [])
                score = h.get('score', 0) if isinstance(h, dict) else getattr(h, 'score', 0)
                for c in c_list:
                    if c not in concept_scores: concept_scores[c] = []
                    concept_scores[c].append(score)
            
            if concept_scores:
                domina, repasar, medio = [], [], []
                for c, s_list in concept_scores.items():
                    avg = sum(s_list) / len(s_list)
                    item_text = f"**{c}** *(Nota: {avg * 10:.1f}/10)*"
                    if avg >= 0.8: domina.append(f"🟢 {item_text}")
                    elif avg < 0.5: repasar.append(f"🔴 {item_text}")
                    else: medio.append(f"🟠 {item_text}")
                        
                col_domina, col_repasar = st.columns(2)
                with col_domina:
                    st.markdown("#### 🌟 Conceptos que domina")
                    if domina:
                        for item in domina: st.write(item)
                    else: st.write("Aún en desarrollo...")
                    if medio:
                        st.markdown("#### 🔄 En proceso")
                        for item in medio: st.write(item)
                with col_repasar:
                    st.markdown("#### 📚 Conceptos a repasar")
                    if repasar:
                        for item in repasar: st.write(item)
                    else: st.write("¡Ninguno! Todo va muy bien.")


# ==========================================
# --- ENRUTADOR PRINCIPAL ---
# ==========================================

# 1. PANTALLA DE LOGIN
if not st.session_state.logged_in:
    st.title("🔐 Acceso a Tutor IA")
    st.markdown("Por favor, inicia sesión para continuar.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit_btn = st.form_submit_button("Entrar", use_container_width=True)
            
            if submit_btn:
                user_data = authenticate_user(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_role = user_data['role']
                    st.session_state.user_name = user_data['name']
                    st.session_state.user_id = user_data['id']
                    st.success(f"¡Bienvenido, {user_data['name']}!")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
                    
        st.info("💡 **Cuentas de prueba:**\n- Profesor: `profe_ana` / `1234`\n- Alumnos: `juan`, `maria`, `luis` / `1234`")

# 2. PANTALLA DEL PROFESOR
elif st.session_state.user_role == 'profesor':
    st.title(f"👨‍🏫 Panel Docente - {st.session_state.user_name}")
    
    raw_data = get_professor_students_data(st.session_state.user_id)
    completed_data = [d for d in raw_data if d['task_history']]
    
    if not completed_data:
        st.info("Aún no hay datos. Espera a que tus alumnos completen alguna sesión para ver las estadísticas.")
        if st.button("🚪 Cerrar Sesión"):
            st.session_state.clear()
            st.rerun()
    else:
        # SELECTOR DE SESIÓN PARA EL PROFESOR
        available_sessions = sorted(list(set([d['session_id'] for d in completed_data])))
        selected_prof_session = st.selectbox("📂 Selecciona la sesión a analizar:", available_sessions)
        
        # Filtramos datos SOLO para la sesión seleccionada
        session_data = [d for d in completed_data if d['session_id'] == selected_prof_session]
        
        if not session_data:
            st.warning("No hay datos completados por tus alumnos para esta sesión.")
        else:
            # --- PROCESAMIENTO DE DATOS DE LA SESIÓN ---
            all_strat_scores = {"navegar": [], "integrar": [], "evaluar": []}
            all_concept_scores = {}
            student_global_scores = [] 
            
            for student in session_data:
                history = student['task_history']
                student_content_scores = []
                
                for task in history:
                    score = task.get('score', 0)
                    strat = task.get('strategy', '')
                    if strat in all_strat_scores:
                        all_strat_scores[strat].append(score)
                    
                    if task.get('contenido'):
                        student_content_scores.append(score)
                        for c in task.get('conceptos', []):
                            if c not in all_concept_scores:
                                all_concept_scores[c] = []
                            all_concept_scores[c].append(score)
                
                s_avg = sum(student_content_scores)/len(student_content_scores) if student_content_scores else 0
                student_global_scores.append({"nombre": student['alumno_name'], "nota": s_avg})
            
            avg_strats = {k: (sum(v)/len(v) if v else 0) for k, v in all_strat_scores.items()}
            avg_concepts = {k: (sum(v)/len(v) if v else 0) for k, v in all_concept_scores.items()}
            
            global_strat_score = sum(avg_strats.values()) / len(avg_strats) if avg_strats else 0
            global_content_score = sum(avg_concepts.values()) / len(avg_concepts) if avg_concepts else 0

            # --- TABS DEL DASHBOARD DOCENTE ---
            tab_general, tab_emparejamiento, tab_individual = st.tabs(["📊 Visión de la Clase", "🤝 Emparejamiento", "👤 Vista Individual de Alumno"])
            
            with tab_general:
                st.markdown(f"### Notas Globales de la Clase ({selected_prof_session})")
                c1, c2 = st.columns(2)
                c1.metric("Nota Global - Contenidos", f"{global_content_score * 10:.1f}")
                c2.metric("Nota Global - Estrategias", f"{global_strat_score * 10:.1f}")
                
                st.divider()
                st.markdown("### 🤖 Análisis e Implicaciones")
                weak_concepts = [c for c, v in avg_concepts.items() if v < 0.6]
                weak_strats = [s for s, v in avg_strats.items() if v < 0.6]
                
                analisis_texto = ""
                if global_content_score >= 0.6: analisis_texto += "✅ **La clase va bien en contenidos generales**"
                else: analisis_texto += "⚠️ **La clase necesita refuerzo en contenidos**"
                    
                if weak_concepts: analisis_texto += f", aunque es prioritario **repasar: {', '.join(weak_concepts)}**."
                else: analisis_texto += " y no se detectan lagunas teóricas graves."
                    
                analisis_texto += "\n\n"
                
                if global_strat_score >= 0.6: analisis_texto += "✅ **El rendimiento en estrategias es adecuado**"
                else: analisis_texto += "⚠️ **Hay dificultades en la aplicación de estrategias**"
                    
                if weak_strats:
                    nombres_estrat = [s.capitalize() for s in weak_strats]
                    analisis_texto += f", pero falta **repasar y practicar: {', '.join(nombres_estrat)}**."
                else: analisis_texto += " mostrando buen dominio competencial."
                    
                st.info(analisis_texto)
                
                st.markdown("### Desglose Detallado")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    df_strats = pd.DataFrame(list(avg_strats.items()), columns=['Estrategia', 'Media'])
                    df_strats['Media'] = df_strats['Media'] * 10
                    fig_s = px.bar(df_strats, x='Estrategia', y='Media', title="Media por Estrategia", range_y=[0, 10], color='Estrategia', color_discrete_sequence=["#2196F3"])
                    st.plotly_chart(fig_s, use_container_width=True)
                    
                with col_chart2:
                    if avg_concepts:
                        df_c = pd.DataFrame(list(avg_concepts.items()), columns=['Concepto', 'Media'])
                        df_c['Media'] = df_c['Media'] * 10
                        fig_c = px.bar(df_c, x='Media', y='Concepto', title="Media por Concepto Teórico", orientation='h', range_x=[0, 10], color='Media', color_continuous_scale='RdYlGn')
                        st.plotly_chart(fig_c, use_container_width=True)
                    else:
                        st.write("No hay conceptos evaluados.")

            with tab_emparejamiento:
                st.markdown("### 🤝 Sistema de Tutoría entre Iguales")
                st.write("El sistema sugiere parejas uniendo a alumnos con alto rendimiento (Mentores) con alumnos que necesitan apoyo (Aprendices).")
                
                sorted_students = sorted(student_global_scores, key=lambda x: x['nota'], reverse=True)
                if len(sorted_students) >= 2:
                    mitad = len(sorted_students) // 2
                    mentores = sorted_students[:mitad]
                    aprendices = sorted_students[-mitad:]
                    aprendices.reverse() 
                    
                    for i in range(mitad):
                        m, a = mentores[i], aprendices[i]
                        st.success(f"**Pareja {i+1}:** 🟢 {m['nombre']} (Nota: {m['nota']*10:.1f}) enseña a 🔴 {a['nombre']} (Nota: {a['nota']*10:.1f})")
                        
                    if len(sorted_students) % 2 != 0:
                        st.warning(f"🙋 Alumno sin pareja (Grupo impar): {sorted_students[mitad]['nombre']} (Nota: {sorted_students[mitad]['nota']*10:.1f})")
                else:
                    st.info("Necesitas al menos 2 alumnos que hayan terminado la sesión para emparejarlos.")

            with tab_individual:
                st.markdown("### 🔍 Análisis de Alumno")
                # Diccionario para que el selectbox muestre el nombre del alumno pero nos devuelva el ID
                student_dict = {d['alumno_id']: d['alumno_name'] for d in session_data}
                sel_student_id = st.selectbox("Selecciona un alumno para ver su dashboard:", list(student_dict.keys()), format_func=lambda x: student_dict[x])
                
                if sel_student_id:
                    # Encontramos los datos específicos de ese alumno en esa sesión
                    student_specific_data = next(d for d in session_data if d['alumno_id'] == sel_student_id)
                    st.markdown(f"**Viendo datos de: {student_specific_data['alumno_name']} - {selected_prof_session}**")
                    st.divider()
                    
                    # Usamos la función reutilizable para pintar el mismo dashboard que ve el alumno
                    render_student_dashboard(
                        history=student_specific_data['task_history'], 
                        final_levels=student_specific_data['competency_levels'], 
                        current_session=selected_prof_session,
                        enable_evolution_chart=False 
                    )

        st.divider()
        if st.button("🚪 Cerrar Sesión", key="logout_prof_2"):
            st.session_state.clear()
            st.rerun()

# 3. PANTALLA DEL ALUMNO
elif st.session_state.user_role == 'alumno':

    if st.session_state.current_step == "setup":
        st.title(f"🎓 Tutor IA - Ciencias")
        st.markdown(f"¡Hola **{st.session_state.user_name}**! Antes de empezar, nos gustaría conocerte mejor.")
        
        sessions = get_available_sessions()
        selected_session = st.selectbox("📂 Selecciona la sesión que deseas realizar:", sessions)
        st.divider()
        
        st.info("¿Deseas realizar un test inicial para adaptar el nivel de las tareas?")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Sí, realizar test inicial", use_container_width=True):
                start_agent(do_test=True, session_id=selected_session)
        with c2:
            if st.button("❌ No, ir directamente a la sesión", use_container_width=True):
                start_agent(do_test=False, session_id=selected_session)

    else:
        state_vals = app.get_state(config).values
        is_test_mode = get_val(state_vals, "test_mode")
        competencies = get_val(state_vals, "competency_levels") or {}
        
        render_competency_sidebar(competencies, is_test_mode)
        
        task = st.session_state.display_task

        if not task:
            # --- DASHBOARD FINAL DE SESIÓN (ALUMNO) ---
            st.balloons()
            st.title("🏆 ¡Sesión finalizada con éxito!")
            st.markdown("Aquí tienes el resumen de tu rendimiento en esta sesión. ¡Buen trabajo!")
            st.progress(1.0)
            
            history = get_val(state_vals, "task_history") or []
            final_levels = competencies
            current_session = get_val(state_vals, "current_session")
            
            # --- GUARDAR EN BASE DE DATOS ---
            if history:
                save_student_results(
                    alumno_id=st.session_state.user_id,
                    session_id=current_session,
                    competency_levels=final_levels,
                    task_history=history
                )
            
            # Llamada a la función reutilizable para pintar el dashboard
            render_student_dashboard(history, final_levels, current_session, enable_evolution_chart=True)

            # --- PESTAÑA EXTRA SOLO PARA ALUMNO: OPINIÓN ---
            st.markdown("### 🗣️ Mi Opinión")
            st.write("¿Qué te ha parecido la sesión de hoy?")
            student_rating = st.feedback("faces")
            if student_rating is not None:
                if student_rating >= 3: st.success("¡Nos alegra mucho que te haya gustado!")
                elif student_rating == 2: st.info("¡Gracias por tu valoración! Seguiremos mejorando.")
                else: st.warning("Sentimos que haya sido dura. ¡Revisaremos el nivel para la próxima!")

            st.divider()
            
            all_sessions = get_available_sessions()
            try:
                current_idx = all_sessions.index(current_session)
                if current_idx < len(all_sessions) - 1:
                    next_session = all_sessions[current_idx + 1]
                    st.info(f"Tienes una sesión pendiente: **{next_session}**")
                    
                    if st.button(f"➡️ Continuar a {next_session}", use_container_width=True):
                        new_tasks = load_curriculum(next_session)
                        app.update_state(config, {
                            "current_session": next_session, "remaining_tasks": new_tasks, "total_tasks": len(new_tasks),
                            "current_task": new_tasks[0], "current_task_prompt": new_tasks[0].get("task_prompt", ""),
                            "task_history": [], "last_feedback_message": None
                        })
                        st.session_state.display_task = new_tasks[0]
                        st.session_state.display_prompt = new_tasks[0].get("task_prompt", "")
                        st.session_state.current_step = "working"
                        st.session_state.last_feedback = None
                        st.rerun()
                        
            except ValueError: pass 

            if st.button("↻ Terminar y Salir por hoy", use_container_width=True):
                st.session_state.clear()
                st.rerun()
                
        else:
            # --- LÓGICA DE TAREAS EN VIVO ---
            if is_test_mode: st.warning("📝 MODO TEST INICIAL - Responde lo mejor que puedas")
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