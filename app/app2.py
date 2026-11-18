import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid, os, re, json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- IMPORTACIONES DEL AGENTE Y BASE DE DATOS ---
from sistema.agente2 import app, load_test_inicial, load_curriculum, load_curriculum_data
from sistema.modulos.motor_evaluacion import generate_strategy_feedback
from sistema.utils.database import (
    authenticate_user, save_student_results, get_professor_students_data, save_student_rating,
)
from sistema.utils.esquemas import get_strategies, normalized_score, aggregate_score
from sistema.modulos.detector_plagio import auditar_sesion_completa

from sistema.config import resolve_project_path, TASKS_PATH, INITIAL_PATH, IMAGES_DIR, PRE_DIR

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor IA", layout="wide")

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

def get_available_data():
    try:
        return load_curriculum_data()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


def set_display_task(state):
    st.session_state.display_task = get_val(state, "current_task")
    st.session_state.display_prompt = get_val(state, "current_task_prompt")
    st.session_state.current_step = "working"
    st.session_state.last_feedback = None
    st.session_state.answer_started_at = time.monotonic()
    st.session_state.evaluation_wait = 0.0
    st.session_state.evaluation_error = False
    st.session_state.navigation_history = []


def start_agent(do_test, subject_id, scenario_id, session_id, initial_levels=None):
    try:
        curriculum = load_curriculum(subject_id, session_id, scenario_id)
        initial_tasks = load_test_inicial() if do_test else curriculum
    except ValueError as exc:
        st.error(str(exc))
        return
    default_levels = {} if do_test else (initial_levels or {
        "navegar": "nivel_1", "integrar": "nivel_1", "evaluar": "nivel_1",
    })

    app.invoke({
        "student_id": st.session_state.user_name,
        "remaining_tasks": initial_tasks,
        "total_tasks": len(initial_tasks),
        "test_mode": do_test,
        "competency_levels": default_levels,
        "current_subject": subject_id,
        "current_scenario": scenario_id,
        "current_session": session_id,
        "task_history": [],
        "detailed_feedback": {},
        "current_task": None,
        "current_student_response": None,
        "current_response_time": 0.0,
        "current_evaluation_time": 0.0,
        "current_navigation_history": [],
        "last_evaluation": None,
    }, config=config)
    
    new_state = app.get_state(config).values
    st.session_state.attempt_id = str(uuid.uuid4())
    set_display_task(new_state)
    st.rerun()

def submit_answer(user_input, retry=False):
    if retry:
        elapsed = st.session_state.response_elapsed
    else:
        elapsed = max(0.0, time.monotonic() - st.session_state.answer_started_at
                      - st.session_state.evaluation_wait)
    st.session_state.response_elapsed = elapsed
    started = time.monotonic()
    with st.spinner("Evaluando..."):
        app.update_state(config, {
            "current_student_response": user_input,
            "current_response_time": elapsed,
            "current_navigation_history": st.session_state.navigation_history,
        })
        try:
            for event in app.stream(None, config=config):
                pass
        except ValueError as exc:
            st.session_state.evaluation_wait += time.monotonic() - started
            st.error(str(exc))
            return
        st.session_state.evaluation_wait += time.monotonic() - started
        new_state = app.get_state(config).values
        st.session_state.last_feedback = get_val(new_state, "last_feedback_message")
        result = get_val(new_state, "last_evaluation") or {}
        st.session_state.evaluation_error = result.get("status") == "error"
        st.session_state.current_step = "working" if st.session_state.evaluation_error else "reviewed"
    st.rerun()

def handle_next():
    new_state = app.get_state(config).values
    set_display_task(new_state)
    st.rerun()

def render_student_dashboard(history, final_levels, current_session, detailed_feedback=None, enable_evolution_chart=False):
    if detailed_feedback is None:
        detailed_feedback = {}
    if any(float(get_val(entry, "max_score") or 0) <= 0 for entry in history):
        st.warning("Hay puntuaciones antiguas sin un máximo válido; se excluyen de las notas de resumen.")
        
    max_streak = 0
    current_streak = 0
    for h in history:
        score = normalized_score(h)
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
            for strat in get_strategies(h):
                strat_scores[strat].append(h)
        
        avg_scores = {k: aggregate_score(v) for k, v in strat_scores.items()}
            
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
                
                if detailed_feedback and strat_key in detailed_feedback:
                    with st.expander(f"🔍 Ver análisis detallado de {strat_name}"):
                        st.write(detailed_feedback[strat_key])
                    
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
            st.info("No hay datos históricos de evolución disponibles.")

    with tab_contenidos:
        st.markdown("### 📊 Progreso en Contenidos")
        content_history = [h for h in history if (h.get('contenido', False) if isinstance(h, dict) else getattr(h, 'contenido', False))]
        
        if not content_history:
            st.info("En esta sesión no se han evaluado tareas de contenido específico.")
        else:
            global_score = aggregate_score(content_history)
            global_score_10 = global_score * 10 
            
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
                for c in c_list:
                    if c not in concept_scores: concept_scores[c] = []
                    concept_scores[c].append(h)
            
            if concept_scores:
                domina, repasar, medio = [], [], []
                for c, s_list in concept_scores.items():
                    avg = aggregate_score(s_list)
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
        available_sessions = sorted({(d['subject_id'], d['scenario_id'], d['session_id']) for d in completed_data})
        selected_prof_subject, selected_prof_scenario, selected_prof_session = st.selectbox(
            "📂 Selecciona la asignatura, escenario y sesión a analizar:", available_sessions,
            format_func=lambda value: f"{value[0].capitalize()} / {value[1]} / {value[2]}",
        )
        session_attempts = [d for d in completed_data
                            if d['subject_id'] == selected_prof_subject
                            and d['scenario_id'] == selected_prof_scenario
                            and d['session_id'] == selected_prof_session]
        latest_by_student = {}
        for attempt in session_attempts:
            latest_by_student.setdefault(attempt['alumno_id'], attempt)
        session_data = list(latest_by_student.values())
        st.caption("La visión de clase y la auditoría utilizan el último intento de cada alumno.")
        if any(float(task.get("max_score", 1)) <= 0
               for result in session_data for task in result['task_history']):
            st.warning("Algunas puntuaciones antiguas no tienen máximo válido y se excluyen de las notas.")
        
        if not session_data:
            st.warning("No hay datos completados por tus alumnos para esta sesión.")
        else:
            all_strat_scores = {"navegar": [], "integrar": [], "evaluar": []}
            all_concept_scores = {}
            student_global_scores = [] 
            class_strategy_history = []
            class_content_history = []
            
            for student in session_data:
                history = student['task_history']
                student_content_scores = []
                
                for task in history:
                    strategies = get_strategies(task)
                    for strat in strategies:
                        all_strat_scores[strat].append(task)
                    if strategies:
                        class_strategy_history.append(task)
                    
                    if task.get('contenido'):
                        student_content_scores.append(task)
                        class_content_history.append(task)
                        for c in task.get('conceptos', []):
                            if c not in all_concept_scores:
                                all_concept_scores[c] = []
                            all_concept_scores[c].append(task)
                
                s_avg = aggregate_score(student_content_scores)
                student_global_scores.append({"nombre": student['alumno_name'], "nota": s_avg})
            
            avg_strats = {k: aggregate_score(v) for k, v in all_strat_scores.items()}
            avg_concepts = {k: aggregate_score(v) for k, v in all_concept_scores.items()}
            
            global_strat_score = aggregate_score(class_strategy_history)
            global_content_score = aggregate_score(class_content_history)

            tab_general, tab_emparejamiento, tab_individual, tab_auditoria = st.tabs([
                "📊 Visión de la Clase", 
                "🤝 Emparejamiento", 
                "👤 Vista Individual de Alumno", 
                "🕵️ Posibles Plagios"
            ])            
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
                student_dict = {d['alumno_id']: d['alumno_name'] for d in session_data}
                sel_student_id = st.selectbox("Selecciona un alumno para ver su dashboard:", list(student_dict.keys()), format_func=lambda x: student_dict[x])
                
                if sel_student_id:
                    student_attempts = [d for d in session_attempts if d['alumno_id'] == sel_student_id]
                    attempt_index = st.selectbox(
                        "Intento:", range(len(student_attempts)),
                        format_func=lambda index: f"{student_attempts[index]['timestamp']} · Intento {len(student_attempts) - index}",
                    )
                    student_specific_data = student_attempts[attempt_index]
                    st.markdown(f"**Viendo datos de: {student_specific_data['alumno_name']} - {selected_prof_session}**")
                    st.divider()
                    
                    detailed_fb = student_specific_data.get('detailed_feedback', {})
                    
                    render_student_dashboard(
                        history=student_specific_data['task_history'], 
                        final_levels=student_specific_data['competency_levels'], 
                        current_session=selected_prof_session,
                        detailed_feedback=detailed_fb,
                        enable_evolution_chart=False 
                    )
            with tab_auditoria:
                st.markdown(f"### 🕵️ Auditoría de Originalidad - {selected_prof_session}")
                st.write("Analiza las respuestas de texto libre de esta sesión buscando coincidencias estructurales sospechosas entre los alumnos.")
                
                if "sin_identificar" in (selected_prof_subject, selected_prof_scenario):
                    st.info("Estos resultados antiguos no tienen asignatura o escenario identificables para cargar sus tareas.")
                if st.button("🔍 Iniciar Auditoría Completa", use_container_width=True, type="primary",
                             disabled="sin_identificar" in (selected_prof_subject, selected_prof_scenario)):
                    with st.spinner("Analizando respuestas y consultando a PoliGPT... Esto puede tardar un poco."):
                        
                        try:
                            tareas_sesion = load_curriculum(selected_prof_subject, selected_prof_session, selected_prof_scenario)
                        except ValueError as exc:
                            st.error(str(exc))
                            st.stop()
                        
                        subjetivas_ids = [
                            t.get("task_id") or t.get("id")
                            for t in tareas_sesion 
                            if t.get("task_type") in ("subjetivo", "v_y_d") or t.get("es_tabla")
                        ]
                        
                        reporte_plagio = auditar_sesion_completa(
                            session_data, 
                            valid_task_ids=subjetivas_ids
                        )
                        
                        if not reporte_plagio:
                            st.success("✅ No se han encontrado casos que requieran revisión en las respuestas abiertas.")
                        else:
                            st.info("Se muestran los casos revisados, distinguiendo el veredicto y los errores de evaluación.")
                            st.divider()
                            
                            for task_id, info_tarea in reporte_plagio.items():
                                st.markdown(f"#### 📝 Pregunta: *{info_tarea['prompt']}*")
                                
                                for alerta in info_tarea['alertas']:
                                    analysis = alerta['analisis']
                                    veredicto = analysis.get('veredicto') or "Error técnico"
                                    
                                    if veredicto == "Posible Plagio":
                                        color_icon = "🔴"
                                    elif veredicto == "Dudoso":
                                        color_icon = "🟠"
                                    elif analysis.get('status') == 'error':
                                        color_icon = "⚪"
                                    else:
                                        color_icon = "🟢" 
                                    
                                    similarity = alerta['similitud_matematica']
                                    similarity_text = f"{similarity}%" if similarity is not None else "No aplica"
                                    titulo_expander = f"{color_icon} {alerta['alumno_1']} vs {alerta['alumno_2']} - Similitud: {similarity_text} [{veredicto}]"
                                    
                                    with st.expander(titulo_expander):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**👤 {alerta['alumno_1']}:**\n> {alerta['respuesta_1']}")
                                        with col2:
                                            st.markdown(f"**👤 {alerta['alumno_2']}:**\n> {alerta['respuesta_2']}")
                                        
                                        st.divider()
                                        st.markdown(f"**🤖 Veredicto IA:** {alerta['analisis'].get('justificacion', '')}")
                                        
                                        frases = alerta['analisis'].get('frases_identicas', [])
                                        if frases:
                                            st.markdown("**🔍 Bloques idénticos detectados:**")
                                            for f in frases:
                                                st.write(f"- *\"{f}\"*")
                                
                                st.divider()

        st.divider()
        if st.button("🚪 Cerrar Sesión", key="logout_prof_2"):
            st.session_state.clear()
            st.rerun()

elif st.session_state.user_role == 'alumno':

    if st.session_state.current_step == "setup":
        st.title(f"🎓 Tutor IA")
        st.markdown(f"¡Hola **{st.session_state.user_name}**! Antes de empezar, nos gustaría conocerte mejor.")
        
        # --- CARGA DINÁMICA DE ASIGNATURAS Y SESIONES ---
        curriculum_data = get_available_data()
        subjects = list(curriculum_data.keys())
        selected_subject = st.selectbox("📚 Selecciona la asignatura:", subjects, format_func=lambda x: x.capitalize())
        
        if selected_subject:
            scenarios = list(curriculum_data[selected_subject].keys())
            selected_scenario = st.selectbox("🗂️ Selecciona el escenario:", scenarios,
                                             format_func=lambda x: x.replace("escenario", "Escenario "))
            sessions = list(curriculum_data[selected_subject][selected_scenario].keys())
            selected_session = st.selectbox("📂 Selecciona la sesión que deseas realizar:", sessions)
            
            st.divider()
            st.info("¿Deseas realizar un test inicial para adaptar el nivel de las tareas?")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Sí, realizar test inicial", use_container_width=True):
                    start_agent(do_test=True, subject_id=selected_subject, scenario_id=selected_scenario, session_id=selected_session)
            with c2:
                if st.button("❌ No, ir directamente a la sesión", use_container_width=True):
                    start_agent(do_test=False, subject_id=selected_subject, scenario_id=selected_scenario, session_id=selected_session)

    else:
        state_vals = app.get_state(config).values
        is_test_mode = get_val(state_vals, "test_mode")
        competencies = get_val(state_vals, "competency_levels") or {}
        current_subject = get_val(state_vals, "current_subject") or "ciencias"
        current_scenario = get_val(state_vals, "current_scenario") or "escenario1"
        current_session = get_val(state_vals, "current_session") or "sesion1"
        
        # --- NUEVA CABECERA SUPERIOR PARA EL ALUMNO (Reemplaza la barra lateral) ---
        col_user_1, col_user_2 = st.columns([8, 2])
        with col_user_1:
            st.info(f"👤 **Usuario:** {st.session_state.user_name} | 📚 **Asignatura:** {current_subject.capitalize()} | 🗂️ **Escenario:** {current_scenario} | 📂 **Sesión:** {current_session}")
        with col_user_2:
            if st.button("🚪 Cerrar Sesión", use_container_width=True, key="logout_student_top"):
                st.session_state.clear()
                st.rerun()
        # --------------------------------------------------------------------------
        
        task = st.session_state.display_task

        if not task:
            st.balloons()
            st.title("🏆 ¡Sesión finalizada con éxito!")
            st.markdown("Aquí tienes el resumen de tu rendimiento en esta sesión. ¡Buen trabajo!")
            st.progress(1.0)
            
            history = get_val(state_vals, "task_history") or []
            final_levels = competencies
            
            detailed_feedback = get_val(state_vals, "detailed_feedback") or {}
            
            if not detailed_feedback and history:
                with st.spinner("🤖 El Tutor IA está analizando tus respuestas para redactar tus consideraciones finales..."):
                    for strat_key in ["navegar", "integrar", "evaluar"]:
                        nivel_actual = final_levels.get(strat_key, "nivel_1")
                        texto = generate_strategy_feedback(strat_key, nivel_actual, history)
                        detailed_feedback[strat_key] = texto
                    
                    app.update_state(config, {"detailed_feedback": detailed_feedback})

            if history:
                save_student_results(
                    alumno_id=st.session_state.user_id,
                    subject_id=current_subject,
                    scenario_id=current_scenario,
                    session_id=current_session,
                    attempt_id=st.session_state.attempt_id,
                    competency_levels=final_levels,
                    task_history=history,
                    detailed_feedback=detailed_feedback
                )
            
            render_student_dashboard(history, final_levels, current_session, detailed_feedback, enable_evolution_chart=True)

            st.markdown("### 🗣️ Mi Opinión")
            st.write("¿Qué te ha parecido la sesión de hoy?")
            student_rating = st.feedback("faces")
            if student_rating is not None:
                save_student_rating(st.session_state.user_id, st.session_state.attempt_id, student_rating)
                if student_rating >= 3: st.success("¡Nos alegra mucho que te haya gustado!")
                elif student_rating == 2: st.info("¡Gracias por tu valoración! Seguiremos mejorando.")
                else: st.warning("Sentimos que haya sido dura. ¡Revisaremos el nivel para la próxima!")

            st.divider()
            
            all_sessions = list(get_available_data().get(current_subject, {}).get(current_scenario, {}).keys())
            try:
                current_idx = all_sessions.index(current_session)
                if current_idx < len(all_sessions) - 1:
                    next_session = all_sessions[current_idx + 1]
                    st.info(f"Tienes una sesión pendiente: **{next_session}**")
                    
                    if st.button(f"➡️ Continuar a {next_session}", use_container_width=True):
                        start_agent(False, current_subject, current_scenario, next_session, initial_levels=competencies)
                        
            except ValueError: pass 

            if st.button("↻ Terminar y Salir por hoy", use_container_width=True):
                st.session_state.clear()
                st.rerun()
                
        else:
            if is_test_mode: st.warning("📝 MODO TEST INICIAL - Responde lo mejor que puedas")
            col_izq, col_der = st.columns([1, 1], gap="large")

            with col_izq:
                if st.session_state.get("evaluation_error", False):
                    st.error(st.session_state.last_feedback)
                    if st.button("Reintentar evaluación"):
                        submit_answer(get_val(state_vals, "current_student_response"), retry=True)
                full_prompt = st.session_state.display_prompt
                t_type = str(task.get("task_type", "")).strip().lower()
                options_pattern = r"(?:^|\n)\s*([a-d][\)\.])\s*([^\n]+)"
                found_options = re.findall(options_pattern, full_prompt)

                if st.session_state.current_step == "working" and t_type == "navegacion_test":
                    split_match = re.search(r"(?i)(\n\s*Opciones:?)|(\n\s*[-•]?\s*[a-d][\)\.])", full_prompt)
                    clean_text = full_prompt[:split_match.start()].strip() if split_match else full_prompt
                    st.markdown(f"### {clean_text}")
                    st.info("💡 Explora la imagen de la derecha haciendo clics antes de responder.")
                    st.divider()
                    
                    if found_options:
                        selection = st.radio("Elige tu respuesta:", [f"{o[0]} {o[1]}" for o in found_options])
                        if st.button("Enviar Respuesta"): 
                            submit_answer(selection[0])
                    else:
                        selection = st.radio("Elige una opción:", ["a", "b", "c", "d"])
                        if st.button("Enviar Respuesta"): 
                            submit_answer(selection)

                elif st.session_state.current_step == "working" and t_type == "objetivo" and found_options:
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

                # --- OTRAS TAREAS ---
                else:
                    st.markdown(f"### {full_prompt}")
                    st.divider()
                    if st.session_state.current_step == "working":
                        if t_type == "interactivo":
                            st.info("🎯 Haz clic directamente en la imagen de la derecha para responder.")
                        elif t_type == "verdadero_falso":
                            with st.form(f"form_vf_{task.get('task_id', '0')}"):
                                answers = []
                                for i, statement in enumerate(task.get("afirmaciones", [])):
                                    answer = st.selectbox(
                                        f"{i + 1}. {statement['texto']}",
                                        [None, True, False],
                                        format_func=lambda value: {
                                            None: "Selecciona una opción",
                                            True: "Verdadero",
                                            False: "Falso",
                                        }[value],
                                        key=f"vf_{task.get('task_id', '0')}_{i}",
                                    )
                                    answers.append(answer)
                                if st.form_submit_button("Enviar respuestas"):
                                    if any(answer is None for answer in answers):
                                        st.error("Debes responder a todas las afirmaciones.")
                                    else:
                                        submit_answer(answers)
                        elif t_type == "v_y_d":
                            st.info("Completa la tabla:")
                            with st.form(f"form_vyd_{task.get('task_id', '0')}"):
                                r1c1, r1c2 = st.columns(2)
                                v = r1c1.text_area("Ventajas")
                                d = r1c2.text_area("Desventajas")
                                if st.form_submit_button("Enviar"):
                                    submit_answer(f"Ventajas: {v} | Desventajas: {d}")
                        
                        elif t_type == "conectar":
                            user_map = {}
                            for concept in task.get("concepts", []):
                                user_map[concept] = st.multiselect(f"Para: **{concept}**", task.get("definitions", []), key=f"k_{concept}")
                            if st.button("Validar"): submit_answer(user_map)
                            
                        # --- NUEVO: LÓGICA TABLA GRID (SI/NO Autocorregible) ---
                        elif t_type == "tabla_grid":
                            st.info(full_prompt)
                            user_answers = {}
                            
                            with st.form(f"form_grid_{task.get('task_id', '0')}"):
                                c1, c2, c3 = st.columns([1, 2, 1])
                                c1.markdown("**Motivos**")
                                c2.markdown("**Implicaciones**")
                                c3.markdown("**¿Causa de prisión hoy?**")
                                st.divider()
                                
                                opciones = ["---"] + task.get("opciones", ["SI", "NO"])
                                
                                for i, fila in enumerate(task.get("filas", [])):
                                    c1, c2, c3 = st.columns([1, 2, 1])
                                    motivo = fila.get("motivo", "")
                                    c1.write(motivo)
                                    c2.info(fila.get("implicacion", ""))
                                    
                                    user_answers[motivo] = c3.selectbox(
                                        "Selecciona", 
                                        opciones, 
                                        key=f"grid_{task.get('task_id')}_{i}",
                                        label_visibility="collapsed"
                                    )
                                    st.divider()
                                    
                                if st.form_submit_button("Enviar Respuestas", use_container_width=True):
                                    if "---" in user_answers.values():
                                        st.error("⚠️ Debes seleccionar una opción para todas las filas.")
                                    else:
                                        submit_answer(user_answers)

                        # --- NUEVO: LÓGICA TABLA SUBJETIVA (Con andamiaje) ---
                        elif task.get("es_tabla") is True:
                            st.info("📝 Por favor, completa cada uno de los apartados solicitados:")
                            user_answers = {}
                            valores_iniciales = task.get("valores_iniciales", {}) 
                            
                            with st.form(f"form_tabla_{task.get('task_id', '0')}"):
                                for apartado in task.get("apartados_tabla", []):
                                    texto_previo = valores_iniciales.get(apartado, "") 
                                    user_answers[apartado] = st.text_area(
                                        f"**{apartado}**", 
                                        value=texto_previo, 
                                        height=120
                                    )
                                
                                if st.form_submit_button("Enviar Tabla Completada", use_container_width=True):
                                    if any(not v.strip() for v in user_answers.values()):
                                        st.error("⚠️ Debes rellenar todos los campos de la tabla antes de enviar.")
                                    else:
                                        resp_formateada = "\n".join([f"### {apartado}:\n{respuesta}" for apartado, respuesta in user_answers.items()])
                                        submit_answer(resp_formateada)

                        # --- POR DEFECTO: TEXTO LIBRE ---
                        else:
                            resp = st.text_area("Escribe tu respuesta:", height=150)
                            if st.button("Enviar"): submit_answer(resp)
                
                if st.session_state.current_step != "working":
                    st.markdown(f"**Feedback:**\n{st.session_state.last_feedback}")
                    if st.button("Siguiente Tarea ➔"): handle_next()

            with col_der:
                t_type = str(task.get("task_type", "")).strip().lower()
                
                if t_type == "navegacion_test" and st.session_state.current_step == "working":
                    task_id_key = task.get("id") or task.get("task_id")
                    
                    if "ruta_nav" not in st.session_state or st.session_state.get("last_task_id") != task_id_key:
                        st.session_state.ruta_nav = ["inicio"]
                        st.session_state.last_task_id = task_id_key

                    pantalla_actual_id = st.session_state.ruta_nav[-1]
                    navigation_history = st.session_state.navigation_history
                    if not navigation_history or navigation_history[-1] != pantalla_actual_id:
                        navigation_history.append(pantalla_actual_id)
                    pantallas = task.get("pantallas", {})
                    pantalla_data = pantallas.get(pantalla_actual_id, {})
                    
                    img_raw = task.get("task_image", [""])
                    img_fallback = img_raw[0] if isinstance(img_raw, list) and len(img_raw) > 0 else img_raw
                    img_path = pantalla_data.get("image", img_fallback)

                    img_path_resolved = resolve_project_path(img_path)
                    
                    if img_path_resolved.exists():
                        coords = streamlit_image_coordinates(
                            str(img_path_resolved),
                             key=f"nav_{task_id_key}_{pantalla_actual_id}_{len(st.session_state.ruta_nav)}"
                                 )
                        
                        if coords:
                            for link in pantalla_data.get("links", []):
                                if (link["x"] <= coords["x"] <= link["x"] + link["w"]) and (link["y"] <= coords["y"] <= link["y"] + link["h"]):
                                    st.session_state.ruta_nav.append(link["target"])
                                    st.rerun()
                    else:
                        st.warning(f"Imagen no encontrada: {img_path}")

                    if len(st.session_state.ruta_nav) > 1:
                        if st.button("⬅️ Volver a la pantalla anterior"):
                            st.session_state.ruta_nav.pop()
                            st.rerun()

                else:
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
                            img_path_resolved = resolve_project_path(img_path)
                            with tabs[idx]:
                                if img_path_resolved.exists():
                                    if task.get("task_type") == "interactivo" and st.session_state.current_step == "working":
                                        coords = streamlit_image_coordinates(str(img_path_resolved), key=f"click_{task.get('task_id', i)}")
                                        if coords: submit_answer(f"CLICK: {coords['x']}, {coords['y']}")
                                    else:
                                        st.image(str(img_path_resolved), use_column_width=True)
                                else:
                                    st.warning(f"Imagen no encontrada: {img_path}")
                            idx += 1
            
            st.divider()
            total = get_val(state_vals, "total_tasks") or 1
            remaining = len(get_val(state_vals, "remaining_tasks"))
            
            completed = total - remaining
            progress_val = max(0.0, min(1.0, completed / total))
            
            st.progress(progress_val, text=f"Progreso de la sesión: {int(progress_val*100)}%")
