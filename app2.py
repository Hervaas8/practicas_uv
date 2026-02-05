import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import uuid
import os
import re
from typing import List, Union, Tuple

# Importaciones del Agente
from agente2 import app
from esquemas import StudentGraphState

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Tutor IA | Ciencias", 
    page_icon="🎓", 
    layout="wide"
)

# --- ESTILOS CSS (Solución de visibilidad) ---
st.markdown("""
    <style>
    /* Fondo general */
    .stApp { background-color: #F0F2F6; }
    
    /* Forzar color de texto en mensajes para que sea visible */
    [data-testid="stChatMessage"] p {
        color: #1E1E1E !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* Burbujas de chat */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
    }

    /* Estilo para las opciones en el historial */
    .history-options {
        color: #666 !important;
        font-style: italic;
        font-size: 0.9rem;
        border-left: 3px solid #DDD;
        padding-left: 10px;
        margin-top: 10px;
    }

    /* Zona de botones derecha */
    .stButton button {
        background-color: #262730 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 15px !important;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #40424a !important;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE APOYO ---
def normalize_images(image_data) -> List[str]:
    if not image_data: return []
    return [image_data] if isinstance(image_data, str) else image_data

def split_text_and_options(text: str) -> Tuple[str, List[str]]:
    if not text: return "", []
    # Busca patrones tipo a), b) o a. b. al inicio de línea
    pattern = r"(?m)^\s*([a-zA-Z][\)\.])\s+(.*)"
    matches = list(re.finditer(pattern, text))
    if len(matches) >= 2:
        clean_text = text[:matches[0].start()].strip()
        options = [m.group(0).strip() for m in matches]
        return clean_text, options
    return text, []

def handle_user_input(user_response_content):
    st.session_state.messages.append({"role": "user", "content": user_response_content})
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    app.update_state(config, {"current_student_response": user_response_content})
    
    with st.spinner("Evaluando..."):
        for event in app.stream(None, config=config): pass
            
    new_state = app.get_state(config).values
    
    # Limpieza si termina el test
    if st.session_state.get("was_in_assessment", False) and not new_state.get("in_initial_assessment"):
        st.session_state.messages = []
    st.session_state.was_in_assessment = new_state.get("in_initial_assessment")

    # Preparar respuesta del tutor
    feedback = new_state.get("last_feedback_message", "")
    current_task = new_state.get("current_task")
    next_prompt = new_state.get("current_task_prompt", "")
    
    # Unimos el feedback previo con el nuevo enunciado
    full_content = f"{feedback}\n\n{next_prompt}".strip()
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_content,
        "context_text": current_task.get("task_text") if current_task else None,
        "image_paths": normalize_images(current_task.get("task_image")) if current_task else []
    })
    st.rerun()

# --- LÓGICA DE SESIÓN ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}
if "graph_init" not in st.session_state:
    app.invoke({"student_id": "alumno_visual"}, config=config)
    state = app.get_state(config).values
    st.session_state.messages.append({
        "role": "assistant",
        "content": state.get("current_task_prompt"),
        "context_text": state.get("current_task", {}).get("task_text"),
        "image_paths": normalize_images(state.get("current_task", {}).get("task_image"))
    })
    st.session_state.graph_init = True

# --- INTERFAZ ---
col_chat, col_actions = st.columns([1.5, 1])

with col_chat:
    st.subheader("📖 Actividad")
    # Contenedor con scroll para el historial
    chat_subcontainer = st.container(height=650, border=False)
    with chat_subcontainer:
        for i, msg in enumerate(st.session_state.messages):
            is_last = (i == len(st.session_state.messages) - 1)
            with st.chat_message(msg["role"]):
                clean_text, options = split_text_and_options(msg["content"])
                st.markdown(clean_text)
                
                # En el historial, mostramos las opciones en pequeño
                if options and not is_last and msg["role"] == "assistant":
                    st.markdown(f'<div class="history-options">Opciones: {" | ".join(options)}</div>', unsafe_allow_html=True)

                if msg.get("context_text"):
                    with st.expander("Ver documento de referencia"):
                        st.write(msg["context_text"])
                
                for img_path in msg.get("image_paths", []):
                    if os.path.exists(img_path):
                        st.image(img_path)

with col_actions:
    st.subheader("⚡ Respuesta")
    current_state = app.get_state(config).values
    task = current_state.get("current_task")
    
    if task:
        # Extraemos opciones del último mensaje del asistente
        _, options = split_text_and_options(st.session_state.messages[-1]["content"])
        
        if task.get("task_type") == "interactivo":
            st.info("📍 Haz clic en la zona correcta de la imagen de la izquierda.")
            # Aquí iría la llamada a streamlit_image_coordinates si es el turno interactivo
        
        elif options:
            for opt in options:
                # Botones grandes para las opciones
                if st.button(opt, use_container_width=True, key=f"btn_{opt[:10]}_{uuid.uuid4()}"):
                    # Enviamos solo la letra (primer carácter)
                    letter = opt.strip()[0].lower()
                    handle_user_input(letter)
        else:
            # Entrada de texto para preguntas subjetivas
            user_input = st.chat_input("Escribe tu respuesta...")
            if user_input:
                handle_user_input(user_input)
    else:
        st.success("¡Has terminado la lección!")