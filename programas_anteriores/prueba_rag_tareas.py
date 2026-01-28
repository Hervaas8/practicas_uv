# **Archivo: probar_rag_contexto.py (Versión k=3 Fijo)**

import json
import sys
from configuracion_llm import get_retriever # Importamos el retriever (con k=3 fijo)
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    """Convierte la lista de documentos RAG en un string limpio y legible."""
    if not docs:
        return "--- SIN CONTEXTO RECUPERADO ---"
    
    clean_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        chunk_header = f"--- [Chunk {i+1} | Fuente: {source} (Pág: {page})] ---"
        
        # Limpieza de texto
        dirty_text = doc.page_content
        text_no_weird_spaces = dirty_text.replace('\xa0', ' ')
        text_no_newlines = text_no_weird_spaces.replace('\n', ' ')
        words = text_no_newlines.split()
        clean_text = ' '.join(words)
        
        clean_chunks.append(f"{chunk_header}\n{clean_text}")
        
    return "\n\n".join(clean_chunks)

# --- Archivos de Configuración ---
CURRICULUM_FILE = "curriculum.json"
OUTPUT_LOG_FILE = "rag_context_log_k3_fijo.txt" # Nombre de archivo nuevo

def test_all_task_contexts_k3_fijo():
    """
    Recorre CADA tarea del currículum, ejecuta el RAG (con k=3 fijo)
    y guarda el contexto recuperado en un archivo de log.
    """
    print(f"Iniciando prueba de contexto RAG (k=3 Fijo)...")
    print(f"Leyendo currículum desde: {CURRICULUM_FILE}")
    
    # 1. Cargar el Currículum
    try:
        with open(CURRICULUM_FILE, "r", encoding="utf-8-sig") as f:
            curriculum = json.load(f)
    except Exception as e:
        print(f"ERROR: Error al leer {CURRICULUM_FILE}. Detalle: {e}")
        return
            
    # 2. Abrir el archivo de Log para escribir
    print(f"Escribiendo log en: {OUTPUT_LOG_FILE}")
    
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        
        total_tasks_tested = 0
        
        # 3. Iterar por todo el currículum (Materia -> Sesión -> Tarea)
        for materia, sesiones in curriculum.items():
            log_file.write(f"===================================================\n")
            log_file.write(f" MATERIA: {materia.upper()}\n")
            log_file.write(f"===================================================\n\n")
            
            for sesion, tasks in sesiones.items():
                log_file.write(f"----------------------------------------\n")
                log_file.write(f"  SESIÓN: {sesion}\n")
                log_file.write(f"----------------------------------------\n\n")
                
                # Cargamos el retriever UNA VEZ por sesión (ya tiene k=3)
                try:
                    retriever = get_retriever(subject=materia, session=sesion)
                except Exception as e:
                    print(f"ERROR: No se pudo crear el retriever para {materia}/{sesion}. Saltando esta sesión. Detalle: {e}")
                    log_file.write(f"*** ERROR AL CARGAR RETRIEVER PARA ESTA SESIÓN ***\n\n")
                    continue
                    
                # Iteramos por las tareas de esta sesión
                for task in tasks:
                    task_id = task.get("task_id", "ID_DESCONOCIDO")
                    task_prompt = task.get("task_prompt", "")
                    
                    # No hay lógica de 'override' ni 'k' dinámico
                    query_para_rag = task_prompt

                    log_file.write(f"### TAREA: {task_id} ###\n")
                    log_file.write(f"PROMPT (Consulta RAG): {query_para_rag} (k=3 Fijo)\n\n")
                    
                    # 4. Invocar RAG
                    try:
                        docs = retriever.invoke(query_para_rag) 
                        contexto_texto = format_docs(docs)
                        
                        log_file.write(f"--- CONTEXTO RAG RECUPERADO ---\n")
                        log_file.write(contexto_texto)
                        log_file.write(f"\n--- FIN DE CONTEXTO PARA {task_id} ---\n\n\n")
                        
                        total_tasks_tested += 1
                    
                    except Exception as e:
                        print(f"ERROR: Fallo al invocar RAG para la tarea {task_id}. Detalle: {e}")
                        log_file.write(f"*** ERROR AL INVOCAR RAG PARA ESTA TAREA ***\n\n")

    print(f"\n¡Prueba completada!")
    print(f"Se ha generado el archivo '{OUTPUT_LOG_FILE}' con el contexto RAG para {total_tasks_tested} tareas.")

if __name__ == "__main__":
    test_all_task_contexts_k3_fijo()