import csv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuración (Asegúrate de que coincida con tus otros scripts) ---
VECTOR_STORE_PATH = "chroma_db_materias"
EMBEDDING_MODEL = "BAAI/bge-m3"
GOLDEN_SET_FILE = "golden_set.csv"

def evaluar_golden_set():
    print("Iniciando evaluación del RAG con el Golden Set...")

    # 1. Cargar los componentes del RAG (solo una vez)
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Cargando base de datos Chroma: {VECTOR_STORE_PATH}...")
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    # 2. Inicializar contadores
    total_preguntas = 0
    aciertos = 0

    # 3. Leer y procesar el archivo CSV
    print(f"Leyendo archivo de pruebas: {GOLDEN_SET_FILE}...")
    try:
        with open(GOLDEN_SET_FILE, mode='r', encoding='utf-8') as f:
            # Usar DictReader para leer el CSV como un diccionario
            reader = csv.DictReader(f)
            
            for row in reader:
                pregunta = row['pregunta']
                materia = row['materia']
                sesion = row['sesion']
                documento_esperado = row['documento_esperado']

                total_preguntas += 1

                # 4. Configurar el filtro para esta pregunta específica
                if materia == "sourcing":
                    filtro_actual = {"subject": "sourcing"}
                else:
                    # Este es el filtro combinado que usará tu agente
                    filtro_actual = {
                        "$or": [
                            {"subject": "sourcing"},
                            {"$and": [{"subject": materia}, {"session": sesion}]}
                        ]
                    }
                
                # 5. Crear el retriever y buscar (solo el resultado k=1)
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 1, "filter": filtro_actual}
                )
                
                docs = retriever.invoke(pregunta)

                # 6. Evaluar el resultado
                if not docs:
                    # FALLO: No se encontró ningún documento
                    print(f"[FALLO] Pregunta: '{pregunta}' - No se encontraron documentos con el filtro.")
                else:
                    # ÉXITO: Se encontró un documento, vamos a comprobarlo
                    doc_encontrado_fuente = docs[0].metadata['source']
                    # Extraemos solo el nombre del archivo (ej. 'escenario1.pdf')
                    doc_encontrado_nombre = os.path.basename(doc_encontrado_fuente)
                    
                    if doc_encontrado_nombre == documento_esperado:
                        aciertos += 1
                        print(f"[ÉXITO] Pregunta: '{pregunta}' -> Encontrado: '{doc_encontrado_nombre}'")
                    else:
                        # FALLO: Se encontró un documento, pero es el incorrecto
                        print(f"[FALLO] Pregunta: '{pregunta}' -> Esperado: '{documento_esperado}', Encontrado: '{doc_encontrado_nombre}'")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{GOLDEN_SET_FILE}'.")
        return
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return

    # 7. Mostrar el informe final
    if total_preguntas > 0:
        hit_rate = (aciertos / total_preguntas) * 100
        print("\n--- INFORME DE EVALUACIÓN RAG ---")
        print(f"Total de Preguntas: {total_preguntas}")
        print(f"Aciertos (Hit @ k=1): {aciertos}")
        print(f"Tasa de Acierto (Hit Rate): {hit_rate:.2f}%")
        print("---------------------------------")
        if hit_rate < 80:
            print("Resultado: El Hit Rate es bajo. Considera mejorar tus documentos o el chunking.")
        else:
            print("Resultado: ¡Excelente! El RAG es fiable.")
    else:
        print("No se encontraron preguntas en el archivo CSV.")

if __name__ == "__main__":
    evaluar_golden_set()