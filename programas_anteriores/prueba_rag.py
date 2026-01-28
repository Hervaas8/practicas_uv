from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Importamos FAISS

# --- Configuración (Debe ser la misma) ---
VECTOR_STORE_PATH = "faiss_index_asignatura"
EMBEDDING_MODEL = "BAAI/bge-m3"

def probar_retriever_faiss():
    print("Cargando modelo de embeddings...")

    # Cargar el modelo de embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Cargar el índice FAISS desde el disco
    # NOTA: FAISS usa 'pickle' para guardar, lo que requiere 
    #       'allow_dangerous_deserialization=True' en versiones recientes de LangChain
    print(f"Cargando índice FAISS desde: {VECTOR_STORE_PATH}...")
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True # Necesario para cargar el índice
    )

    # Crear el "Recuperador" (Retriever)
    retriever = vectorstore.as_retriever(
        search_type="similarity", # Busca por similitud
        search_kwargs={"k": 3}      # Devuelve los 3 mejores resultados
    )

    print("Índice FAISS cargado. Realizando consulta de prueba...")

    # --- Consulta de Prueba ---
    pregunta_de_prueba = "Explica el concepto de presas" 
    
    documentos_recuperados = retriever.invoke(pregunta_de_prueba)

    print(f"\n--- RESULTADOS PARA: '{pregunta_de_prueba}' ---")
    
    if not documentos_recuperados:
        print("No se encontraron documentos relevantes.")
    else:
        for i, doc in enumerate(documentos_recuperados):
            print(f"\n--- Documento {i+1} ---")
            print(doc.page_content)
            print(f"Fuente: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}")
            print("-----------------")

if __name__ == "__main__":
    probar_retriever_faiss()