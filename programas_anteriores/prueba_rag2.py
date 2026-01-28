from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuración ---
VECTOR_STORE_PATH = "chroma_db_materias"
EMBEDDING_MODEL = "BAAI/bge-m3"

# --- SIMULACIÓN DEL ESTADO DEL AGENTE ---
MATERIA_ACTUAL = "ciencias"
SESION_ACTUAL = "sesion1"

def probar_retriever_filtrado_avanzado():
    print(f"Cargando base de datos Chroma: {VECTOR_STORE_PATH}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    # --- El NUEVO FILTRO MÁGICO con $or ---
    # Este filtro busca documentos que cumplan CUALQUIERA de las dos condiciones
    filtro_combinado = {
        "$or": [
            # Condición 1: Documentos globales de "sourcing"
            {
                "subject": "sourcing"
            },
            # Condición 2: Documentos específicos de la materia Y sesión
            {
                "$and": [
                    { "subject": MATERIA_ACTUAL },
                    { "session": SESION_ACTUAL }
                ]
            }
        ]
    }

    print(f"Creando retriever filtrado para: (Materia='{MATERIA_ACTUAL}' Y Sesion='{SESION_ACTUAL}') O (Subject='sourcing')")
    
    retriever_filtrado = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": filtro_combinado}
    )

    print("Retriever cargado. Realizando consultas de prueba...")

    # --- Consulta de Prueba 1 (Específica de la materia) ---
    pregunta_presas = "Explica el concepto de presas"
    docs_presas = retriever_filtrado.invoke(pregunta_presas)
    
    print(f"\n--- RESULTADOS PARA: '{pregunta_presas}' ---")
    if not docs_presas:
        print("No se encontraron documentos relevantes.")
    else:
        for doc in docs_presas:
            print(f"-> Chunk: {doc.page_content[:100]}...")
            print(f"   Metadatos: {doc.metadata}")

    # --- Consulta de Prueba 2 (Global de Sourcing) ---
    # (Asumiendo que 'sourcing' tiene info sobre "fuentes primarias"
    #  como vimos en tus primeros resultados de prueba)
    pregunta_sourcing = "Qué es una fuente primaria"
    docs_sourcing = retriever_filtrado.invoke(pregunta_sourcing)
    
    print(f"\n--- RESULTADOS PARA: '{pregunta_sourcing}' ---")
    if not docs_sourcing:
        print("No se encontraron documentos relevantes.")
    else:
        for doc in docs_sourcing:
            print(f"-> Chunk: {doc.page_content[:100]}...")
            print(f"   Metadatos: {doc.metadata}")
            
    # --- Consulta de Prueba 3 (De otra materia, no debe aparecer) ---
    pregunta_rocas = "Qué son las rocas ígneas"
    # Cambiamos el retriever para que SÓLO busque en geología
    retriever_geologia = vectorstore.as_retriever(
        search_kwargs={"k": 1, "filter": {"subject": "geologia"}}
    )
    docs_rocas = retriever_geologia.invoke(pregunta_rocas)
    print(f"\n--- RESULTADOS PARA: '{pregunta_rocas}' (Solo geologia) ---")
    if not docs_rocas:
        print("-> No se encontraron documentos (o no hay docs de geologia).")
    else:
         for doc in docs_rocas:
            print(f"   Metadatos: {doc.metadata}")


if __name__ == "__main__":
    probar_retriever_filtrado_avanzado()