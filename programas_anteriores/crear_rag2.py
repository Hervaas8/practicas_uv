import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuración ---
PDF_DIRECTORY_PATH = "documentos"
SOURCING_DIRECTORY_PATH = "sourcing" 
VECTOR_STORE_PATH = "chroma_db_materias" 
EMBEDDING_MODEL = "BAAI/bge-m3"

def construir_rag_con_metadatos_y_sourcing():
    print("Iniciando ingesta...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    todos_los_trozos = []

    # --- 1. CARGA DE MATERIAS (Específico) ---
    print(f"Cargando documentos específicos desde: {PDF_DIRECTORY_PATH}")
    if os.path.exists(PDF_DIRECTORY_PATH):
        for subject_folder in os.listdir(PDF_DIRECTORY_PATH):
            subject_path = os.path.join(PDF_DIRECTORY_PATH, subject_folder)
            if not os.path.isdir(subject_path): continue

            for session_folder in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session_folder)
                if not os.path.isdir(session_path): continue
                    
                print(f"-> Cargando: Materia='{subject_folder}', Sesion='{session_folder}'")
                loader = PyPDFDirectoryLoader(session_path)
                documentos = loader.load()

                for doc in documentos:
                    doc.metadata["subject"] = subject_folder
                    doc.metadata["session"] = session_folder
                
                trozos = text_splitter.split_documents(documentos)
                todos_los_trozos.extend(trozos)
    else:
        print(f"Advertencia: No se encontró la carpeta {PDF_DIRECTORY_PATH}.")

    # --- 2. CARGA DE SOURCING (Global) ---
    print(f"\nCargando documentos globales desde: {SOURCING_DIRECTORY_PATH}")
    if os.path.exists(SOURCING_DIRECTORY_PATH):
        loader_sourcing = PyPDFDirectoryLoader(SOURCING_DIRECTORY_PATH)
        documentos_sourcing = loader_sourcing.load()

        # Añadimos metadato simple para identificar estos documentos
        for doc in documentos_sourcing:
            doc.metadata["subject"] = "sourcing" # Usamos "sourcing" como la "materia"
        
        trozos_sourcing = text_splitter.split_documents(documentos_sourcing)
        todos_los_trozos.extend(trozos_sourcing)
        print(f"-> Cargados {len(documentos_sourcing)} docs de sourcing, divididos en {len(trozos_sourcing)} trozos.")
    else:
        print(f"Advertencia: No se encontró la carpeta {SOURCING_DIRECTORY_PATH}.")


    if not todos_los_trozos:
        print("Error: No se encontraron documentos para indexar.")
        return

    print(f"\nTotal de trozos a indexar: {len(todos_los_trozos)}")

    # --- 3. EMBED y 4. GUARDAR (Unificado) ---
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Creando/actualizando base de datos Chroma en: {VECTOR_STORE_PATH}...")
    # Borra la base de datos antigua para reconstruirla
    if os.path.exists(VECTOR_STORE_PATH):
        import shutil
        print(f"Borrando base de datos antigua en {VECTOR_STORE_PATH}...")
        shutil.rmtree(VECTOR_STORE_PATH)

    vectorstore = Chroma.from_documents(
        documents=todos_los_trozos,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("¡Base de datos Chroma con metadatos y sourcing creada con éxito!")

if __name__ == "__main__":
    construir_rag_con_metadatos_y_sourcing()