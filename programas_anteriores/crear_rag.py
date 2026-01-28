import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Importamos FAISS

# --- Configuración ---
PDF_DIRECTORY_PATH = "documentos"
# FAISS se guarda como una carpeta con un índice
VECTOR_STORE_PATH = "faiss_index_asignatura" 
# Modelo de embedding actualizado
EMBEDDING_MODEL = "BAAI/bge-m3" 

def construir_rag_faiss():
    print(f"Iniciando el proceso de ingesta desde: {PDF_DIRECTORY_PATH}")

    # 1. CARGAR (Load)
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY_PATH)
    documentos = loader.load()
    
    if not documentos:
        print("Error: No se encontraron documentos PDF en la carpeta.")
        return
    
    print(f"Cargados {len(documentos)} documentos PDF.")

    # 2. TROCEAR (Split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    trozos = text_splitter.split_documents(documentos)
    print(f"Documentos divididos en {len(trozos)} trozos (chunks).")

    # 3. EMBED (Embeddings)
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, # Cambiar a 'cuda' si tienes GPU
        # bge-base recomienda normalizar embeddings
        encode_kwargs={'normalize_embeddings': True} 
    )
    print("Modelo de embeddings cargado.")

    # 4. GUARDAR (Store) - Usando FAISS
    # FAISS primero crea el índice en memoria desde los documentos
    print(f"Creando índice FAISS en memoria...")
    vectorstore = FAISS.from_documents(
        documents=trozos,
        embedding=embeddings
    )
    
    # Luego, lo guardamos en disco
    print(f"Guardando índice FAISS en disco en: {VECTOR_STORE_PATH}...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    
    print("¡Índice FAISS creado y guardado con éxito!")

if __name__ == "__main__":
    if not os.path.exists(PDF_DIRECTORY_PATH):
        os.makedirs(PDF_DIRECTORY_PATH)
        print(f"Carpeta '{PDF_DIRECTORY_PATH}' creada.")
        print("Por favor, añade tus archivos PDF y vuelve a ejecutar el script.")
    else:
        construir_rag_faiss()