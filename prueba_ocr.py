import os
import sys



# Importamos la función de tu módulo utils_ocr
try:
    from utils_ocr import extract_text_from_image
except ImportError:
    print("Error: No se encontró 'utils_ocr.py' en este directorio.")
    sys.exit(1)


# Carpeta donde tienes tus fotos (cámbialo si tu carpeta se llama diferente)
CARPETA_IMAGENES = "./data/img" 

def probar_ocr_en_carpeta():
    print(f"--- INICIANDO PRUEBA DE OCR EN CARPETA: '{CARPETA_IMAGENES}' ---")

    # 1. Verificar si la carpeta existe
    if not os.path.exists(CARPETA_IMAGENES):
        print(f"Error: La carpeta '{CARPETA_IMAGENES}' no existe.")
        print("Por favor, crea la carpeta y pon tus imágenes dentro.")
        return

    # 2. Listar archivos de imagen
    archivos = os.listdir(CARPETA_IMAGENES)
    imagenes = [f for f in archivos if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    if not imagenes:
        print(f"No se encontraron imágenes (png, jpg, jpeg...) en '{CARPETA_IMAGENES}'.")
        return

    print(f"Se encontraron {len(imagenes)} imágenes. Procesando...\n")

    # 3. Procesar cada imagen
    for i, nombre_imagen in enumerate(imagenes, 1):
        ruta_completa = os.path.join(CARPETA_IMAGENES, nombre_imagen)
        
        print(f"[{i}/{len(imagenes)}] Procesando: {nombre_imagen}...")
        
        # Llamada a tu función de OCR
        texto_extraido = extract_text_from_image(ruta_completa)
        
        print("-" * 40)
        print(texto_extraido)
        print("-" * 40 + "\n")

    print("--- PRUEBA FINALIZADA ---")

if __name__ == "__main__":
    probar_ocr_en_carpeta()