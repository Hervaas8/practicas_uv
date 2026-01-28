import os
import sys
from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text_from_image(image_path: str) -> str:
    """
    Intenta extraer texto de una imagen usando OCR.
    """
    # 1. Chequeo de librerías
    if 'pytesseract' not in sys.modules or 'PIL' not in sys.modules:
        return "[ERROR SISTEMA: Librerías 'pillow' o 'pytesseract' no instaladas.]"

    # 2. Verificar archivo
    if not os.path.exists(image_path):
        # Buscamos en subcarpeta 'imagenes' por si acaso
        alt_path = os.path.join("imagenes", image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            return f"[ERROR: Imagen '{image_path}' no encontrada]"

    try:
        # 3. Ejecutar OCR
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='spa+eng') # Español e Inglés
        
        clean_text = text.strip()
        if not clean_text:
            return "[OCR: La imagen no contiene texto legible]"
            
        print(f"   (OCR extrajo {len(clean_text)} caracteres)")
        return f"[TRANSCRIPCIÓN DE IMAGEN]:\n{clean_text}"

    except Exception as e:
        return f"[ERROR OCR: {str(e)}]"