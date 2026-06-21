import easyocr
import base64
import os
import io
from PIL import Image
from langchain_core.messages import HumanMessage
from sistema.utils.configuracion_llm import get_llm
from sistema.config import resolve_project_path

# ==========================================
# --- 1. MODELO ANTIGUO: EASYOCR ---
# ==========================================
reader = easyocr.Reader(['es'])

def extract_text_from_image(path):
    """(ANTIGUO) Extrae texto usando la librería local EasyOCR."""
    try:
        resultados = reader.readtext(path, detail=0)
        texto_final = "\n".join(resultados)
        return texto_final
    except Exception as e:
        return f"[ERROR OCR: No se pudo extraer texto de {path}. Detalles: {e}]"

# ==========================================
# --- 2. NUEVO MODELO: DEEPSEEK-OCR (API) ---
# ==========================================
def encode_image(image_path, max_size=(1024, 1024)):
    """Redimensiona y codifica la imagen a Base64 para la API."""
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None

def extract_text_with_llm(path):
    """(NUEVO) Extrae texto usando DeepSeek-OCR a través de la API configurada."""
    try:
        img_b64 = encode_image(path)
        if not img_b64:
            return f"[ERROR OCR: Imagen no encontrada en {path}]"

        llm_ocr = get_llm(tipo_tarea="ocr")

        instruccion_ocr = (
            "Extrae todo el texto visible en esta imagen. "
            "Devuelve única y exclusivamente el texto extraído, respetando los saltos de línea donde sea posible. "
            "No añadas ningún comentario, introducción, conclusión ni formateo markdown extra."
        )

        msg = HumanMessage(content=[
            {"type": "text", "text": instruccion_ocr},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ])

        resultado = llm_ocr.invoke([msg])
        return resultado.content.strip()

    except Exception as e:
        return f"[ERROR OCR LLM: No se pudo extraer texto de {path}. Detalles: {e}]"