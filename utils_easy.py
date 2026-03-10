import easyocr

# Cargamos el modelo en español ('es').
# La primera vez que se ejecute, descargará unos archivos pequeños (modelos) en tu carpeta de usuario.
reader = easyocr.Reader(['es'])

def extract_text_from_image(path):
    try:
        # detail=0 hace que devuelva directamente una lista con el texto detectado
        resultados = reader.readtext(path, detail=0)
        # Unimos las líneas detectadas con saltos de línea
        texto_final = "\n".join(resultados)
        return texto_final
    except Exception as e:
        return f"[ERROR OCR: No se pudo extraer texto de {path}. Detalles: {e}]"