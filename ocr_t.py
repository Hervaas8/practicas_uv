import os
try:
    from utils_easy import extract_text_from_image
except ImportError:
    print("❌ ERROR: No se encuentra el archivo 'utils_ocr.py' o la función 'extract_text_from_image'.")
    exit(1)

# Usamos las imágenes de las noticias que sabemos que tienen bastante texto
imagenes_a_probar = [
    "./data/img/actualidad.png",
    "./data/img/naquera.png"
]

print("🔍 Iniciando prueba del OCR...\n")

for ruta_img in imagenes_a_probar:
    print(f"📸 Procesando imagen: {ruta_img}")
    
    if not os.path.exists(ruta_img):
        print(f"⚠️ AVISO: La imagen no existe en la ruta especificada. Revisa si la carpeta './data/img/' está en el lugar correcto.")
        print("-" * 40)
        continue
        
    try:
        texto_extraido = extract_text_from_image(ruta_img)
        print("✅ Resultado del OCR:")
        print(texto_extraido if texto_extraido.strip() else "[El OCR no devolvió ningún texto]")
    except Exception as e:
        print(f"❌ Error al ejecutar el OCR en esta imagen: {e}")
        
    print("-" * 40)