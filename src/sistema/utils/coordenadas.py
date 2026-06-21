import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Generador de Regiones", layout="wide")

st.title("✂️ Selector de Zonas para Tareas Interactivas")
st.markdown("""
1. Sube la imagen.
2. Dibuja un rectángulo sobre la zona correcta.
3. Copia el código JSON generado abajo.
""")

# 1. Subir Imagen
uploaded_file = st.file_uploader("Sube tu imagen (PNG o JPG):", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Abrimos la imagen para obtener sus dimensiones reales
    image = Image.open(uploaded_file)
    w_img, h_img = image.size
    
    st.write(f"Dimensiones de la imagen: {w_img}x{h_img} px")
    st.write("---")

    # 2. Configurar el Canvas (Lienzo de dibujo)
    # stroke_width: grosor del borde
    # stroke_color: color del borde
    # fill_color: color de relleno (con transparencia rgba)
    # background_image: la imagen que subiste
    # height/width: dimensiones del canvas (iguales a la imagen para precisión 1:1)
    # drawing_mode: forzamos que sea "rect" (rectángulo)
    
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",  # Verde semitransparente
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=image,
        update_streamlit=True,
        height=h_img,
        width=w_img,
        drawing_mode="rect",
        key="canvas_selector",
        display_toolbar=True, # Muestra la barra para deshacer/borrar
    )

    # 3. Procesar lo dibujado
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if len(objects) > 0:
            # Cogemos el último objeto dibujado (por si dibujas varios y borras)
            last_object = objects[-1]
            
            # Extraemos los datos que necesita tu sistema
            # 'left' es X, 'top' es Y, 'width' es W, 'height' es H
            x = int(last_object["left"])
            y = int(last_object["top"])
            w = int(last_object["width"])
            h = int(last_object["height"])
            
            st.success("✅ ¡Zona capturada!")
            
            st.subheader("Copiar para `tareas.json`:")
            
            # Formateamos el diccionario listo para tu JSON
            json_block = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "is_correct": True,
                "feedback_ok": "¡Correcto! (Edita este mensaje)"
            }
            
            st.code(str(json_block).replace("'", '"'), language="json")
            
            st.info(f"Datos crudos: x={x}, y={y}, w={w}, h={h}")
            
        else:
            st.info("Dibuja un rectángulo en la imagen para obtener los datos.")