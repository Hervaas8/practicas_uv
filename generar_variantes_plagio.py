"""
Generador de variantes de plagio usando Gemini.
Toma las respuestas originales de las 5 tareas seleccionadas (4 perfiles cada una)
y genera 6 variantes por respuesta usando la API de Gemini.

Requisitos:
    pip install google-generativeai pillow

Uso:
    1. Pon tu clave de Gemini en la variable GEMINI_API_KEY
    2. Asegúrate de que las imágenes están en ./data/img/
    3. Ejecuta: python generar_variantes_plagio.py
    4. El resultado se guarda en variantes_plagio.json
"""

import json
import time
import base64
import os
import google.generativeai as genai
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

GEMINI_API_KEY = "AQ.Ab8RN6KBQSrnGTpUBuJYc5PHT7V_gHSo4--z7TP4Q9NtvMDqyw"  # https://aistudio.google.com/app/apikey
OUTPUT_FILE = "variantes_plagio.json"
PAUSA_ENTRE_LLAMADAS = 10  # segundos entre llamadas (límite capa gratuita: 15 rpm)

# ==========================================
# DATOS: respuestas originales extraídas del txt
# Solo tareas subjetivas seleccionadas, perfiles: alto, normal, menos nivel, ambiguo
# ==========================================

TAREAS = [
    {
        "task_id": "cien_s1_t02",
        "pregunta": "¿Qué página es mejor para informarnos sobre la Sierra Calderona? Justifica tu respuesta de manera breve y con tus propias palabras.",
        "imagenes": ["./data/img/sierra_calderona.png"],
        "respuestas": {
            "alto": "La mejor página sería la de Parcs Naturals de la Comunitat Valenciana. Es una página oficial de la Generalitat, así que me parece más fiable. Las otras pueden servir, pero algunas son más de rutas o turismo.",
            "normal": "Yo creo que la mejor es la de Parcs Naturals, o sea la de la Generalitat. Porque parece oficial y habla del parque de verdad. Las otras también dicen cosas, pero son más de excursiones y eso.",
            "menos_nivel": "Yo pondría la de AllTrails porque sale la montaña y rutas. Pero también la de la Generalitat puede estar bien porque es del gobierno. Creo que esa es más verdadera.",
            "ambiguo": "Depende de lo que queramos buscar. Puede ser mejor la oficial, pero las otras también pueden tener información. Yo creo que más o menos la de Parcs Naturals, aunque no estoy seguro del todo.",
        },
    },
    {
        "task_id": "cien_s1_t04",
        "pregunta": "¿Cuáles son las ventajas y desventajas de construir un centro comercial cerca de la Sierra Calderona?",
        "imagenes": ["./data/img/actualidad.png"],
        "respuestas": {
            "alto": "Ventajas: creación de empleo y turismo local.\nDesventajas: aumento del tráfico y alteración de especies sensibles.",
            "normal": "Ventajas: crea empleo y puede traer más turismo, o sea, más dinero al pueblo.\nDesventajas: puede haber más tráfico y afectar a animales o plantas.",
            "menos_nivel": "Ventajas: que hacen tiendas y la gente compra más.\nDesventajas: se puede romper el abitad y los coches molestan.",
            "ambiguo": "Ventajas: puede crear empleo y ayudar al pueblo, más o menos.\nDesventajas: quizá aumente el tráfico y afecte al parque.",
        },
    },
    {
        "task_id": "cien_s1_t06",
        "pregunta": "¿Las dos fuentes de información son complementarias o contradictorias? Justifica tu respuesta.",
        "imagenes": ["./data/img/actualidad.png", "./data/img/naquera.png"],
        "respuestas": {
            "alto": "Son complementarios, porque los dos hablan del mismo centro comercial y dan ideas parecidas. Uno explica más los impactos ambientales y el otro cuenta mejor la opinión de vecinos y ecologistas.",
            "normal": "Yo creo que son complementarios, o sea, no dicen cosas contrarias. Los dos dicen que puede crear empleo, pero también que puede afectar al parque y al tráfico.",
            "menos_nivel": "Son contradictorias porque una dice que es bueno y la otra que es malo. Pero las dos hablan del centro comercial y de que puede aver trabajo.",
            "ambiguo": "Puede ser complementaria, pero depende de cómo se mire. Más o menos los dos textos dicen cosas parecidas, aunque no sé si cuentan exactamente lo mismo.",
        },
    },
    {
        "task_id": "cien_s1_t07",
        "pregunta": "¿Cuál es el tema principal de los dos textos? Resume las ideas más importantes.",
        "imagenes": ["./data/img/actualidad.png", "./data/img/naquera.png"],
        "respuestas": {
            "alto": "Los dos textos hablan de la propuesta de construir un centro comercial cerca de la Sierra Calderona. La idea principal es que puede traer empleo, turismo y más servicios, pero también puede causar problemas en el parque, como más tráfico, daño al ecosistema y pérdida de tranquilidad.",
            "normal": "Los textos tratan de un centro comercial que quieren hacer cerca de la Sierra Calderona. Puede ser bueno porque crea trabajo y viene más gente, pero también puede ser malo porque puede dañar el monte y aumentar los coches.",
            "menos_nivel": "Los textos dicen que van a hacer un centro comercial en la montaña. Eso puede ser bueno porque la gente compra y trabaja, pero malo porque los animales se pueden ir y hay muchos coches y basura.",
            "ambiguo": "Los textos hablan más o menos de un centro comercial cerca de la Sierra Calderona. Puede tener cosas buenas y malas, depende de si afecta mucho al parque o si al final lo hacen con cuidado.",
        },
    },
    {
        "task_id": "cien_s1_t09",
        "pregunta": "¿Estás de acuerdo con Elena o con Jaime sobre cómo usar los comentarios de Instagram para tomar una decisión? Justifica tu respuesta.",
        "imagenes": ["./data/img/mobil1.png", "./data/img/mobil2.png"],
        "respuestas": {
            "alto": "La postura de Elena me parece más acertada. Los comentarios de Instagram son opiniones de gente, pero antes de decidir hay que mirar fuentes oficiales y lo que digan los expertos.",
            "normal": "Yo creo que Elena tiene más razón, o sea, los comentarios pueden ayudar a saber lo que piensa la gente. Pero no vale solo con eso, hay que mirar información más fiable y ya.",
            "menos_nivel": "Yo creo que Jaime porque si todos ponen comentarios malos será que el centro comercial está mal. Aunque Elena dice de mirar expertos y eso también puede servir.",
            "ambiguo": "Puede que Elena tenga más razón, pero depende. Los comentarios dicen una cosa, aunque igual habría que mirar más información para estar seguros.",
        },
    },
]

# ==========================================
# PROMPT DE GENERACIÓN DE VARIANTES
# ==========================================

PROMPT_VARIANTES = """Eres un generador de datos sintéticos para evaluar un sistema de detección de plagio.
Dada una respuesta original de un alumno, genera 5 variantes que simulan distintos
niveles de copia. Cada variante debe partir del texto original y aplicar ÚNICAMENTE
la transformación indicada para ese nivel.

Pregunta original: "{pregunta}"
Respuesta original: "{respuesta_original}"

Genera exactamente estas 6 versiones:

NIVEL_0 (copia exacta):
Reproduce el texto original sin cambiar absolutamente nada.

NIVEL_1 (reordenación):
Cambia el orden de las frases del texto. No modifiques ninguna palabra dentro
de cada frase. El contenido de cada frase debe quedar intacto.

NIVEL_2 (sustitución léxica):
Sustituye aproximadamente el 25% de las palabras de contenido (sustantivos,
verbos, adjetivos) por sinónimos directos. No reescribas frases enteras ni
cambies la estructura. El resultado puede sonar algo forzado, como una
sustitución mecánica, eso es correcto.

NIVEL_3 (paráfrasis parcial):
Reescribe entre 1 y 3 frases completas con otras palabras, dejando el resto
exactamente igual. Indica entre corchetes qué frases has cambiado,
por ejemplo: [FRASE MODIFICADA: "nueva frase aquí"].

NIVEL_4 (paráfrasis total):
Reescribe el texto completo manteniendo el mismo significado pero sin reutilizar
ninguna secuencia de más de 3 palabras consecutivas del original. El registro
y la longitud deben ser similares al original.

NIVEL_5 (respuesta independiente):
Escribe una respuesta completamente nueva a la misma pregunta. No te bases en
el texto original en absoluto: ni su estructura, ni su léxico, ni su orden de
ideas. Debe ser una respuesta plausible de un alumno del mismo nivel que el original.

REGLAS GENERALES:
- Mantén en todas las variantes el mismo registro y nivel de lenguaje que el original.
- No mejores ni empeores la calidad de la respuesta respecto al original.
- No añadas ni elimines información relevante en los niveles 0-3.
- Cada nivel debe aplicar SOLO su transformación, sin mezclar técnicas de otros niveles.
- No incluyas explicaciones ni comentarios fuera de los corchetes indicados en NIVEL_3.

Devuelve ÚNICAMENTE un objeto JSON válido con este formato, sin texto adicional, sin bloques de código:
{{
  "nivel_0": "...",
  "nivel_1": "...",
  "nivel_2": "...",
  "nivel_3": "...",
  "nivel_4": "...",
  "nivel_5": "..."
}}
"""

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def cargar_imagen_base64(ruta: str) -> dict | None:
    """Carga una imagen desde disco y la devuelve como parte inline para Gemini."""
    path = Path(ruta)
    if not path.exists():
        print(f"  Imagen no encontrada: {ruta} (se omite)")
        return None

    extension = path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_types.get(extension, "image/png")

    with open(path, "rb") as f:
        datos = base64.b64encode(f.read()).decode("utf-8")

    return {"mime_type": mime, "data": datos}

def llamar_gemini(model, pregunta: str, respuesta: str, imagenes: list[str]) -> dict | None:
    """Llama a Gemini con el prompt y opcionalmente imágenes. Devuelve el JSON de variantes."""
    prompt_texto = PROMPT_VARIANTES.format(
        pregunta=pregunta,
        respuesta_original=respuesta
    )

    # Construir contenido: primero las imágenes (si hay), luego el texto
    contenido = []
    for ruta_img in imagenes:
        img = cargar_imagen_base64(ruta_img)
        if img:
            contenido.append(img)
    contenido.append(prompt_texto)

    try:
        response = model.generate_content(contenido)
        texto = response.text.strip()

        # Limpiar posibles bloques de código markdown que Gemini a veces añade
        if texto.startswith("```"):
            texto = texto.split("```")[1]
            if texto.startswith("json"):
                texto = texto[4:]
        texto = texto.strip()

        return json.loads(texto)

    except json.JSONDecodeError as e:
        print(f"  ❌ Error parseando JSON: {e}")
        print(f"  Respuesta recibida: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"  ❌ Error en llamada a Gemini: {e}")
        return None

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    # Configurar Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    resultados = {}
    total = sum(len(t["respuestas"]) for t in TAREAS)
    contador = 0

    print(f"Generando variantes para {len(TAREAS)} tareas × 4 perfiles = {total} respuestas originales")
    print(f"Total de variantes a generar: {total * 6}\n")

    for tarea in TAREAS:
        task_id = tarea["task_id"]
        pregunta = tarea["pregunta"]
        imagenes = tarea["imagenes"]
        resultados[task_id] = {}

        print(f"📋 Tarea: {task_id}")

        for perfil, respuesta_original in tarea["respuestas"].items():
            contador += 1
            print(f"  [{contador}/{total}] Perfil: {perfil}...", end=" ", flush=True)

            variantes = llamar_gemini(model, pregunta, respuesta_original, imagenes)

            if variantes:
                resultados[task_id][perfil] = {
                    "respuesta_original": respuesta_original,
                    "variantes": variantes
                }
                print("✅")
            else:
                resultados[task_id][perfil] = {
                    "respuesta_original": respuesta_original,
                    "variantes": None,
                    "error": True
                }
                print("❌ (guardado con error para reintentar)")

            # Guardar progreso tras cada llamada por si falla a mitad
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

            # Pausa para no superar el límite de la capa gratuita
            if contador < total:
                time.sleep(PAUSA_ENTRE_LLAMADAS)

        print()

    print(f"\n✅ Proceso completado. Resultado guardado en: {OUTPUT_FILE}")

    # Resumen de errores
    errores = [
        f"{tid}/{perfil}"
        for tid, perfiles in resultados.items()
        for perfil, datos in perfiles.items()
        if datos.get("error")
    ]
    if errores:
        print(f"\n⚠️  {len(errores)} llamadas fallaron:")
        for e in errores:
            print(f"   - {e}")
        print("Puedes volver a ejecutar el script; saltará las que ya tienen variantes.")
    else:
        print("Sin errores.")


if __name__ == "__main__":
    main()
