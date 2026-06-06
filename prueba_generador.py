import yaml
from langchain_core.prompts import ChatPromptTemplate
from esquemas import LLMScaffoldingOutput
from configuracion_llm import get_llm_estructurado 

def probar_generacion_pistas():
    try:
        with open("prompts.yaml", "r", encoding="utf-8-sig") as f:
            prompts = yaml.safe_load(f)
        prompt_texto = prompts["generador_pistas_prompt"]
    except Exception as e:
        print(f"Error crítico al cargar prompts.yaml: {e}")
        return

    # 2. Cargamos el LLM forzando la salida a nuestro nuevo esquema Pydantic
    llm_generador = get_llm_estructurado("generador").with_structured_output(LLMScaffoldingOutput)
    
    # 3. Creamos el template con el texto extraído del YAML
    prompt_tmpl = ChatPromptTemplate.from_template(prompt_texto)
    cadena = prompt_tmpl | llm_generador
    
    # 4. Datos de prueba (Basados en la tarea cien_s1_t09 de tu tareas2.json)
    datos_prueba = {
        "contexto_tarea": "El alumno debe comparar un comentario de instagram con varias respuestas para decidir qué postura es más acertada sobre el impacto ambiental.",
        "enunciado_base": "He buscado en internet “Sierra Calderona” y esto es lo que he encontrado. Docente: ¿Qué página es mejor para informarnos sobre el tema? Justifica tu respuesta de manera breve y con tus propias palabras",
        "estrategia": "evaluar",
        "subestrategia": "Analizar expertise del autor"
    }
    
    # 5. Invocamos al LLM
    print("🤖 Consultando al LLM para generar las pistas...\n")
    try:
        resultado = cadena.invoke(datos_prueba)
        
        # 6. Ensamblamos y mostramos los 3 niveles finales
        print("=== RESULTADOS DEL ENSAMBLAJE ===")
        
        print("\n🟢 NIVEL ALTO (Sin pistas):")
        print(f"{datos_prueba['enunciado_base']}")
        
        print("\n🟠 NIVEL MEDIO (Enunciado + Pista 1):")
        print(f"{datos_prueba['enunciado_base']} {resultado.pista_1}")
        
        print("\n🔴 NIVEL BAJO (Enunciado + Pista 1 + Pista 2):")
        print(f"{datos_prueba['enunciado_base']} {resultado.pista_1} {resultado.pista_2}")
        
    except Exception as e:
        print(f"Error en la generación: {e}")

if __name__ == "__main__":
    probar_generacion_pistas()