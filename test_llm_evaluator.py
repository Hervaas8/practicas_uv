import json
import yaml
from langchain_core.prompts import ChatPromptTemplate
from configuracion_llm import get_llm_evaluador

# Intentamos importar tu función OCR por si las tareas del test necesitan leer imágenes
try:
    from utils_easy import extract_text_from_image
except ImportError:
    def extract_text_from_image(path): return "[Texto de imagen no disponible en el test]"

# --- 1. CARGA DE DATOS Y CONFIGURACIÓN ---

def cargar_tareas(ruta_json="tareas2.json"):
    """Carga las tareas del JSON y las indexa por task_id."""
    with open(ruta_json, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    tareas_dict = {}
    for sesion, tareas in data.get("ciencias", {}).items():
        for t in tareas:
            if t.get("task_type") in ["subjetivo", "v_y_d"]: # Filtramos solo las que evalúa el LLM
                tareas_dict[t["task_id"]] = t
                
    return tareas_dict

def cargar_casos_prueba(ruta_casos="casos_prueba.json"):
    """Carga los ejemplos a testear."""
    try:
        with open(ruta_casos, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo {ruta_casos}.")
        return []

def cargar_prompt_template(ruta_yaml="prompts.yaml"):
    """Carga la plantilla del prompt exactamente igual que el motor de evaluación."""
    with open(ruta_yaml, "r", encoding="utf-8-sig") as f:
        prompts = yaml.safe_load(f)
    return prompts["eval_subjective_prompt"]

# --- 2. LÓGICA DE EVALUACIÓN IDÉNTICA AL MOTOR ---

def evaluar_con_llm(tarea, respuesta_alumno, llm, template_str):
    """Construye las variables y llama al LLM usando LangChain."""
    
    # 1. Preparar el contexto (Texto + OCR de imágenes)
    context_parts = [tarea.get("task_text", "")]
    images = tarea.get("task_image", [])
    if isinstance(images, str): images = [images]
    for img in images:
        context_parts.append(extract_text_from_image(img))
    final_context = "\n\n".join(filter(None, context_parts))
    
    # 2. Preparar la rúbrica
    rubric = tarea.get("rubric", "Evaluar calidad general.")
    rubric_text = json.dumps(rubric, ensure_ascii=False) if isinstance(rubric, dict) else str(rubric)
    
    # 3. Preparar la cadena de LangChain
    prompt_tmpl = ChatPromptTemplate.from_template(template_str)
    chain = prompt_tmpl | llm
    
    # 4. Invocar
    resultado = chain.invoke({
        "task_text": final_context,
        "rubrica_text": rubric_text,
        "task_prompt": tarea.get("task_prompt", ""),
        "student_answer": respuesta_alumno
    })
    
    return resultado

# --- 3. EJECUCIÓN PRINCIPAL ---

def ejecutar_tests():
    print("🚀 Iniciando batería de pruebas del LLM Evaluador...\n")
    
    llm = get_llm_evaluador()
    if llm is None:
        print("❌ Error: No se pudo inicializar el LLM.")
        return

    tareas = cargar_tareas("tareas2.json")
    casos_prueba = cargar_casos_prueba("casos_prueba.json")
    template_str = cargar_prompt_template("prompts.yaml")
    
    if not casos_prueba:
        print("No hay casos de prueba para ejecutar. Saliendo.")
        return

    aciertos = 0
    total = len(casos_prueba)
    
    print("-" * 50)
    for i, caso in enumerate(casos_prueba, 1):
        task_id = caso.get("task_id")
        respuesta = caso.get("respuesta_alumno")
        nota_esperada = float(caso.get("nota_esperada"))
        
        if task_id not in tareas:
            print(f"⚠️ Caso {i}: Tarea '{task_id}' no encontrada o no es evaluable por LLM. Omitiendo.")
            total -= 1
            continue
            
        tarea = tareas[task_id]
        print(f"🔄 Test {i}/{len(casos_prueba)} | Tarea: {task_id}")
        
        try:
            resultado = evaluar_con_llm(tarea, respuesta, llm, template_str)
            nota_obtenida = round(float(resultado.score), 2)
            
            # Tolerancia de 0.05 para evitar falsos negativos por decimales
            es_correcto = abs(nota_esperada - nota_obtenida) <= 0.05
            
            if es_correcto:
                aciertos += 1
                print(f"✅ PASSED: Esperada {nota_esperada} | Obtenida {nota_obtenida}")
            else:
                print(f"❌ FAILED: Esperada {nota_esperada} | Obtenida {nota_obtenida}")
                print(f"   💡 Razonamiento del LLM: {resultado.reasoning}")
                print(f"   📝 Respuesta evaluada: {respuesta}")
                
        except Exception as e:
            print(f"⚠️ ERROR de ejecución en este caso: {e}")
            
        print("-" * 50)

    if total > 0:
        precision = (aciertos / total) * 100
        print(f"\n📊 RESULTADOS FINALES:")
        print(f"Total evaluados: {total}")
        print(f"Aciertos exactos: {aciertos}")
        print(f"Desviaciones: {total - aciertos}")
        print(f"Precisión del LLM: {precision:.2f}%")

if __name__ == "__main__":
    ejecutar_tests()