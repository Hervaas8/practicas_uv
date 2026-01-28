import json
import sys

# --- LÓGICA DE CÁLCULO (LA MISMA QUE USAREMOS EN EL AGENTE) ---

def calculate_starting_levels(scores: dict, total_questions_per_strategy: dict) -> dict:
    """
    Calcula los niveles basándose en el % de aciertos.
    Umbrales:
    - < 50% -> nivel_3 (Básico)
    - 50%-79% -> nivel_2 (Medio)
    - >= 80% -> nivel_1 (Avanzado)
    """
    levels = {}
    print("\n[CÁLCULO INTERNO]")
    for strategy, score in scores.items():
        total = total_questions_per_strategy.get(strategy, 0)
        
        if total == 0:
            ratio = 0
            print(f" - {strategy}: No hubo preguntas -> Se asigna nivel_2 por defecto.")
            levels[strategy] = "nivel_2"
            continue
            
        ratio = score / total
        percentage = ratio * 100
        
        if ratio < 0.5:
            assigned = "nivel_3"
        elif ratio < 0.8:
            assigned = "nivel_2"
        else:
            assigned = "nivel_1"
            
        levels[strategy] = assigned
        print(f" - {strategy}: {score}/{total} aciertos ({percentage:.0f}%) -> {assigned}")
        
    return levels

# --- SIMULACIÓN DEL TEST ---

def run_simulation():
    print("=== SIMULADOR DE TEST INICIAL ===")
    
    # 1. Cargar preguntas
    try:
        with open("test_inicial.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"-> Se han cargado {len(questions)} preguntas correctamente.\n")
    except FileNotFoundError:
        print("ERROR: No encuentro 'test_inicial.json'. Asegúrate de crearlo.")
        return

    # 2. Inicializar contadores
    user_scores = {"navegar": 0, "evaluar": 0, "integrar": 0}
    strategy_counts = {"navegar": 0, "evaluar": 0, "integrar": 0}

    # 3. Preguntar si quiere empezar
    start = input("Agente: ¿Quieres realizar la Evaluación Inicial? (si/no): ")
    if start.lower() not in ["si", "sí", "s", "yes", "y"]:
        print("\n[Usuario rechazó el test. Se asignarían niveles por defecto.]")
        return

    # 4. Bucle de preguntas
    for i, q in enumerate(questions):
        strategy = q["strategy"]
        strategy_counts[strategy] += 1
        
        print("-" * 40)
        print(f"PREGUNTA {i+1}/{len(questions)} [{strategy.upper()}]")
        print(q["question"])
        for opt in q["options"]:
            print(f"  {opt}")
            
        user_input = input("\nTu respuesta (a, b, c, d): ").strip().lower()
        
        # Validación simple
        correct_letter = q["correct"].lower()
        
        # Aceptamos "c" o "c)" o "c."
        if user_input.startswith(correct_letter):
            print("✅ ¡CORRECTO!")
            user_scores[strategy] += 1
        else:
            print(f"❌ INCORRECTO. La correcta era '{correct_letter}'.")

    # 5. Calcular Resultados Finales
    print("\n" + "="*30)
    print("RESULTADOS FINALES DE LA SIMULACIÓN")
    print("="*30)
    
    final_levels = calculate_starting_levels(user_scores, strategy_counts)
    
    print("\n[CONFIGURACIÓN RESULTANTE PARA EL AGENTE]")
    print(json.dumps(final_levels, indent=2))

if __name__ == "__main__":
    run_simulation()