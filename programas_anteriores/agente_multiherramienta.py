from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 🧠 Modelo local
llm = ChatOllama(model="mistral")

# 🧮 Calculadora segura
def safe_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool(
    name="CalculadoraSegura",
    func=safe_calculator,
    description="Resuelve operaciones matemáticas básicas. Usa +, -, *, /, **, etc."
)

# 🔍 Wikipedia en español
wiki_wrapper = WikipediaAPIWrapper(lang="es", top_k_results=3, doc_content_chars_max=1000)
wiki_tool = Tool(
    name="BuscadorWikipedia",
    func=wiki_wrapper.run,
    description="Usa esta herramienta para buscar información en Wikipedia en español sobre personas, fechas o eventos históricos."
)

tools = [calculator_tool, wiki_tool]

# 🚀 Creamos el agente
agent = initialize_agent(
    tools, llm, agent_type="zero-shot-react-description", verbose=True
)

# 🧪 Prueba
consulta = (
    "Busca en Wikipedia en español el año de nacimiento de Albert Einstein "
    "y calcula cuántos años tenía en 1920."
)
result = agent.invoke({"input": consulta})
print("\n🧩 Resultado final:", result["output"])

