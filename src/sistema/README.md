
---

## `src/sistema/README.md`

```md
# Paquete `sistema`

Esta carpeta contiene el código principal del prototipo de sistema tutor inteligente.

Aquí se implementan los módulos reutilizables del proyecto, incluyendo la lógica del agente, la evaluación automática de respuestas, la detección de similitud textual, la generación de enunciados adaptados, la configuración de modelos de lenguaje y las utilidades para el tratamiento de imágenes.

## Estructura general

- `agente.py`: define el flujo adaptativo del sistema tutor.
- `motor_evaluacion.py`: contiene la lógica de evaluación automática de respuestas.
- `detector_plagio.py`: implementa las métricas de similitud textual y la revisión mediante modelo de lenguaje.
- `generador_enunciados.py`: genera pistas y enunciados adaptados a distintos niveles de apoyo.
- `configuracion_llm.py`: centraliza la configuración de los modelos de lenguaje utilizados.
- `config.py`: centraliza rutas del proyecto, archivos JSON, prompts, resultados e imágenes.
- `database.py`: gestiona la persistencia de usuarios, sesiones y resultados.
- `utils_easy.py`: contiene utilidades de procesamiento de imágenes y extracción de texto.
- `esquemas.py`: define los modelos de datos utilizados por el sistema.

## Uso como paquete

Para poder importar correctamente estos módulos desde la aplicación, los experimentos y los notebooks, se recomienda instalar el proyecto en modo editable desde la raíz:

```bash
pip install -e .