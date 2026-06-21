
---

## `experimentos/README.md`

```md
# Carpeta `experimentos`

Esta carpeta contiene los scripts utilizados para evaluar experimentalmente los módulos principales del prototipo.

Los experimentos permiten comparar modelos, analizar métricas, generar resultados intermedios y producir archivos CSV que posteriormente se utilizan para elaborar gráficas y tablas de la memoria del TFG.

## Experimentos incluidos

- Evaluación del módulo de evaluación automática de respuestas.
- Evaluación del generador de enunciados adaptados.
- Evaluación semántica mediante embeddings.
- Evaluación con LLM como juez externo.
- Evaluación del detector de similitud textual sobre datos sintéticos.
- Evaluación del detector de similitud textual sobre el corpus Clough & Stevenson.

## Ejecución

Los scripts deben ejecutarse desde la raíz del proyecto. Por ejemplo:

```bash
python -m experimentos.evaluar_plagio_sintetico