# Prototipo de sistema tutor inteligente para D-PAL

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado centrado en el diseño, implementación y evaluación de un prototipo de sistema tutor inteligente basado en modelos de lenguaje. El sistema se enmarca en el proyecto D-PAL y tiene como objetivo apoyar la personalización del aprendizaje en actividades de lectura digital mediante evaluación automática, adaptación del andamiaje, análisis de similitud textual y visualización de resultados para alumnado y profesorado.

## Descripción general

El prototipo permite gestionar sesiones educativas adaptativas en las que el alumnado realiza tareas asociadas a distintas estrategias de lectura digital. A partir de las respuestas proporcionadas, el sistema evalúa el desempeño, actualiza el nivel competencial del estudiante y almacena los resultados para su posterior consulta.

El sistema combina distintos tipos de componentes:

* Una aplicación web desarrollada con Streamlit.
* Módulos internos para agentes, evaluación, andamiaje y detección de similitud.
* Archivos de configuración, tareas, prompts e imágenes.
* Scripts experimentales para evaluar los módulos principales.
* Notebooks utilizados para el análisis de resultados y generación de gráficas.

## Estructura del repositorio

```text
.
├── app/
│   └── Aplicación web del prototipo.
│
├── data/
│   ├── Archivos JSON con tareas educativas.
│   ├── Prompts utilizados por los modelos de lenguaje.
│   ├── Imágenes y recursos asociados a las tareas.
│   └── Datos sintéticos o de prueba utilizados en la evaluación.
│
├── experimentos/
│   ├── Scripts para evaluar el módulo de evaluación automática.
│   ├── Scripts para evaluar la generación de enunciados adaptados.
│   └── Scripts para evaluar el detector de similitud textual.
│   
│
├── notebooks/
│   └── Cuadernos Jupyter utilizados para análisis exploratorio,
│       revisión de resultados y generación de gráficos.
│
├── resultados/
│   ├── Figuras generadas para la memoria.
│   ├── Tablas de resultados.
│   └── Salidas experimentales procesadas.
│
├── src/
│   └── Código principal del sistema:
│       agentes, módulos de evaluación, detección de similitud,
│       utilidades y lógica interna del prototipo.
│
├── .streamlit/
│   └── Configuración local de Streamlit.
│
├── pyproject.toml
│   └── Definición del proyecto y dependencias gestionadas con uv.
│
├── uv.lock
│   └── Archivo de bloqueo generado por uv.
│
├── requirements.txt
│   └── Dependencias principales para instalar el proyecto con pip.
│
├── requirements_linux.txt
│   └── Dependencias utilizadas en el entorno del servidor Linux.
│
└── README.md
```

## Instalación

Se recomienda utilizar un entorno virtual de Python. El proyecto requiere Python 3.10 o superior.

### Opción 1: instalación con pip

```bash
python -m venv .venv
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

En Windows:

```bash
.venv\Scripts\activate
```

Después, instalar las dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Si se desea utilizar el entorno exacto del servidor Linux, puede instalarse:

```bash
pip install -r requirements_linux.txt
```

Finalmente, instalar el proyecto en modo editable:

```bash
pip install -e .
```

Esto permite importar los módulos de `src/` desde la aplicación, los experimentos y los notebooks.

### Opción 2: instalación con uv

En local, también puede utilizarse `uv`:

```bash
uv sync
```

Para ejecutar comandos dentro del entorno gestionado por `uv`:

```bash
uv run python archivo.py
```

o:

```bash
uv run streamlit run app/main.py
```

## Ejecución de la aplicación

Para iniciar la aplicación web:

```bash
streamlit run app/main.py
```

Si el archivo principal de la aplicación tiene otro nombre, sustituir `app/main.py` por la ruta correspondiente.

Por ejemplo:

```bash
streamlit run app/streamlit_app.py
```

## Configuración

El sistema puede requerir variables de entorno para acceder a modelos de lenguaje, APIs o rutas específicas. En ese caso, se recomienda crear un archivo `.env` en la raíz del proyecto.

Ejemplo:

```text
OPENAI_API_KEY=
MODEL_ENDPOINT=
DATABASE_PATH=
```

El archivo `.env` no debe subirse al repositorio si contiene claves privadas o credenciales.

## Datos y recursos

La carpeta `data/` contiene los recursos necesarios para ejecutar el prototipo, como tareas en formato JSON, prompts e imágenes asociadas a las actividades.

Algunos materiales pueden estar sujetos a restricciones del proyecto D-PAL. Por este motivo, en una versión pública del repositorio pueden sustituirse por ejemplos sintéticos o datos anonimizados.

En caso de que ciertos datos no estén incluidos, debe consultarse la documentación interna del proyecto o solicitar acceso a los responsables correspondientes.

## Experimentos

La carpeta `experimentos/` contiene los scripts utilizados para evaluar los módulos principales del sistema:

* Evaluación automática de respuestas abiertas.
* Generación de enunciados adaptados.
* Detección de similitud textual.
* Comparación entre modelos de lenguaje.
* Análisis de métricas y tiempos de respuesta.

Los scripts deben ejecutarse desde la raíz del repositorio para que las rutas relativas funcionen correctamente.

Ejemplo:

```bash
python experimentos/nombre_del_experimento.py
```

Si el proyecto está instalado en modo editable, también pueden importarse directamente los módulos definidos en `src/`.

## Notebooks

La carpeta `notebooks/` contiene cuadernos utilizados para análisis exploratorio, revisión de resultados y generación de figuras.

Para utilizar los notebooks, se recomienda instalar el kernel del entorno virtual:

```bash
python -m ipykernel install --user --name dpal-tutor --display-name "DPAL Tutor"
```

Después, seleccionar el kernel `DPAL Tutor` desde Jupyter o VS Code.

## Resultados

La carpeta `resultados/` recoge salidas generadas durante la evaluación experimental, incluyendo gráficos, tablas y resultados procesados.

Estos resultados se utilizan como apoyo para la memoria del Trabajo Fin de Grado y permiten reproducir o revisar los análisis realizados.

## Reproducción básica

Un flujo básico para reproducir el proyecto sería:

```bash
git clone URL_DEL_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

streamlit run app/app2.py
```

En Windows, la activación del entorno sería:

```bash
.venv\Scripts\activate
```

## Consideraciones sobre dependencias

El archivo `requirements.txt` contiene una versión general de las dependencias necesarias para ejecutar el proyecto.

El archivo `requirements_linux.txt` puede contener dependencias específicas del servidor, incluidas librerías generadas por el entorno Linux. Algunas de ellas pueden no ser necesarias o no ser compatibles con Windows, especialmente paquetes relacionados con CUDA, GPU o herramientas específicas del sistema operativo.

Si se trabaja en Windows, se recomienda empezar con:

```bash
pip install -r requirements.txt
```

y utilizar `requirements_linux.txt` solo si se desea reproducir el entorno del servidor.

## Privacidad y limitaciones

Este repositorio puede contener materiales asociados al proyecto D-PAL. Por tanto, no deben publicarse datos sensibles, credenciales, respuestas reales de estudiantes ni recursos protegidos sin autorización.

En caso de publicar una versión abierta del repositorio, se recomienda:

* Sustituir datos privados por ejemplos sintéticos.
* No incluir claves de API.
* No subir bases de datos reales.
* No incluir materiales educativos sujetos a restricciones.
* Documentar claramente qué partes del sistema no se incluyen por motivos de privacidad o propiedad intelectual.

## Autoría

Este proyecto ha sido desarrollado como parte de un Trabajo Fin de Grado del Grado en Ciencia de Datos, en el marco del proyecto D-PAL y de las prácticas realizadas en VRAIN, Universitat Politècnica de València.

## Licencia

El uso, distribución o publicación del código y de los materiales asociados debe respetar las condiciones establecidas por el proyecto D-PAL y por la Universitat Politècnica de València. En caso de que el repositorio se mantenga privado, el acceso podrá facilitarse al tribunal, tutores o responsables del proyecto bajo solicitud.
