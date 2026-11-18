from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_PATH = DATA_DIR / "prompts/prompts.yaml"
JSON_DIR = DATA_DIR / "json"
TASKS_PATH = JSON_DIR / "tareas2.json"
INITIAL_PATH = JSON_DIR / "test_inicial.json"
ENUNCIADOS_PATH = JSON_DIR / "tareas_enunciados.json"
PLAGIO_PATH = JSON_DIR / "respuestas_plagio.json"
EVALUATION_PATH = JSON_DIR / "evaluacion_1.json"
IMAGES_DIR = DATA_DIR / "img"
PRE_DIR = DATA_DIR / "img_pre"
DATABASE_DIR = DATA_DIR / "db"
DB_PATH = DATABASE_DIR / "tutor_ia.db"
RESULTS_DIR = PROJECT_ROOT / "resultados"
PLAGIARISM_THRESHOLD = 0.25
PLAGIARISM_MIN_WORDS = 15


def resolve_project_path(path_str: str) -> Path:
    path_str = path_str.replace("\\", "/")

    if path_str.startswith("./"):
        path_str = path_str[2:]

    path = Path(path_str)

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()
