import sqlite3
import json
import re
from pathlib import Path

from sistema.config import DB_PATH, TASKS_PATH

DB_NAME = str(DB_PATH)


def _migrate_results(conn):
    """Añade identidad por asignatura/intento sin perder resultados anteriores."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(resultados_sesion)")}
    additions = {
        "detailed_feedback": "TEXT",
        "subject_id": "TEXT",
        "attempt_id": "TEXT",
        "student_rating": "INTEGER",
        "scenario_id": "TEXT",
    }
    if set(additions) - columns and conn.execute(
        "SELECT COUNT(*) FROM resultados_sesion"
    ).fetchone()[0]:
        backup_path = Path(DB_NAME + ".before_scenarios.bak")
        if not backup_path.exists():
            with sqlite3.connect(str(backup_path)) as backup:
                conn.backup(backup)
    for name, sql_type in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE resultados_sesion ADD COLUMN {name} {sql_type}")

    legacy_rows = conn.execute(
        "SELECT id, task_history FROM resultados_sesion WHERE subject_id IS NULL"
    ).fetchall()
    if legacy_rows:
        with open(TASKS_PATH, encoding="utf-8-sig") as file:
            curriculum = json.load(file)
        subjects_by_task = {}
        for subject, scenarios in curriculum.items():
            for sessions in scenarios.values():
                for tasks in sessions.values():
                    for task in tasks:
                        subjects_by_task.setdefault(task["task_id"], set()).add(subject)
        for result_id, raw_history in legacy_rows:
            subjects = set()
            try:
                history = json.loads(raw_history or "[]")
            except (TypeError, json.JSONDecodeError):
                history = []
            for task in history if isinstance(history, list) else []:
                if isinstance(task, dict) and isinstance(task.get("task_id"), str):
                    task_id = re.sub(r"^(retirada:)?(cien|lit|arte)_s", r"\1\2_e1_s",
                                     task.get("task_id", ""))
                    subjects.update(subjects_by_task.get(task_id, set()))
            subject = next(iter(subjects)) if len(subjects) == 1 else "sin_identificar"
            conn.execute(
                "UPDATE resultados_sesion SET subject_id = ?, attempt_id = ? WHERE id = ?",
                (subject, f"legacy-{result_id}", result_id),
            )
    legacy_scenarios = conn.execute(
        "SELECT id, subject_id, task_history FROM resultados_sesion WHERE scenario_id IS NULL"
    ).fetchall()
    for result_id, subject_id, raw_history in legacy_scenarios:
        try:
            history = json.loads(raw_history or "[]")
        except (TypeError, json.JSONDecodeError):
            history = None
        changed = False
        if isinstance(history, list):
            for task in history:
                if isinstance(task, dict) and isinstance(task.get("task_id"), str):
                    old_id = task["task_id"]
                    task["task_id"] = re.sub(r"^(retirada:)?(cien|lit|arte)_s", r"\1\2_e1_s", old_id)
                    changed |= task["task_id"] != old_id
        scenario = "escenario1" if subject_id in ("ciencias", "literatura", "historia") else "sin_identificar"
        conn.execute(
            "UPDATE resultados_sesion SET scenario_id = ?, task_history = ? WHERE id = ?",
            (scenario, json.dumps(history, ensure_ascii=False) if changed else raw_history, result_id),
        )
    index_columns = [row[2] for row in conn.execute("PRAGMA index_info(resultados_por_intento)")]
    if index_columns and index_columns != ["alumno_id", "subject_id", "scenario_id", "session_id", "attempt_id"]:
        conn.execute("DROP INDEX resultados_por_intento")
    conn.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS resultados_por_intento
        ON resultados_sesion(alumno_id, subject_id, scenario_id, session_id, attempt_id)
    ''')

def init_db():
    """Crea las tablas y añade datos de prueba si la base de datos está vacía."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. Tabla de Usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT NOT NULL
        )
    ''')

    # 2. Tabla Relación Profesor -> Alumno
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alumnos_profesor (
            profesor_id INTEGER,
            alumno_id INTEGER,
            FOREIGN KEY(profesor_id) REFERENCES usuarios(id),
            FOREIGN KEY(alumno_id) REFERENCES usuarios(id),
            PRIMARY KEY (profesor_id, alumno_id)
        )
    ''')

    # 3. Tabla Resultados de Sesiones (Actualizada con detailed_feedback)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_sesion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alumno_id INTEGER,
            session_id TEXT,
            competency_levels TEXT,
            task_history TEXT,
            detailed_feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(alumno_id) REFERENCES usuarios(id)
        )
    ''')
    
    _migrate_results(conn)

    # --- DATOS DE PRUEBA ---
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO usuarios (username, password, role, name) VALUES (?, ?, ?, ?)",
            ('profe_ana', '1234', 'profesor', 'Ana López (Profesora)')
        )
        prof_id = cursor.lastrowid

        alumnos_demo = [
            ('juan', '1234', 'alumno', 'Juan Pérez'),
            ('maria', '1234', 'alumno', 'María Gómez'),
            ('luis', '1234', 'alumno', 'Luis Martínez')
        ]
        
        for al in alumnos_demo:
            cursor.execute("INSERT INTO usuarios (username, password, role, name) VALUES (?, ?, ?, ?)", al)
            al_id = cursor.lastrowid
            cursor.execute("INSERT INTO alumnos_profesor (profesor_id, alumno_id) VALUES (?, ?)", (prof_id, al_id))

    conn.commit()
    conn.close()

# --- FUNCIONES DE ACCESO A DATOS ---

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM usuarios WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def save_student_results(alumno_id, subject_id, scenario_id, session_id, attempt_id,
                         competency_levels, task_history, detailed_feedback=None):
    if not all((subject_id, scenario_id, session_id, attempt_id)):
        raise ValueError("El resultado debe identificar asignatura, escenario, sesión e intento.")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    levels_json = json.dumps(competency_levels)
    history_json = json.dumps([h.model_dump() if hasattr(h, 'model_dump') else h for h in task_history])
    feedback_json = json.dumps(detailed_feedback) if detailed_feedback else "{}"
    
    cursor.execute('''
        INSERT INTO resultados_sesion
            (alumno_id, subject_id, scenario_id, session_id, attempt_id,
             competency_levels, task_history, detailed_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alumno_id, subject_id, scenario_id, session_id, attempt_id) DO UPDATE SET
            competency_levels = excluded.competency_levels,
            task_history = excluded.task_history,
            detailed_feedback = excluded.detailed_feedback
    ''', (alumno_id, subject_id, scenario_id, session_id, attempt_id,
          levels_json, history_json, feedback_json))
        
    conn.commit()
    conn.close()


def save_student_rating(alumno_id, attempt_id, rating):
    if isinstance(rating, bool) or not isinstance(rating, int) or rating not in range(5):
        raise ValueError("La valoración debe ser un entero entre 0 y 4.")
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "UPDATE resultados_sesion SET student_rating = ? WHERE alumno_id = ? AND attempt_id = ?",
            (rating, alumno_id, attempt_id),
        )

def get_professor_students_data(profesor_id):
    """Devuelve los datos de todos los alumnos asignados a un profesor."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.id as alumno_id, u.name as alumno_name, rs.subject_id, rs.scenario_id, rs.session_id,
               rs.attempt_id, rs.student_rating, rs.competency_levels,
               rs.task_history, rs.detailed_feedback, rs.timestamp
        FROM alumnos_profesor ap
        JOIN usuarios u ON ap.alumno_id = u.id
        LEFT JOIN resultados_sesion rs ON u.id = rs.alumno_id
        WHERE ap.profesor_id = ?
        ORDER BY rs.timestamp DESC, rs.id DESC
    ''', (profesor_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "alumno_id": r["alumno_id"],
            "alumno_name": r["alumno_name"],
            "session_id": r["session_id"],
            "subject_id": r["subject_id"],
            "scenario_id": r["scenario_id"],
            "attempt_id": r["attempt_id"],
            "student_rating": r["student_rating"],
            "competency_levels": json.loads(r["competency_levels"]) if r["competency_levels"] else None,
            "task_history": json.loads(r["task_history"]) if r["task_history"] else None,
            "detailed_feedback": json.loads(r["detailed_feedback"]) if r["detailed_feedback"] else {},
            "timestamp": r["timestamp"]
        })
    return results

init_db()
