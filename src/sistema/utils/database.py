import sqlite3
import json

DB_NAME = "tutor_ia.db"

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
    
    # Intentamos añadir la columna detailed_feedback por si la tabla ya existía de antes
    try:
        cursor.execute("ALTER TABLE resultados_sesion ADD COLUMN detailed_feedback TEXT")
    except sqlite3.OperationalError:
        pass # La columna ya existe

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

def save_student_results(alumno_id, session_id, competency_levels, task_history, detailed_feedback=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    levels_json = json.dumps(competency_levels)
    history_json = json.dumps([h.dict() if hasattr(h, 'dict') else h for h in task_history])
    feedback_json = json.dumps(detailed_feedback) if detailed_feedback else "{}"
    
    cursor.execute("SELECT id FROM resultados_sesion WHERE alumno_id = ? AND session_id = ?", (alumno_id, session_id))
    row = cursor.fetchone()
    
    if row:
        cursor.execute('''
            UPDATE resultados_sesion 
            SET competency_levels = ?, task_history = ?, detailed_feedback = ?, timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (levels_json, history_json, feedback_json, row[0]))
    else:
        cursor.execute('''
            INSERT INTO resultados_sesion (alumno_id, session_id, competency_levels, task_history, detailed_feedback)
            VALUES (?, ?, ?, ?, ?)
        ''', (alumno_id, session_id, levels_json, history_json, feedback_json))
        
    conn.commit()
    conn.close()

def get_professor_students_data(profesor_id):
    """Devuelve los datos de todos los alumnos asignados a un profesor."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.id as alumno_id, u.name as alumno_name, rs.session_id, rs.competency_levels, rs.task_history, rs.detailed_feedback, rs.timestamp
        FROM alumnos_profesor ap
        JOIN usuarios u ON ap.alumno_id = u.id
        LEFT JOIN resultados_sesion rs ON u.id = rs.alumno_id
        WHERE ap.profesor_id = ?
    ''', (profesor_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "alumno_id": r["alumno_id"],
            "alumno_name": r["alumno_name"],
            "session_id": r["session_id"],
            "competency_levels": json.loads(r["competency_levels"]) if r["competency_levels"] else None,
            "task_history": json.loads(r["task_history"]) if r["task_history"] else None,
            "detailed_feedback": json.loads(r["detailed_feedback"]) if r["detailed_feedback"] else {},
            "timestamp": r["timestamp"]
        })
    return results

init_db()