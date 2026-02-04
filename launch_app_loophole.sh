#!/bin/bash

# ================= CONFIGURACIÓN =================
# Elige el nombre para tu URL (ej: https://proyecto-psicologia.loophole.site)
HOSTNAME_DESEADO="proyecto-psicologia" 

# Tu puerto (DEBE ser el mismo que uses en tu código Python)
PORT=8501

# Carpeta de tu entorno virtual
VENV_DIR=".venv"
# =================================================

echo "🔧 Iniciando despliegue de la aplicación..."

# 1. Limpieza de procesos antiguos en este puerto para evitar conflictos
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  kill -9 $PID
fi
pkill -f "./tools/loophole/loophole" || true


# 2. Activar entorno virtual
if [ -d "$VENV_DIR" ]; then
    source $VENV_DIR/bin/activate
fi


# 3. Lanzar la aplicación Python en segundo plano con nohup
# El uso de -u es para que los logs se escriban al instante
echo "🚀 Lanzando aplicación Python..."
nohup streamlit run app.py \ --server.port 8501 \ --server.address 127.0.0.1 \ --server.headless true \ --server.enableCORS false \ --server.enableXsrfProtection false > app_output.log 2>&1 &

# Esperar a que la app esté lista localmente antes de abrir el túnel
echo "⏳ Esperando a que la web app arranque en el puerto $PORT..."
while ! nc -z localhost $PORT; do   
  sleep 1
done
echo "✅ App online."


# 4. Lanzar el túnel de Loophole para abrir la web al exterior
# Asume que el binario 'loophole' está en la misma carpeta
if [ -f "./tools/loophole/loophole" ]; then
    nohup ./tools/loophole/loophole http $PORT --hostname $HOSTNAME_DESEADO > tunnel_output.log 2>&1 &
    echo "========================================================"
    echo "🎉 URL PÚBLICA: https://$HOSTNAME_DESEADO.loophole.site"
    echo "========================================================"
else
    echo "❌ Error: No se encuentra el ejecutable 'loophole' en la carpeta."
fi
