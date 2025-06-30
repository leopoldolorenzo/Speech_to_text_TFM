#!/bin/bash

# Activa el entorno Conda
echo "Activando entorno Conda: asr_env"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env

# Arranca el backend con FastAPI (en background)
echo "Levantando backend..."
uvicorn backend.main:app --reload &

# (Opcional) Arranca LanguageTool si lo necesitas
# echo "Levantando LanguageTool..."
# cd tools/LanguageTool-6.6
# java -jar languagetool-server.jar --port 8081 &
# cd ../../

# Arranca el frontend (en background)
echo "Levantando frontend..."
cd frontend
yarn dev &

echo "✅ Todos los servicios están levantados."
