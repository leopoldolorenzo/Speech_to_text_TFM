#!/bin/bash

echo "🔧 Matando procesos viejos de uvicorn y vite..."
pkill -f uvicorn
pkill -f vite

echo "🚀 Reiniciando entorno..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env

echo "🟢 Levantando backend..."
uvicorn backend.main:app --reload &

echo "🟢 Levantando frontend..."
cd frontend
yarn dev &

echo "✅ Todos los servicios están levantados."
