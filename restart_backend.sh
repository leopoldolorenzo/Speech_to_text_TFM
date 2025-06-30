#!/bin/bash

echo "🔧 Deteniendo procesos antiguos de uvicorn..."
pkill -f uvicorn

echo "🚀 Reiniciando backend con uvicorn..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env

uvicorn backend.main:app --reload
