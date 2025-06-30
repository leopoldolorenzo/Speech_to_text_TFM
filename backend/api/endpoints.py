# backend/api/endpoints.py

from fastapi import APIRouter, UploadFile, File
import os
import shutil
from backend.services.pipeline_runner import run_pipeline

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Crear carpeta tmp si no existe
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Guardar el archivo temporal
    temp_path = os.path.join(tmp_dir, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ejecutar el pipeline
    transcription_data = run_pipeline(temp_path)

    # (Opcional) eliminar el archivo temporal después de procesar
    # os.remove(temp_path)

    return transcription_data
