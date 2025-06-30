import os
import shutil
from pathlib import Path
from backend.services.audio_preprocessing import preprocess_audio
from backend.services.diarization import diarize_audio
from backend.services.asr_transcription import transcribe_segment
from backend.services.postprocessing import simple_postprocess
from backend.services.translation import translate_text_to_english  # Importamos la función de traducción

def run_pipeline(input_path):
    print(f"🔧 Procesando archivo: {input_path}")

    # 1. Mover archivo temporal a carpeta tmp
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    input_name = os.path.basename(input_path)
    tmp_path = os.path.join(tmp_dir, input_name)

    if not os.path.samefile(input_path, tmp_path):
        shutil.move(input_path, tmp_path)
        print(f"🔧 Archivo movido a: {tmp_path}")

    # 2. Preprocesar audio
    preprocessed_name = f"preprocessed_{Path(tmp_path).stem}.wav"
    preprocessed_path = os.path.join("data/diarizacion", preprocessed_name)
    preprocess_audio(tmp_path, preprocessed_path)

    # 3. Diarización
    segments = diarize_audio(preprocessed_path)

    # 4. Añadir la ruta del archivo procesado a cada segmento
    for segment in segments:
        segment["path"] = preprocessed_path

    # 5. Transcripción + Postprocesado + Traducción
    results = []
    for segment in segments:
        raw_text = transcribe_segment(segment)
        grammar_corrected = simple_postprocess(raw_text)
        translation = translate_text_to_english(grammar_corrected)  # Traducción al inglés

        results.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "raw_text": raw_text,
            "grammar_corrected": grammar_corrected,
            "translation": translation
        })

    # 6. URL del audio procesado
    audio_url = f"http://127.0.0.1:8000/audios/{preprocessed_name}"

    # 7. Borrar el archivo temporal
    try:
        os.remove(tmp_path)
        print(f"🗑️ Archivo temporal eliminado: {tmp_path}")
    except Exception as e:
        print(f"⚠️ No se pudo eliminar el archivo temporal: {e}")

    return {
        "transcription": results,
        "audio_url": audio_url
    }
