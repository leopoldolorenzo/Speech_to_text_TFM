# backend/services/diarization.py

from pyannote.audio import Pipeline

# Ajusta esta ruta según donde tengas el config.yaml entrenado
DIARIZATION_CONFIG = "models/pyannote_speaker_diarization_ready/config.yaml"

def diarize_audio(audio_path):
    """
    Segmenta el audio en fragmentos de hablantes con pyannote.
    """
    # Cargar el pipeline
    pipeline = Pipeline.from_pretrained(DIARIZATION_CONFIG)

    # Ejecutar diarización
    diarization = pipeline(audio_path)

    # Convertir resultados a una lista de segmentos
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return segments
