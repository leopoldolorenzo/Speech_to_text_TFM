from pyannote.audio import Pipeline

# Token de Hugging Face
HF_TOKEN = "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"

# Ruta local para almacenar el modelo descargado
LOCAL_DIR = "models/pyannote_speaker_diarization"

# Descargar y guardar en disco
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN, cache_dir=LOCAL_DIR)

# Confirmar que fue descargado
print("✅ Pipeline descargado en:", LOCAL_DIR)
