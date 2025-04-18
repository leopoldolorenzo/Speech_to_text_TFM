# build_clean_processor.py
# ----------------------------------------
# Genera un processor completo (tokenizer + feature extractor) 
# desde un vocabulario previamente generado y lo guarda 
# en una carpeta lista para usar en entrenamiento.
# ----------------------------------------

import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# === CONFIGURACIÓN ===
VOCAB_PATH = "models/tokenizer_v1_nospecial/vocab.json"  # Ruta al vocab.json existente
OUTPUT_DIR = "models/tokenizer_v1_nospecial"             # Carpeta donde guardar el processor

# === Crear tokenizer desde vocab.json ===
tokenizer = Wav2Vec2CTCTokenizer(
    VOCAB_PATH,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

# === Crear feature extractor genérico para audio 16kHz ===
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

# === Combinar tokenizer y extractor en un processor
processor = Wav2Vec2Processor(
    tokenizer=tokenizer,
    feature_extractor=feature_extractor
)

# === Guardar processor completo (crea preprocessor_config.json)
os.makedirs(OUTPUT_DIR, exist_ok=True)
processor.save_pretrained(OUTPUT_DIR)

print(f"[✅] Processor limpio guardado en: {OUTPUT_DIR}")
