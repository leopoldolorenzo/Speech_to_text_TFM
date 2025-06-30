from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# Ruta donde están los archivos del tokenizer
TOKENIZER_DIR = "models/fine_tune_01"

# Ruta donde vas a guardar el processor completo
OUTPUT_DIR = "models/fine_tune_01"

# Cargar tokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_DIR)

# Crear feature extractor manualmente
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

# Combinar ambos en un processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Guardar en disco (sobreescribe preprocessor_config.json con uno válido)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ Processor reconstruido y guardado en: {OUTPUT_DIR}")
