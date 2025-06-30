# scripts/exportar_tokenizer_base41.py

from transformers import Wav2Vec2Processor

# Modelo base preentrenado de Hugging Face
modelo_base = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

# Directorio donde se guardará el tokenizer extraído
ruta_destino = "models/tokenizer_v2_base41"

print(f"📥 Cargando processor desde: {modelo_base}")
processor = Wav2Vec2Processor.from_pretrained(modelo_base)

print(f"💾 Guardando tokenizer y preprocessor en: {ruta_destino}")
processor.save_pretrained(ruta_destino)

print("✅ Tokenizer exportado correctamente.")
