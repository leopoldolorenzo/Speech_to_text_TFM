from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"
CACHE_DIR = "models/opus-mt-es-en"

print("🔧 Descargando modelo Helsinki-NLP/opus-mt-es-en...")

# Descarga y almacena localmente el tokenizador y el modelo
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = MarianMTModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

print(f"✅ Modelo descargado y almacenado localmente en: {CACHE_DIR}")
