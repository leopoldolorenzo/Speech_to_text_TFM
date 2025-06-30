from transformers import MarianMTModel, MarianTokenizer

# Ruta local donde tienes el modelo descargado
MODEL_PATH = "models/opus-mt-es-en"

# Carga el tokenizador y el modelo desde el disco (esto se hace solo una vez al arrancar el servidor)
print(f"🔧 Cargando modelo de traducción desde: {MODEL_PATH}")
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
translation_model = MarianMTModel.from_pretrained(MODEL_PATH)
print("✅ Modelo de traducción cargado y listo para usar.")

def translate_text_to_english(text):
    """
    Traduce un texto en español al inglés usando el modelo local Helsinki-NLP/opus-mt-es-en.
    """
    if not text or not text.strip():
        return ""

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text
