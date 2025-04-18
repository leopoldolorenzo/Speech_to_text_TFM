# scripts/verificar_modelo_finetune.py

import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Ruta absoluta al modelo
MODEL_DIR = "./models/fine_tune_01"

AUDIO_PATH = "data/eval/prueba01.wav"

print("📦 Cargando processor y modelo fine-tuneado...")
processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).eval()


# Verificar tokenizer
print("\n🔠 Tokens en el vocabulario:")
vocab = processor.tokenizer.get_vocab()
print("Total:", len(vocab), "tokens")
print("Ejemplo encode/decode ('hola mundo'):")
encoded = processor.tokenizer.encode("hola mundo")
decoded = processor.tokenizer.decode(encoded)
print(" → Codificado:", encoded)
print(" → Decodificado:", decoded)

# Cargar audio
print("\n🎧 Cargando audio de prueba:", AUDIO_PATH)
audio, _ = librosa.load(AUDIO_PATH, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Obtener logits
print("🔍 Ejecutando inferencia...")
with torch.no_grad():
    logits = model(inputs.input_values).logits

print("📐 Forma de los logits:", logits.shape)
print("🔢 Valores de ejemplo:", logits[0][0][:5])

# Decodificación greedy
print("\n🧠 Decodificando (greedy)...")
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)
print("📝 Transcripción generada:", transcription[0])

# Evaluación final
if transcription[0].strip() == "":
    print("\n❌ El modelo NO está generando texto. Posiblemente falló el entrenamiento o el vocabulario.")
else:
    print("\n✅ El modelo genera texto correctamente. El tokenizer y el modelo parecen estar alineados.")
