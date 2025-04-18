from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("models/debug_w2v2")

# Una secuencia simple de letras (a, b, c, espacio, d)
sample_ids = [1, 2, 3, 0, 4]  # debería ser: a b c _ d

decoded = processor.batch_decode([sample_ids], group_tokens=False)

print("🧪 Decodificación de ejemplo:")
print("IDs:", sample_ids)
print("Texto:", decoded[0])
