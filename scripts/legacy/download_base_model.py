from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
local_path = "models/base"

# Descarga y guarda el modelo y tokenizer en local
processor = Wav2Vec2Processor.from_pretrained(model_name)
processor.save_pretrained(local_path)

model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.save_pretrained(local_path)

# Guarda el nombre original como referencia (opcional)
with open(f"{local_path}/huggingface_model.txt", "w") as f:
    f.write(model_name)

print("✅ Modelo base guardado en models/base")
print("✅ Tokenizer guardado en models/base")
print("✅ Nombre del modelo original guardado en models/base/huggingface_model.txt")
