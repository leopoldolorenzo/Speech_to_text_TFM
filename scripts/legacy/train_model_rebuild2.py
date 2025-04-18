import os
import re
import torch
import numpy as np
import evaluate
import shutil
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from data_collator_ctc import DataCollatorCTCWithPadding
from torch.utils.data import DataLoader

# === Configuración de rutas y modelo base === 
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"   
ORIGINAL_DATASET_PATH = "data/common_voice_subset"
PROCESSED_DATASET_PATH = "data/common_voice_subset_processed"
TOKENIZER_DIR = "models/tokenizer_v1"
OUTPUT_DIR = "training/spanish_w2v2_rebuild"
FINAL_MODEL_DIR = "models/spanish_w2v2_rebuild"

print("[✅] Entorno listo.")

# === Verificar existencia del dataset y su estructura ===
assert os.path.exists(ORIGINAL_DATASET_PATH), f"No se encontró el dataset en {ORIGINAL_DATASET_PATH}"
dataset = load_from_disk(ORIGINAL_DATASET_PATH)
assert "train" in dataset and "test" in dataset, "El dataset debe tener splits 'train' y 'test'"
assert all("audio" in x and "sentence" in x for x in dataset["train"].select(range(3))), "Faltan campos requeridos"

# === Cargar processor (tokenizer + feature extractor) ===
try:
    processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
except Exception as e:
    raise ValueError(f"No se pudo cargar el processor desde {TOKENIZER_DIR}: {e}")

torch.cuda.empty_cache()

# === Limpieza de texto ===
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-záéíóúñü ]+", "", text)

# === Preprocesamiento robusto ===
def prepare(batch):
    try:
        text = clean_text(batch.get("sentence", ""))
        if not text.strip():
            return None

        # Procesar audio
        audio_inputs = processor(batch["audio"]["array"], sampling_rate=16000)

        # Generar labels con el tokenizer (no usar text= en processor)
        with processor.as_target_processor():
            labels = processor.tokenizer(text).input_ids

        if not labels:
            print(f"⚠️ Etiquetas vacías para texto: '{text}'")
            return None

        return {
            "input_values": audio_inputs["input_values"][0],
            "attention_mask": audio_inputs["attention_mask"][0],
            "labels": labels  # lista; el collator se encarga del padding
        }

    except Exception as e:
        print(f"⚠️ Error procesando batch: {e}")
        return None

# === Aplicar preprocesamiento ===
print("[🔄] Preprocesando dataset...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names)
dataset = dataset.filter(lambda x: x is not None and len(x["labels"]) > 0)

# === Guardar dataset preprocesado ===
if os.path.exists(PROCESSED_DATASET_PATH):
    print(f"⚠️ Eliminando dataset procesado previo: {PROCESSED_DATASET_PATH}")
    shutil.rmtree(PROCESSED_DATASET_PATH)
dataset.save_to_disk(PROCESSED_DATASET_PATH)

# === Collator y métrica ===
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    try:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}
    except Exception as e:
        print(f"⚠️ Error en compute_metrics: {e}")
        return {"wer": float("inf")}

# === Cargar modelo base con vocab actualizado ===
print("[🧠] Cargando modelo base con vocab personalizado...")
vocab_size = len(processor.tokenizer)
assert 0 < vocab_size < 1000, f"Tamaño de vocabulario inválido: {vocab_size}"

config = Wav2Vec2Config.from_pretrained(BASE_MODEL)
config.vocab_size = vocab_size

model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    config=config,
    ignore_mismatched_sizes=True,
)

model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# === Verificación rápida de batch ===
print("🧪 Verificando batch de entrenamiento...")
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)
try:
    batch = next(iter(loader))
    print("🔢 Labels en batch:", batch["labels"])
    print("📐 Formas:", {k: v.shape for k, v in batch.items()})
except StopIteration:
    raise ValueError("❌ El conjunto de datos de entrenamiento está vacío")

# === Argumentos del entrenamiento ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
    learning_rate=3e-4,
    num_train_epochs=5,
    warmup_steps=200,
    fp16=torch.cuda.is_available(),
    logging_dir="logs_rebuild",
    logging_steps=10,  # menor para debug
    logging_first_step=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    log_level="info",
)

# === Inicializar Trainer ===
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor
)

# === Reanudar desde último checkpoint si existe ===
latest_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
    if checkpoints:
        latest_checkpoint = os.path.join(
            OUTPUT_DIR,
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"[⏸️] Reanudando desde: {latest_checkpoint}")
    else:
        print("[🚀] Iniciando desde cero.")
else:
    print("[🚀] Carpeta nueva. Iniciando desde cero.")

# === Entrenamiento ===
print("[🔥] Entrenando modelo...")
try:
    trainer.train(resume_from_checkpoint=latest_checkpoint)
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    raise

# === Guardar modelo final ===
print("[💾] Guardando modelo entrenado...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"[📦] Modelo final disponible en: {FINAL_MODEL_DIR}")
print("[✅] Entrenamiento finalizado.")
