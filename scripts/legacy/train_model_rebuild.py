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

# === Configuración ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
ORIGINAL_DATASET_PATH = "data/common_voice_subset"
PROCESSED_DATASET_PATH = "data/common_voice_subset_processed"
TOKENIZER_DIR = "models/tokenizer_v1"
OUTPUT_DIR = "training/spanish_w2v2_rebuild"
FINAL_MODEL_DIR = "models/spanish_w2v2_rebuild"

print("[✅] Entorno listo.")
dataset = load_from_disk(ORIGINAL_DATASET_PATH)
processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.synchronize()

# === Limpieza textual
def clean_text(text):
    return re.sub(r"[^a-záéíóúñü ]+", "", text.lower())

# === Preprocesamiento
def prepare(batch):
    text = clean_text(batch["sentence"])
    if len(text.strip()) == 0:
        print(f"⚠️ Transcripción vacía: {batch['sentence']}")
    inputs = processor(
        batch["audio"]["array"],
        sampling_rate=16000,
        text=text,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return {
        "input_values": inputs.input_values[0],
        "attention_mask": inputs.attention_mask[0],
        "labels": inputs["labels"][0]
    }

print("[🔄] Preprocesando dataset...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names)

# Verificación rápida
print("🔍 Ejemplo post-map:", dataset["train"][0])

# Opcional: filtrar ejemplos sin etiquetas
dataset = dataset.filter(lambda x: len(x["labels"]) > 0, num_proc=4)

# Guardar dataset procesado en nueva ruta
if os.path.exists(PROCESSED_DATASET_PATH):
    shutil.rmtree(PROCESSED_DATASET_PATH)
dataset.save_to_disk(PROCESSED_DATASET_PATH)

# === Data Collator y Métrica
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# === Cargar modelo base con vocab personalizado
print("[🧠] Cargando modelo base con vocab personalizado...")
config = Wav2Vec2Config.from_pretrained(BASE_MODEL)
config.vocab_size = len(processor.tokenizer)

model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    config=config,
    ignore_mismatched_sizes=True,
)

model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

# === Verificación de batch antes de entrenar
print("🧪 Verificando batch de entrenamiento...")
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)
batch = next(iter(loader))
print("🔢 Labels en batch:", batch["labels"])
print("📐 Formas:", {k: v.shape for k, v in batch.items()})

# === Argumentos de entrenamiento
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
    fp16=True,
    logging_dir="logs_rebuild",
    logging_steps=50,
    logging_first_step=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    log_level="info",
)

# === Inicializar Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor  # Usa el processor completo
)

# === Reanudación automática si hay checkpoint
latest_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = os.path.join(
            OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"[⏸️] Reanudando desde: {latest_checkpoint}")
    else:
        print("[🚀] No hay checkpoints previos. Comenzando de cero.")
else:
    print("[🚀] Carpeta nueva. Iniciando desde cero.")

# === Entrenamiento
print("[🔥] Entrenando modelo...")
trainer.train(resume_from_checkpoint=latest_checkpoint)

# === Guardado final
print("[💾] Guardando modelo entrenado...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

if os.path.exists(FINAL_MODEL_DIR):
    shutil.rmtree(FINAL_MODEL_DIR)
shutil.copytree(OUTPUT_DIR, FINAL_MODEL_DIR)
print(f"[📦] Modelo final disponible en: {FINAL_MODEL_DIR}")
print("[✅] Entrenamiento finalizado.")
