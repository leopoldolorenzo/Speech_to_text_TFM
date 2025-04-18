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

# === PARÁMETROS OPCIONALES ===
ENABLE_PARALLEL_MAP = False   # Si True, usa num_proc=4 en dataset.map(...)
ENABLE_ENCODER_FREEZE = True  # Si True, se congela el feature encoder
ENABLE_CER_METRIC = True      # Si True, se calcula CER además de WER
KEEP_PUNCTUATION = False      # Si True, se permitirá puntuación en la función clean_text

# === CONFIGURACIÓN DE RUTAS Y MODELO BASE ===
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
ORIGINAL_DATASET_PATH = "data/common_voice_subset"
PROCESSED_DATASET_PATH = "data/common_voice_subset_processed"
TOKENIZER_DIR = "models/tokenizer_v1"
OUTPUT_DIR = "training/spanish_w2v2_rebuild"
FINAL_MODEL_DIR = "models/spanish_w2v2_rebuild"

print("[✅] Entorno listo.")

# === 1. Verificar dataset y estructura ===
assert os.path.exists(ORIGINAL_DATASET_PATH), f"No se encontró el dataset en {ORIGINAL_DATASET_PATH}"
dataset = load_from_disk(ORIGINAL_DATASET_PATH)
assert "train" in dataset and "test" in dataset, "El dataset debe tener splits 'train' y 'test'"
assert all("audio" in x and "sentence" in x for x in dataset["train"].select(range(3))), "Faltan campos requeridos en el dataset"

# === 2. Cargar processor (tokenizer + feature extractor) ===
try:
    processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_DIR)
except Exception as e:
    raise ValueError(f"No se pudo cargar el processor desde {TOKENIZER_DIR}: {e}")

# === 3. Liberar memoria GPU previa (buen hábito) ===
torch.cuda.empty_cache()

# === 4. Función de limpieza de texto (básica para español) ===
# Si KEEP_PUNCTUATION es True, permitimos signos de puntuación (.!?), por ejemplo.
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    if KEEP_PUNCTUATION:
        # Permitimos .!? además de letras acentuadas
        text = re.sub(r"[^a-záéíóúñü .!\?]", "", text)
    else:
        # Mantiene sólo letras latinas, acentos, espacios
        text = re.sub(r"[^a-záéíóúñü ]+", "", text)
    return text

# === 5. Preprocesamiento robusto para cada ejemplo del dataset ===
def prepare(batch):
    try:
        # Limpieza básica del texto
        text = clean_text(batch.get("sentence", ""))
        if not text.strip():
            return None

        # Procesar audio a 16kHz y obtener input_values + attention_mask
        audio_inputs = processor(batch["audio"]["array"], sampling_rate=16000)

        # Generar labels con tokenizer
        with processor.as_target_processor():
            labels = processor.tokenizer(text).input_ids

        if not labels:
            print(f"⚠️ Etiquetas vacías para texto: '{text}'")
            return None

        return {
            "input_values": audio_inputs["input_values"][0],
            "attention_mask": audio_inputs["attention_mask"][0],
            "labels": labels
        }

    except Exception as e:
        print(f"⚠️ Error procesando ejemplo: {e}")
        return None

print("[🔄] Preprocesando dataset...")

# === 6. Aplicar preprocesamiento y filtrar ejemplos inválidos ===
map_kwargs = {
    "remove_columns": dataset["train"].column_names
}
if ENABLE_PARALLEL_MAP:
    # Si confiás en que no hay audios corruptos, podés activar esto para
    # aprovechar múltiples núcleos y acelerar el .map()
    map_kwargs["num_proc"] = 4

dataset = dataset.map(prepare, **map_kwargs)
dataset = dataset.filter(lambda x: x is not None and len(x["labels"]) > 0)

# === 7. Guardar dataset procesado ===
if os.path.exists(PROCESSED_DATASET_PATH):
    print(f"⚠️ Eliminando dataset procesado previo: {PROCESSED_DATASET_PATH}")
    shutil.rmtree(PROCESSED_DATASET_PATH)
dataset.save_to_disk(PROCESSED_DATASET_PATH)

# === 8. Preparar collator y métricas de evaluación (WER/CER) ===
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer") if ENABLE_CER_METRIC else None

# === 9. Función para calcular métricas durante la validación (WER y opcionalmente CER) ===
def compute_metrics(pred):
    try:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        results = {}
        # WER siempre
        results["wer"] = wer_metric.compute(predictions=pred_str, references=label_str)

        # CER opcional
        if cer_metric is not None:
            results["cer"] = cer_metric.compute(predictions=pred_str, references=label_str)

        return results
    except Exception as e:
        print(f"⚠️ Error en compute_metrics: {e}")
        fallback = {"wer": float("inf")}
        if cer_metric is not None:
            fallback["cer"] = float("inf")
        return fallback

print("[🧠] Cargando modelo base con vocab personalizado...")

# === 10. Ajustar config del modelo para el nuevo vocab_size ===
vocab_size = len(processor.tokenizer)
assert 0 < vocab_size < 2000, f"Tamaño de vocabulario inválido: {vocab_size}"  # Subimos a 2000 por si

config = Wav2Vec2Config.from_pretrained(BASE_MODEL)
config.vocab_size = vocab_size

# === 11. Cargar modelo base con config actualizada ===
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    config=config,
    ignore_mismatched_sizes=True,
)

# === 12. Opciones de fine-tuning: congelar encoder y usar gradient checkpoint ===
if ENABLE_ENCODER_FREEZE:
    model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

print("🧪 Verificando batch de entrenamiento...")

# === 13. Validación de batch antes de entrenar ===
loader = DataLoader(dataset["train"], batch_size=2, collate_fn=data_collator)
try:
    batch = next(iter(loader))
    print("🔢 Labels en batch:", batch["labels"])
    print("📐 Formas:", {k: v.shape for k, v in batch.items()})
except StopIteration:
    raise ValueError("❌ El conjunto de datos de entrenamiento está vacío")

# === 14. Configuración de TrainingArguments ===
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
    logging_steps=10,               
    logging_first_step=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    # Seleccionamos el mejor modelo según WER (menor=mejor)
    metric_for_best_model="wer",
    greater_is_better=False,
    log_level="info",
)

# === 15. Inicializar Trainer ===
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor
)

# === 16. Reanudar desde último checkpoint si existe ===
latest_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
    ]
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

# === 17. Lanzar entrenamiento ===
print("[🔥] Entrenando modelo...")
try:
    trainer.train(resume_from_checkpoint=latest_checkpoint)
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    raise

# === 18. Guardado final del modelo entrenado ===
print("[💾] Guardando modelo entrenado...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)

print(f"[📦] Modelo final disponible en: {FINAL_MODEL_DIR}")
print("[✅] Entrenamiento finalizado.")
