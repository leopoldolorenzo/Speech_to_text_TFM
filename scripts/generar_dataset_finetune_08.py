#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para preparar y guardar el dataset fine_tune_08 en formato Hugging Face DatasetDict.
Convierte el TSV a AudioDataset y lo guarda para su uso posterior sin errores de carga.
"""

import os
from datasets import load_dataset, Audio, DatasetDict

# === Rutas ===
TSV_PATH = "data/common_voice_es/es/fine_tune_08/dataset_100k.tsv"
OUTPUT_PATH = "data/common_voice_es/es/fine_tune_08/dataset_hf"

# === Cargar dataset desde TSV ===
print("📥 Cargando dataset desde TSV...")
dataset = load_dataset(
    "csv",
    data_files=TSV_PATH,
    delimiter="\t",
    column_names=["path", "sentence"]
)

# === Convertir columna 'path' en tipo Audio ===
print("🔊 Aplicando cast_column a 'path' como Audio...")
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

# === Forzar carga del campo 'audio' para evitar errores ===
print("🔄 Forzando lectura de audio...")
dataset = dataset.map(lambda x: x, desc="Cargando audio")

# === Separar en train / validation ===
print("✂️ Dividiendo en train/validation...")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# === Guardar dataset en disco ===
print(f"💾 Guardando dataset en: {OUTPUT_PATH}")
dataset.save_to_disk(OUTPUT_PATH)

print("✅ Dataset preparado y almacenado con éxito.")
