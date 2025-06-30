#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verificar_paths_tsv.py

Verifica que todos los archivos .wav referenciados en dataset_100k.tsv existan.

Uso:
    python verificar_paths_tsv.py --tsv data/common_voice_es/es/fine_tune_08/dataset_100k.tsv
"""

import csv
import sys
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tsv", required=True, type=Path, help="Ruta al archivo TSV")
args = parser.parse_args()

missing_files = []
total_lines = 0

with open(args.tsv, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, row in enumerate(reader, 1):
        total_lines += 1
        if len(row) != 2:
            print(f"⚠️ Línea {i} malformada: {row}")
            continue
        path, _ = row
        if not Path(path).is_file():
            missing_files.append((i, path))

print(f"\n📊 Resultados de la verificación de paths:")
print(f"• Total líneas procesadas: {total_lines}")
print(f"• Archivos encontrados:    {total_lines - len(missing_files)}")
print(f"• Archivos faltantes:      {len(missing_files)}")

if missing_files:
    print("\n❌ Archivos faltantes encontrados:")
    for i, path in missing_files[:10]:  # Mostrar hasta 10 ejemplos
        print(f"  Línea {i}: {path}")
    print("\n❗ Regenera el TSV o verifica los archivos en clips_wav/")
else:
    print("\n✅ Todos los archivos de audio existen.")