# scripts/generate_val_dataset.py

import os
import pandas as pd

EVAL_DIR = "data/eval"
OUTPUT_PATH = "data/val_dataset.tsv"

data = []

for fname in sorted(os.listdir(EVAL_DIR)):
    if fname.endswith("_fixed.wav"):
        base = fname.replace("_fixed.wav", "")
        txt_file = os.path.join(EVAL_DIR, f"{base}.txt")
        wav_file = os.path.join(EVAL_DIR, fname)

        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
            data.append({"file": wav_file, "transcription": transcription})
        else:
            print(f"⚠️ No se encontró: {txt_file}")

# Guardar en TSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"[✅] Archivo generado: {OUTPUT_PATH} con {len(df)} ejemplos")
