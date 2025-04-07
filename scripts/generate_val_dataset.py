# scripts/generate_val_dataset.py
import os

real_dir = "data/eval"
hyp_dir = "results/batch_transcriptions"
logits_dir = "outputs/logits"
output_tsv = "data/val_dataset.tsv"

with open(output_tsv, "w", encoding="utf-8") as out:
    out.write("file\ttranscription\n")
    for file in sorted(os.listdir(real_dir)):
        if not file.endswith(".txt"):
            continue
        base = os.path.splitext(file)[0]
        real_path = os.path.join(real_dir, file)
        logits_path = os.path.join(logits_dir, base + ".npy")
        if not os.path.exists(logits_path):
            print(f"⚠️ Falta logits para {base}, salteando")
            continue
        with open(real_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip().lower()
        out.write(f"{logits_path}\t{transcript}\n")

print(f"✅ Archivo generado: {output_tsv}")
