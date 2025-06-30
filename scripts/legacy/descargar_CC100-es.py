from datasets import load_dataset

# Cargar un subconjunto razonable (ajusta el porcentaje si tienes más recursos)
dataset = load_dataset("cc100", "es", split="train[:10%]")

# Filtrar y guardar solo líneas útiles (ni muy cortas ni basura)
with open("cc100_es_clean.txt", "w", encoding="utf-8") as f:
    for row in dataset:
        text = row["text"].strip()
        if 15 < len(text) < 300 and any(p in text for p in ".?!"):
            text = text.replace("\n", " ").replace("\t", " ").strip()
            f.write(text + "\n")
